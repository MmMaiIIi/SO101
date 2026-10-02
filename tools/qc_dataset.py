#!/usr/bin/env python
"""Read-only LeRobot v3 QC. Usage: python tools/qc_dataset.py DATASET --report.

Artifacts are outside raw dataset. Missing fields/errors are reported, not repaired.
"""
import argparse
import csv
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from qc_signal import (parse_task, finite_values, timestamp_check, normalized_motion,
                       joint_stats, boundary_stats, length_outliers)
from qc_video import scan_file, analyze_clip, contact_sheet, combined_video
from plots import lines_svg
from generate_qc_report import generate, write_csv, manual_csv

LOG=logging.getLogger('so101.qc')


def host_camera_telemetry(root):
    """Map new collector capture-update sidecars to saved episodes, excluding discarded attempts."""
    folder=root/'telemetry'
    if folder.is_dir():
        return parquet_timing_telemetry(folder), []
    events_path=root/'events.jsonl'; frames_path=root/'frame_quality.jsonl'
    if not events_path.is_file() or not frames_path.is_file():
        return {}, []
    result={}; errors=[]
    try:
        events=[json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        attempts={event['attempt']:event['episode'] for event in events if event.get('event')=='save'}
        samples={episode:[] for episode in attempts.values()}
        with frames_path.open() as stream:
            for line in stream:
                frame=json.loads(line)
                if frame.get('attempt') in attempts:
                    samples[attempts[frame['attempt']]].append(frame)
        for episode, frames in samples.items():
            result[episode]={'sample_count':len(frames),'clock':'host capture/postprocess update; not sensor exposure','cameras':{}}
            for name in ('front','wrist'):
                stamps=[f.get('cameras',{}).get(name,{}).get('camera_last_update_monotonic_ns') for f in frames]
                ages=[f.get('cameras',{}).get(name,{}).get('frame_age_ms') for f in frames]
                known_ages=[age for age in ages if age is not None]
                unique=sum(stamp is not None and (i==0 or stamp!=stamps[i-1]) for i,stamp in enumerate(stamps))
                result[episode]['cameras'][name]={'unique_update_ratio':unique/max(len(stamps),1),
                    'unknown_timestamp_frames':sum(s is None for s in stamps),
                    'p95_frame_age_ms':float(np.percentile(known_ages,95)) if known_ages else None,
                    'max_frame_age_ms':max(known_ages) if known_ages else None}
    except Exception as exc:
        errors.append(f'Optional collector telemetry unreadable: {exc}')
    return result, errors


def parquet_timing_telemetry(folder):
    """Validate saved-frame host timing and summarize it without sensor-sync claims."""
    import re
    result={}
    def stats(values):
        finite=[float(value) for value in values if value is not None and np.isfinite(value)]
        return {'p50':float(np.percentile(finite,50)),'p95':float(np.percentile(finite,95)),'max':max(finite)} if finite else {'p50':None,'p95':None,'max':None}
    for path in sorted(folder.glob('episode_*.parquet')):
        match=re.fullmatch(r'episode_(\d+)\.parquet',path.name)
        if not match:
            continue
        index=int(match[1]); entry={'sample_count':0,'errors':[],'warnings':[],
            'clock':'time.monotonic_ns host camera update timestamp != sensor exposure timestamp',
            'age_reference':'tick_monotonic_ns = observation/action snapshot after send; dataset write age separate',
            'cameras':{},'path':str(path)}
        result[index]=entry
        try:
            table=pq.read_table(path); rows=table.to_pylist(); entry['sample_count']=len(rows)
            required={'episode_index','frame_index','tick_monotonic_ns','state_timestamp_ns','action_send_timestamp_ns','control_dt_ms'}
            for name in ('front','wrist'):
                required.update(f'{name}_{suffix}' for suffix in ('frame_seq','update_monotonic_ns','age_ms','is_new'))
            missing=required-set(table.column_names)
            if missing:
                raise ValueError(f'Missing timing fields: {sorted(missing)}')
            if [r['frame_index'] for r in rows]!=list(range(len(rows))) or any(r['episode_index']!=index for r in rows):
                entry['errors'].append('Timing episode_index/frame_index mismatch')
            ticks=[r['tick_monotonic_ns'] for r in rows]
            if any(t is None for t in ticks) or any(b<=a for a,b in zip(ticks,ticks[1:])):
                entry['errors'].append('Timing ticks must be strictly monotonic')
            for camera in ('front','wrist'):
                previous=None; flags=[]
                for row in rows:
                    seq=row[f'{camera}_frame_seq']; update=row[f'{camera}_update_monotonic_ns']; age=row[f'{camera}_age_ms']
                    expected=seq is not None and seq!=previous
                    flags.append(expected)
                    if bool(row[f'{camera}_is_new'])!=expected:
                        entry['errors'].append(f'{camera} is_new / sequence mismatch')
                    if update is not None:
                        expected_age=(row['tick_monotonic_ns']-update)/1e6
                        if age is None or not np.isfinite(age) or abs(expected_age-age)>1e-6 or expected_age<0:
                            entry['errors'].append(f'{camera} age / timestamp mismatch')
                    previous=seq
                unknown=sum(r[f'{camera}_frame_seq'] is None or r[f'{camera}_update_monotonic_ns'] is None for r in rows)
                if unknown:
                    entry['warnings'].append(f'{camera}: {unknown} unknown host camera timestamps')
                entry['cameras'][camera]={'fresh_ratio':sum(flags)/max(len(rows),1),
                    'age_ms':stats([r[f'{camera}_age_ms'] for r in rows]),'unknown_timestamp_frames':unknown,
                    'dataset_write_age_ms':stats([r.get(f'{camera}_dataset_write_age_ms') for r in rows])}
            entry['control_dt_ms']=stats([r['control_dt_ms'] for r in rows])
            entry['state_to_action_ms']=stats([(r['action_send_timestamp_ns']-r['state_timestamp_ns'])/1e6 for r in rows if r['state_timestamp_ns'] is not None and r['action_send_timestamp_ns'] is not None])
            if any(r['state_timestamp_ns'] is None or r['action_send_timestamp_ns'] is None for r in rows):
                entry['warnings'].append('Unknown state/action host timing')
            for row in rows:
                stamps=[row.get(key) for key in ('state_timestamp_ns','action_read_done_ns','action_send_timestamp_ns','tick_monotonic_ns','dataset_write_monotonic_ns')]
                known=[stamp for stamp in stamps if stamp is not None]
                if any(b<a for a,b in zip(known,known[1:])):
                    entry['errors'].append('Host state/action/snapshot/write timing order invalid')
            entry['errors']=sorted(set(entry['errors']))
        except Exception as exc:
            entry['errors'].append(f'Timing sidecar unreadable: {exc}')
    return result


def source_manifest(root):
    """Hash every original file; reruns verify QC never changes raw bytes or adds files."""
    result={}
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        digest=hashlib.sha256()
        with path.open('rb') as f:
            while data:=f.read(4*1024*1024):
                digest.update(data)
        result[str(path.relative_to(root))]={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
    return result


def episode_summary(ep):
    """Flat per-episode summary; unknown camera/signal metrics remain empty."""
    row={k:ep.get(k) for k in ('episode','task','task_index','frames','duration_s','fps','state_dim','action_dim','auto_status','recommendation')}
    row['decode_ok']=all(c.get('decode_ok',False) and c.get('frame_count_matches',False) for c in ep.get('cameras',{}).values()) and len(ep.get('cameras',{}))==2
    row['warnings']='; '.join(ep.get('warnings',[])); row['errors']='; '.join(ep.get('errors',[]))
    for name in ('front','wrist'):
        stats=ep.get('cameras',{}).get(name,{})
        row[f'{name}_frames']=stats.get('decoded_frames')
        for metric in ('exact_duplicate_ratio','near_duplicate_ratio','longest_exact_duplicate_run_frames','longest_near_static_run_frames','moving_but_same_frame_ratio'):
            row[f'{name}_{metric}']=stats.get(metric)
    row['joint_lags_frames']=json.dumps({j:s.get('lag',{}).get('best_lag_frames') for j,s in ep.get('joints',{}).items()})
    row['joint_RMSE']=json.dumps({j:s.get('lag',{}).get('RMSE') for j,s in ep.get('joints',{}).items()})
    timing=ep.get('host_camera_telemetry') or {}
    for camera in ('front','wrist'):
        metrics=timing.get('cameras',{}).get(camera,{})
        row[f'{camera}_fresh_ratio']=metrics.get('fresh_ratio')
        for statistic in ('p50','p95','max'):
            row[f'{camera}_age_{statistic}_ms']=metrics.get('age_ms',{}).get(statistic)
    for metric in ('control_dt_ms','state_to_action_ms'):
        for statistic in ('p50','p95','max'):
            row[f'{metric}_{statistic}']=timing.get(metric,{}).get(statistic)
    row['start_idle_s']=ep.get('boundary',{}).get('start_idle_s')
    row['expected_start_context_s']=ep.get('boundary',{}).get('expected_start_context_s')
    return row


def vector_column(table,name,errors):
    """Read numeric vectors from official Parquet columns, reporting ragged/missing fields."""
    try:
        values=np.asarray(table[name].to_pylist(),dtype=float)
        if values.ndim!=2:
            raise ValueError('expected NxD vectors')
        return values
    except Exception as exc:
        errors.append(f'{name}: {exc}')
        return np.empty((len(table),0))


def run_qc(root,output,width=640,height=480,make_review=True):
    """Run all repeatable integrity/signal/video checks through official v3 metadata paths."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    import av
    av.logging.set_level(av.logging.ERROR)
    root=root.resolve(); output=output.resolve()
    if output==root or root in output.parents:
        raise ValueError('QC output must be outside immutable dataset root')
    output.mkdir(parents=True,exist_ok=True)
    for folder in ('plots','contact_sheets','review','timeseries'):
        (output/folder).mkdir(exist_ok=True)
    before=source_manifest(root)
    report={'dataset_root':str(root),'output':str(output),'metric_version':'1.0',
            'episodes':[],'global_errors':[],'method_notes':[
                'LeRobotDatasetMetadata supplies schema, task mapping, Parquet paths and shared-video intervals.',
                'Encoded PTS and pixels cannot recover sensor exposure times. Optional host-update sidecars are reported separately.',
                'RMSE lag is command vs later measured state, not image/state sensor alignment proof.',
                'No automatic task-success or RESET-leak classification is attempted.',
                'All thresholds are descriptive heuristics; no invented motor safety limits.']}
    try:
        meta=LeRobotDatasetMetadata('local/qc',root=root)
        report['dataset_info']=json.loads((root/'meta'/'info.json').read_text())
        telemetry, telemetry_errors=host_camera_telemetry(root)
        report['optional_telemetry_warnings']=telemetry_errors
        if meta.info.get('codebase_version')!='v3.0':
            raise ValueError('This CLI supports LeRobot v3.0 only')
        fps=meta.fps
        task_map={int(row['task_index']):str(task) for task,row in meta.tasks.iterrows()}
        episodes=[meta.episodes[i] for i in range(len(meta.episodes))]
        report['episode_metadata']=episodes
        if (root/'telemetry').is_dir():
            declared={int(md['episode_index']) for md in episodes}
            for extra in sorted(set(telemetry)-declared):
                report['global_errors'].append(f'Timing sidecar has undeclared episode {extra}')
        tables={}
        for path in sorted((root/'data').rglob('*.parquet')):
            try:
                tables[path]=pq.read_table(path)
            except Exception as exc:
                report['global_errors'].append(f'Unreadable Parquet {path}: {exc}')
        for path in sorted((root/'meta').rglob('*.parquet')):
            try:
                pq.read_table(path)
            except Exception as exc:
                report['global_errors'].append(f'Unreadable metadata {path}: {exc}')
        actual_total=sum(len(t) for t in tables.values())
        if actual_total!=meta.info['total_frames']:
            report['global_errors'].append(f'Global frame mismatch: parquet={actual_total}, metadata={meta.info["total_frames"]}')
        if len(episodes)!=meta.info['total_episodes']:
            report['global_errors'].append('Episode metadata count mismatch')
        camera_files={}
        expected_per_file=Counter()
        for md in episodes:
            for camera in ('front','wrist'):
                key=f'observation.images.{camera}'
                try:
                    path=root/meta.get_video_file_path(md['episode_index'],key)
                    expected_per_file[str(path)]+=md['length']
                except Exception:
                    pass
        for path in sorted((root/'videos').rglob('*.mp4')):
            LOG.info('Full source decode: %s',path)
            stats=scan_file(path); camera_files[str(path)]=stats
            expected=expected_per_file.get(str(path))
            if expected is not None and stats['decoded_frames']!=expected:
                report['global_errors'].append(f'Video global count mismatch: {path} decoded={stats["decoded_frames"]} expected={expected}')
        report['source_videos']=camera_files
        lengths=length_outliers([md['length']/fps for md in episodes])
        report['length_distribution']=lengths
        names=meta.features.get('action',{}).get('names',[])
        for position,md in enumerate(episodes):
            idx=int(md['episode_index']); prefix=f'episode_{idx:03d}'
            LOG.info('Episode %d / %d',idx,len(episodes))
            ep={'episode':idx,'fps':fps,'frames':0,'duration_s':0.,'task':'unknown','task_index':[],
                'errors':[],'warnings':[],'cameras':{},'joints':{}}
            report['episodes'].append(ep)
            ep['host_camera_telemetry']=telemetry.get(idx)
            try:
                path=root/meta.get_data_file_path(idx)
                table=tables[path]
                import pyarrow.compute as pc
                table=table.filter(pc.equal(table['episode_index'],idx))
                table=table.take(pc.sort_indices(table,sort_keys=[('frame_index','ascending')]))
                n=len(table); ep['frames']=n; ep['duration_s']=n/fps
                timing_metrics=ep.get('host_camera_telemetry')
                if (root/'telemetry').is_dir():
                    if timing_metrics is None:
                        ep['errors'].append('Missing episode timing sidecar')
                    else:
                        ep['errors'].extend(timing_metrics.get('errors',[]))
                        ep['warnings'].extend(timing_metrics.get('warnings',[]))
                        if timing_metrics['sample_count']!=n:
                            ep['errors'].append('Timing sidecar / dataset frame count mismatch')
                if n!=md['length'] or n!=md['dataset_to_index']-md['dataset_from_index']:
                    ep['errors'].append('Parquet / metadata episode length mismatch')
                if not n:
                    raise ValueError('Empty episode')
                frame_index=np.asarray(table['frame_index'])
                if not np.array_equal(frame_index,np.arange(n)):
                    ep['errors'].append('Non-contiguous frame_index')
                global_index=np.asarray(table['index'])
                if not np.array_equal(global_index,np.arange(md['dataset_from_index'],md['dataset_to_index'])):
                    ep['errors'].append('Global index / metadata range mismatch')
                timestamp=np.asarray(table['timestamp'],float)
                ep['timestamps']=timestamp_check(timestamp,fps)
                if not all(ep['timestamps'][k] for k in ('finite','strictly_monotonic','grid_ok')):
                    ep['errors'].append('Invalid timestamp sequence')
                task_indices=sorted(set(int(i) for i in table['task_index'].to_pylist()))
                ep['task_index']=task_indices
                tasks=[task_map.get(i,f'UNKNOWN task_index {i}') for i in task_indices]
                ep['task']=' | '.join(tasks)
                if len(tasks)!=1 or not parse_task(tasks[0]):
                    ep['errors'].append('Task semantic or single-instruction violation')
                if tasks!=list(md['tasks']):
                    ep['errors'].append('Frame task_index / episode tasks mismatch')
                state=vector_column(table,'observation.state',ep['errors'])
                action=vector_column(table,'action',ep['errors'])
                ep['state_dim']=state.shape[1]; ep['action_dim']=action.shape[1]
                for name,values in (('state',state),('action',action)):
                    ep[f'{name}_finite']=finite_values(values)
                    if not ep[f'{name}_finite']['ok']:
                        ep['errors'].append(f'NaN/Inf in {name}')
                    feature='observation.state' if name=='state' else 'action'
                    if values.shape[1:]!=tuple(meta.features[feature]['shape']):
                        ep['errors'].append(f'{name} dimensions differ from declared schema')
                if action.shape!=state.shape or not state.shape[1]:
                    raise ValueError('Incompatible state/action dimensions')
                if not np.isfinite(action).all() or not np.isfinite(state).all():
                    raise ValueError('Nonfinite control values prevent signal/video-motion analysis')
                motion,scales=normalized_motion(action,state)
                ep['motion_scales']=scales.tolist()
                joint_names=names if len(names)==state.shape[1] else [f'joint_{i}' for i in range(state.shape[1])]
                ep['joints']=joint_stats(action,state,joint_names,fps)
                ep['boundary']=boundary_stats(motion,fps)
                ep['boundary']['expected_start_context_s']=[0.3,0.5] if (root/'telemetry').is_dir() else None
                ep['boundary']['comparison_note']='New ARMED pre-roll aims for approximately 0.3–0.5 s context; old batch start idle was approximately 1.1–1.7 s. Success requires real-data comparison, not zero idle.'
                ep['length_outlier']=lengths['episodes'][position]
                if ep['length_outlier']['flag']:
                    ep['warnings'].append(ep['length_outlier']['flag'])
                for joint,stats in ep['joints'].items():
                    for signal in ('action','state'):
                        if stats.get(signal,{}).get('jump_candidate_frames'):
                            ep['warnings'].append(f'{joint} {signal} discontinuity candidate')
                if ep['boundary']['start_idle_s']>1 or ep['boundary']['end_idle_s']>2:
                    ep['warnings'].append('Boundary idle padding: inspect trim candidates')
                if ep['boundary']['last_frame_moving'] and ep['boundary']['tail_moving_fraction']>.5:
                    ep['warnings'].append('Motion continues through final 2s: inspect boundary, not proof of RESET')
                signal_panels=[(joint,[('actual sent action',timestamp,action[:,j]),('measured state',timestamp,state[:,j])]) for j,joint in enumerate(joint_names)]
                lines_svg(output/'plots'/f'{prefix}_signals.svg',signal_panels,f'Episode {idx}: joint position vs time (s); gripper 0–100, others per schema')
                lag_panels=[]
                for joint,stats in ep['joints'].items():
                    curves=stats.get('lag',{}).get('curve',[])
                    lag_panels.append((joint+' RMSE vs lag frames',[('RMSE',[r['lag'] for r in curves],[r['rmse'] for r in curves])]))
                lines_svg(output/'plots'/f'{prefix}_lag.svg',lag_panels,f'Episode {idx}: action[t] versus state[t+lag]')
                boundary_panels=[]
                for title,mask in (('first 1 s',timestamp<=1),('last 2 s',timestamp>=timestamp[-1]-2)):
                    boundary_panels.append((title+' normalized joint motion',[('motion',timestamp[mask],motion[mask])]))
                lines_svg(output/'plots'/f'{prefix}_boundary.svg',boundary_panels,f'Episode {idx}: boundaries; no automatic success/RESET judgment')
                samples={}; camera_series={}
                for camera in ('front','wrist'):
                    key=f'observation.images.{camera}'
                    try:
                        video_path=root/meta.get_video_file_path(idx,key)
                        start=float(md[f'videos/{key}/from_timestamp']); end=float(md[f'videos/{key}/to_timestamp'])
                        if abs((end-start)-n/fps)>1/fps:
                            ep['errors'].append(f'{camera} video interval length mismatch')
                        stats,series,images=analyze_clip(video_path,start,end,n,motion,(height,width,3))
                        ep['cameras'][camera]=stats; camera_series[camera]=series; samples[camera]=images
                        for flag in ('decode_ok','frame_count_matches','shape_ok','dtype_ok','pts_monotonic'):
                            if not stats.get(flag):
                                ep['errors'].append(f'{camera}: {flag} failed')
                        full=camera_files.get(str(video_path),{})
                        if not full.get('decode_ok') or full.get('corrupt_frames',0) or not full.get('pts_monotonic',False):
                            ep['errors'].append(f'{camera} source file decode/corruption/PTS failure')
                        if stats.get('moving_but_same_frame_ratio',0)>.05:
                            ep['warnings'].append(f'{camera}: repeated-image proxy during motion; review sensor freshness')
                    except Exception as exc:
                        ep['errors'].append(f'{camera} unavailable: {exc}')
                contact_sheet(output/'contact_sheets'/f'{prefix}.png',samples,idx,ep['task'])
                camera_panels=[]
                for camera,series in camera_series.items():
                    t=np.arange(len(series['pixel_MAE']))/fps
                    camera_panels.extend([(camera+' decoded pixel MAE',[('MAE',t,series['pixel_MAE'])]),
                                          (camera+' unchanged proxies',[('exact RGB',t,series['exact_same']),('compact fingerprint',t,series['compact_same'])])])
                camera_panels.append(('normalized robot motion',[('motion',timestamp,motion)]))
                lines_svg(output/'plots'/f'{prefix}_camera.svg',camera_panels,f'Episode {idx}: image content proxies (no raw camera clock)')
                rows=[]
                for i in range(n):
                    row={'frame':i,'timestamp':float(timestamp[i]),'normalized_motion':float(motion[i])}
                    for camera,series in camera_series.items():
                        for metric,values in series.items():
                            row[f'{camera}_{metric}']=values[i] if i<len(values) else ''
                    rows.append(row)
                write_csv(output/'timeseries'/f'{prefix}.csv',rows)
                if make_review and len(ep['cameras'])==2 and not ep['errors']:
                    try:
                        count=combined_video(output/'review'/f'{prefix}_combined.mp4',ep['cameras'],idx,ep['task'],state,fps)
                        ep['review_video_frames']=count
                    except Exception as exc:
                        ep['warnings'].append(f'Review video asset error: {exc}')
            except Exception as exc:
                ep['errors'].append(f'Episode processing: {type(exc).__name__}: {exc}')
            if report['global_errors']:
                ep['errors'].append('Dataset global integrity error; see report')
            ep['auto_status']='FAIL' if ep['errors'] else 'WARN' if ep['warnings'] else 'PASS'
            ep['recommendation']='REVIEW' if ep['auto_status']!='PASS' else 'KEEP pending manual stacking-success review'
        expected={f'Stack the {a} block on top of the {b} block.' for a in ('red','blue','yellow') for b in ('red','blue','yellow') if a!=b}
        counts=Counter(ep['task'] for ep in report['episodes'])
        report['task_coverage']={'counts':dict(counts),'missing':sorted(expected-set(counts)),
                                 'unexpected':sorted(set(counts)-expected),'all_six_covered':expected==set(counts)}
    except Exception as exc:
        report['global_errors'].append(f'Dataset load: {type(exc).__name__}: {exc}')
    after=source_manifest(root)
    report['raw_dataset_unchanged']=before==after
    if before!=after:
        report['global_errors'].append('Raw dataset changed during QC; inspect external writer activity')
    report['source_manifest']=before
    report['status_counts']=dict(Counter(e['auto_status'] for e in report['episodes']))
    (output/'qc_report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),allow_nan=False))
    write_csv(output/'episode_summary.csv',[episode_summary(e) for e in report['episodes']])
    manual_csv(output/'manual_review.csv',report['episodes'])
    generate(report,output)
    LOG.info('Report: %s; statuses %s; immutable=%s',output/'report.html',report['status_counts'],report['raw_dataset_unchanged'])
    return report


def review_csv(output):
    """Keyboard CSV labels; preserves raw episodes and keeps labels in a sidecar only."""
    path=output/'manual_review.csv'
    with path.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    index=0
    while rows:
        row=rows[index]
        print(f"\nEpisode {row['episode']} {row['task']} auto={row['auto_status']} manual={row['manual_status']}\n{output/'review.html'}")
        key=input('k KEEP / r REVIEW / d DROP / n next / p previous / q quit: ').strip().lower()
        if key=='q':
            break
        if key in ('k','r','d'):
            row['manual_status']={'k':'KEEP','r':'REVIEW','d':'DROP'}[key]
            row['reason']=input('reason (optional): '); row['notes']=input('notes (optional): ')
            write_csv(path,rows)
        index=max(0,index-1) if key=='p' else min(len(rows)-1,index+1) if key in ('n','k','r','d') else index


def main():
    import sys
    reviewing=len(sys.argv)>1 and sys.argv[1]=='review'
    argv=sys.argv[2:] if reviewing else sys.argv[1:]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--report',action='store_true',help='generate report (default)')
    parser.add_argument('--no-review-video',action='store_true')
    parser.add_argument('--width',type=int,default=640); parser.add_argument('--height',type=int,default=480)
    args=parser.parse_args(argv)
    output=args.output or Path(__file__).resolve().parents[1]/'qc'/args.dataset.name
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    if reviewing:
        review_csv(output)
    else:
        report=run_qc(args.dataset,output,args.width,args.height,not args.no_review_video)
        if report['global_errors'] or any(e['auto_status']=='FAIL' for e in report['episodes']):
            raise SystemExit(2)


if __name__=='__main__':
    main()
