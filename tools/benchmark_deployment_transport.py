"""Measure real requests. One-way WAN latency is deliberately unknown without clock sync."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from deployment.pi05_stack2.protocol import RemotePolicy
from deployment.pi05_stack2.action_adapter import JointBounds, filter_action, SafetyRejection, append_rejection


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--server',default='http://127.0.0.1:8765')
    ap.add_argument('--samples',type=Path,default=Path('artifacts/deployment_samples'))
    ap.add_argument('--requests',type=int,default=100)
    ap.add_argument('--topology',choices=['localhost','WAN_SSH','LAN'],required=True)
    ap.add_argument('--transport',choices=['png','binary'],default='png')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--safety-profile',type=Path,default=Path('artifacts/safety_profile.json'))
    args=ap.parse_args(); paths=sorted(args.samples.glob('*.npz'))
    if not paths: raise ValueError('No real samples')
    client=RemotePolicy(args.server,10,transport=args.transport); rows=[]
    p=json.loads(args.safety_profile.read_text())
    bounds=JointBounds(tuple(p['lower']),tuple(p['upper']),tuple(p['max_delta_per_step']),tuple(p['max_velocity_per_s']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.with_suffix('.jsonl').open('w') as log:
        for i in range(args.requests+3):
            d=np.load(paths[i%len(paths)],allow_pickle=False)
            obs={'observation.state':d['state'],'observation.images.front':d['front'],'observation.images.wrist':d['wrist']}
            instruction=str(d['instruction']) if 'instruction' in d else 'Stack the yellow block on top of the blue block.'
            actions,elapsed,result=client.infer_chunk(obs,instruction)
            row={'request':i,'warmup':i<3,**result['client_trace']}
            st=result.get('server_trace',{}); mt=result.get('model_trace',{})
            previous=d['state']
            for k,target in enumerate(actions):
                context={'source':'MODEL','sample':paths[i%len(paths)].name,'request':i,'chunk_index':k,
                         'observation_state':d['state'].tolist(),
                         'normalized_model_output':mt.get('normalized_model_output',[None]*10)[k],
                         'denormalized_model_output':mt.get('denormalized_model_output',[None]*10)[k],
                         'relative_arm_action':mt['denormalized_model_output'][k][:5] if mt else None}
                try: filter_action(target,previous,bounds,context=context)
                except SafetyRejection as exc: append_rejection(exc.report,args.output.parent/'safety_rejections.jsonl')
                previous=target
            for k in ('preprocessing_ms','model_ms','postprocessing_ms'): row[k]=mt.get(k)
            row['server_trace']=st
            row['image_decode_ms']=st.get('image_decode_ms')
            service_ms=(st['T6_server_send']-st['T2_server_receive'])*1000 if st else result['latency_s']*1000
            row['transport_residual_ms']=(row['T7_client_receive']-row['T1_serialization_complete'])*1000-service_ms
            # On localhost both processes use the same system monotonic clock. Never subtract cross-host clocks.
            row['network_upload_to_headers_ms']=(st['T2_server_receive']-row['T1_serialization_complete'])*1000 if st and args.topology=='localhost' else None
            row['network_download_plus_server_serialization_ms']=(row['T7_client_receive']-st['T6_server_send'])*1000 if st and args.topology=='localhost' else None
            log.write(json.dumps(row)+'\n'); log.flush()
            if i>=3: rows.append(row)
            if i%20==0: print(f'{args.topology} {i}/{args.requests+3} total={elapsed:.3f}s',flush=True)
    metrics={}
    for key in rows[0]:
        if key.endswith('_ms') and rows[0][key] is not None:
            metrics[key]=dict(zip(['P50','P90','P95','P99'],np.percentile([r[key] for r in rows],[50,90,95,99]).tolist()))
    total=metrics['total_ms']; horizon=1000*10/30
    report={'topology':args.topology,'transport':args.transport,'requests':len(rows),'warmup_requests':3,'metrics_ms':metrics,
            'one_way_network_measurement':'same-host monotonic; upload body read reported separately' if args.topology=='localhost' else 'UNIDENTIFIABLE without synchronized clocks; combined transport residual reported, never RTT/2',
            'queue_horizon_ms':horizon,'latency_margin_ms_p95':horizon-total['P95'],
            'queue_timing':'SAFE_TO_PIPELINE' if total['P99']<horizon else 'QUEUE_STARVATION_EXPECTED',
            'runtime_mode':'synchronous; no overlap implemented, this is necessary timing feasibility only, not proof of continuous execution',
            'hardware_commands_sent':0}
    args.output.write_text(json.dumps(report,indent=2)); client.close(); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
