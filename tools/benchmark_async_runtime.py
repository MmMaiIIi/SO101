"""Paced virtual robot: recorded observations, real localhost GPU, 1000 control ticks, zero motor I/O."""
import argparse,json
from pathlib import Path
import numpy as np
from deployment.pi05_stack2.async_runtime import AsyncRuntime
from deployment.pi05_stack2.action_adapter import JointBounds
from deployment.pi05_stack2.protocol import RemotePolicy


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset-root',type=Path,required=True)
    ap.add_argument('--server',default='http://127.0.0.1:8765')
    ap.add_argument('--ticks',type=int,default=1000)
    ap.add_argument('--profile',type=Path,default=Path('artifacts/safety_profile_v2.json'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/async_1000'))
    args=ap.parse_args()
    import torch
    torch.set_num_threads(1)  # Dataset CPU conversion only; inference runs in the separate GPU server.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from deployment.pi05_stack2.offline import _as_rgb_hwc
    ds=LeRobotDataset(repo_id='local/stack2-contract',root=args.dataset_root,video_backend='pyav')
    eps=np.asarray(ds.hf_dataset['episode_index']); values,counts=np.unique(eps,return_counts=True)
    ep=int(values[np.argmax(counts)]); indices=np.flatnonzero(eps==ep)
    observations=[]
    for i in indices[:args.ticks+30]:
        d=ds[int(i)]
        observations.append({'observation.state':np.asarray(d['observation.state'],np.float32),
             'observation.images.front':_as_rgb_hwc(d['observation.images.front']),
             'observation.images.wrist':_as_rgb_hwc(d['observation.images.wrist'])})
    instruction=str(ds[int(indices[0])]['task'])
    print(f'Preloaded episode {ep}: {len(observations)} recorded observations',flush=True)
    profile=json.loads(args.profile.read_text())
    bounds=JointBounds(tuple(profile['lower']),tuple(profile['upper']),tuple(profile['max_delta_per_step']),tuple(profile['max_velocity_per_s']))
    client=RemotePolicy(args.server,timeout_s=2,transport='binary')
    def infer(obs):
        a,_,r=client.infer_chunk(obs,instruction,full_horizon=True)
        return a,r['model_trace']
    infer(observations[0])  # Warmup completed before starting the measured control clock.
    runtime=AsyncRuntime(infer,lambda tick:observations[min(tick,len(observations)-1)],bounds,
                         motor_write_enabled=False,shadow_safety=True)
    try: result=runtime.run(args.ticks)
    finally: client.close()
    args.output.mkdir(parents=True,exist_ok=True)
    for key in ('ticks','requests','first_executions','rejections'):
        (args.output/(key+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in result[key]))
    report=result['summary']
    report['source_episode']=ep; report['recorded_frames']=len(observations)
    report['source']='prerecorded open-loop observations; shadow safety rejects unsafe predictions; not closed-loop robot success'
    report['virtual_clock']='30 Hz scheduled ticks paced by host monotonic time; real GPU wall latency is not simulated'
    starts=[r['selected_start_index'] for r in result['first_executions']]
    available=[min(10,50-i)*1000/30 for i in starts]
    report['selected_start_index_counts']={str(i):starts.count(i) for i in sorted(set(starts))}
    report['available_chunk_horizon_ms_min']=min(available) if available else 0
    report['acceptance']={
        '1000_ticks':report['control_ticks']>=1000,
        'overlap':report['control_ticks_overlapping_inference']>0,
        'independent_threads':report['control_thread_ident']!=report['inference_thread_ident'],
        'p99_inference_below_available_horizon':bool(available) and report['inference_latency_ms']['P99']<min(available),
        'no_stale_replay':report['stale_replays']==0,'no_sequence_regression':report['sequence_regressions']==0,
        'no_watchdog_stop':report['watchdog_activations']==0,'zero_motor_writes':report['motor_commands_sent']==0}
    report['SOFTWARE_RUNTIME_READY']=all(report['acceptance'].values())
    report['hardware_safety_status']='HARDWARE_SAFETY_UNVERIFIED'
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
