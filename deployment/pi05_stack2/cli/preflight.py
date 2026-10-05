"""Actual read-only probes. Missing physical evidence never produces a PASS."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from deployment.pi05_stack2.config import ACTION_ORDER
from deployment.pi05_stack2.safety import Watchdog


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_motor_gate(report_path, devices, calibration, profile):
    """Require fresh, content-bound full preflight evidence before constructing a writable bus."""
    if report_path is None: raise ValueError('Motor gate: --preflight-report is required')
    report=json.loads(Path(report_path).read_text())
    age=time.time()-report.get('timestamp_unix_s',0)
    if report.get('status')!='PASS' or not 0<=age<=900:
        raise ValueError('Motor gate: full preflight must PASS within 15 minutes')
    expected={'devices':digest(devices),'calibration':digest(calibration),'safety_profile':digest(profile)}
    if report.get('artifact_hashes')!=expected: raise ValueError('Motor gate: preflight artifact mismatch')
    for key in ('calibration','camera_mapping','joint_mapping','contract','safety','state_distribution'):
        if report.get('gates',{}).get(key)!='PASS': raise ValueError('Motor gate: unresolved '+key)
    if json.loads(Path(profile).read_text()).get('software_gate')!='PASS':
        raise ValueError('Motor gate: GT oracle/software safety gate unresolved')
    p=json.loads(Path(profile).read_text())
    if p.get('hardware_safety_status')!='VERIFIED' or p.get('tracking_error_limit') is None:
        raise ValueError('Motor gate: physical safety and tracking-error limits unverified')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    modes=ap.add_mutually_exclusive_group()
    for mode in ('all','robot-only','camera-only'): modes.add_argument('--'+mode,action='store_true')
    ap.add_argument('--config',type=Path,default=Path('deployment-readonly.yaml'))
    ap.add_argument('--devices-json',type=Path)
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--safety-profile',type=Path)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--mapping-attestation',type=Path)
    ap.add_argument('--output',type=Path,default=Path('artifacts/preflight'))
    args=ap.parse_args()
    cfg={}
    if args.config.exists():
        import yaml
        cfg=yaml.safe_load(args.config.read_text())
    devices_path=args.devices_json or cfg.get('robot',{}).get('devices_json')
    calibration=args.calibration or cfg.get('robot',{}).get('calibration')
    profile_path=args.safety_profile or cfg.get('safety',{}).get('profile')
    checkpoint=args.checkpoint or cfg.get('checkpoint',{}).get('local_path')
    if not all((devices_path,calibration,profile_path)):
        raise SystemExit('Read-only preflight needs config robot.devices_json, robot.calibration, safety.profile (or explicit flags)')
    devices=json.loads(Path(devices_path).read_text()); profile=json.loads(Path(profile_path).read_text())
    report={'timestamp_unix_s':time.time(),'motor_write_enabled':False,'status':'BLOCKED',
            'artifact_hashes':{'devices':digest(devices_path),'calibration':digest(calibration),'safety_profile':digest(profile_path)},
            'gates':{k:'REQUIRES_HARDWARE_VERIFICATION' for k in ('calibration','camera_mapping','joint_mapping','contract','safety','state_distribution')},
            'errors':[]}
    args.output.mkdir(parents=True,exist_ok=True)
    robot=None; captures={}
    try:
        if args.robot_only or args.all or not args.camera_only:
            from deployment.pi05_stack2.hardware import SO101Hardware
            robot=SO101Hardware(devices['follower']['port'],Path(calibration),profile['calibration_sha256'],motor_output=False)
            robot.connect(); state,ts=robot.read_state()
            raw=robot.bus.sync_read('Present_Position',normalize=False)
            Watchdog().check(state_ts=ts,camera_ts=(ts,))
            ranges=np.asarray(profile['state_p1_p50_p99'])
            ood=(state<ranges[0])|(state>ranges[2])
            report['robot']={'joint_order':ACTION_ORDER,'raw_encoder':raw,'converted_position':state.tolist(),
                             'units':['degrees']*5+['0-100'],'timestamp_monotonic':ts,'ood_dimensions':np.flatnonzero(ood).tolist(),
                             'p1_p50_p99':ranges.tolist()}
            if checkpoint:
                import torch
                import lerobot.policies.pi05.processor_pi05
                from lerobot.processor.factory import load_pretrained_policy_processors
                from lerobot.lerobot_types import TransitionKey as K
                pre,_=load_pretrained_policy_processors(Path(checkpoint),preprocessor_overrides={'device_processor':{'device':'cpu'}})
                normalized=pre.steps[3]({K.OBSERVATION:{'observation.state':torch.tensor(state)[None]}})
                report['robot']['normalized_state']=normalized[K.OBSERVATION]['observation.state'][0].tolist()
            else: report['robot']['normalized_state']='UNAVAILABLE_CHECKPOINT_PROCESSORS'
            report['gates']['calibration']='PASS'; report['gates']['state_distribution']='BLOCKED_OOD' if ood.any() else 'PASS'
        if args.camera_only or args.all:
            from deployment.pi05_stack2.camera import CameraCapture
            from PIL import Image
            for name in ('front','wrist'): captures[name]=CameraCapture(devices['cameras'][name])
            time.sleep(1)
            report['cameras']={}
            for name,cam in captures.items():
                seq0=cam.sequence; start=time.monotonic(); time.sleep(.5)
                frame=cam.latest(); now=time.monotonic()
                Watchdog().check(state_ts=now,camera_ts=(frame.timestamp_monotonic,),now=now)
                if frame.image.shape!=(480,640,3) or frame.image.dtype!=np.uint8:
                    raise ValueError('Camera contract error: '+name)
                Image.fromarray(frame.image).save(args.output/('preflight_camera_'+name+'.jpg'))
                report['cameras'][name]={'camera_id':devices['cameras'][name]['path'],'training_feature_key':'observation.images.'+name,
                     'shape':list(frame.image.shape),'fps_observed':(cam.sequence-seq0)/(now-start),
                     'timestamp':frame.timestamp_monotonic,'age_ms':(now-frame.timestamp_monotonic)*1000,
                     'mean':frame.image.mean(axis=(0,1)).tolist(),'std':frame.image.std(axis=(0,1)).tolist(),
                     'rgb_conversion':'OpenCV BGR -> RGB','orientation_mirror_crop':'REQUIRES_HARDWARE_VERIFICATION'}
                reference_path=next(iter(sorted(Path('artifacts/deployment_samples').glob('*.npz'))),None)
                if reference_path:
                    with np.load(reference_path,allow_pickle=False) as data:
                        reference=data[name]
                        Image.fromarray(reference).save(args.output/('reference_camera_'+name+'.jpg'))
                        report['cameras'][name]['recorded_reference']={'shape':list(reference.shape),
                            'mean':reference.mean(axis=(0,1)).tolist(),'std':reference.std(axis=(0,1)).tolist(),
                            'split':'recorded reference; exact training membership unverified'}
            report['gates']['contract']='PASS' if 'robot' in report else 'REQUIRES_ROBOT'
        if args.mapping_attestation:
            att=json.loads(args.mapping_attestation.read_text())
            if att.get('artifact_hashes')!=report['artifact_hashes']: raise ValueError('Mapping attestation does not match artifacts')
            for k in ('camera_mapping','joint_mapping'):
                if att.get(k)=='PASS': report['gates'][k]='PASS'
            report['manual_mapping_evidence']=att
        report['gates']['safety']='PASS' if profile.get('software_gate')=='PASS' and profile.get('hardware_safety_status')=='VERIFIED' else 'HARDWARE_SAFETY_UNVERIFIED'
        if args.all and all(v=='PASS' for v in report['gates'].values()): report['status']='PASS'
    except Exception as exc:
        report['errors'].append({'type':type(exc).__name__,'reason':str(exc)})
    finally:
        for cam in captures.values(): cam.close()
        if robot is not None: robot.close()
        (args.output/'preflight.json').write_text(json.dumps(report,indent=2,default=str))
        print(json.dumps(report,indent=2,default=str))
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__': raise SystemExit(main())
