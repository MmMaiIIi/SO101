from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from deployment.pi05_stack2.action_adapter import JointBounds, filter_action, SafetyRejection, append_rejection, rejection_category
from deployment.pi05_stack2.action_queue import SafeActionQueue
from deployment.pi05_stack2.logging_utils import RunLogger
from deployment.pi05_stack2.observation_adapter import make_observation
from deployment.pi05_stack2.protocol import RemotePolicy
from deployment.pi05_stack2.safety import Watchdog


def stage_action_budget(stage: int, requested_steps: int) -> int:
    return {0: 0, 1: 0, 2: 1, 3: 5, 4: 10, 5: requested_steps}[stage]


def main():
    ap = argparse.ArgumentParser(description="Staged SO101 deployment; dry-run by default")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--observe-only", action="store_true")
    mode.add_argument("--enable-hardware", action="store_true")
    ap.add_argument("--stage", type=int, choices=range(6), default=0)
    ap.add_argument("--server")
    ap.add_argument("--transport",choices=['png','binary'],default='png')
    ap.add_argument("--checkpoint", type=Path, default=Path("models/so101-pi05-4228"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample", type=Path)
    ap.add_argument("--devices-json", type=Path)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--safety-profile", type=Path)
    ap.add_argument("--preflight-report", type=Path)
    ap.add_argument("--instruction", default="Stack the yellow block on top of the blue block.")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--policy-timeout", type=float, default=2.0)
    ap.add_argument("--config",type=Path)
    known,_ = ap.parse_known_args()
    if known.config:
        import yaml
        config = yaml.safe_load(known.config.read_text())
        if config.get("control",{}).get("hz",30) != 30:
            raise SystemExit("Frozen checkpoint runtime requires 30 Hz target execution")
        defaults = {
            "server":config.get("inference",{}).get("server"),
            "checkpoint":config.get("checkpoint",{}).get("local_path","models/so101-pi05-4228"),
            "instruction":config.get("policy",{}).get("language_instruction",ap.get_default("instruction")),
            "devices_json":config.get("robot",{}).get("devices_json"),
            "calibration":config.get("robot",{}).get("calibration"),
            "safety_profile":config.get("safety",{}).get("profile"),
            "policy_timeout":config.get("control",{}).get("policy_timeout_s",2.0),
        }
        ap.set_defaults(**defaults)
    args = ap.parse_args()
    if args.steps < 1: raise SystemExit("--steps must be positive")
    dry = not (args.observe_only or args.enable_hardware)
    motor_output = args.enable_hardware and args.stage >= 2
    if args.observe_only and args.stage: raise SystemExit("--observe-only uses stage 0")
    if dry:
        print("DRY RUN — NO MOTOR COMMANDS WILL BE SENT", flush=True)
    elif not all((args.devices_json, args.calibration, args.safety_profile)):
        raise SystemExit("Real read-only/staged use requires --devices-json, --calibration and --safety-profile")
    runner = None
    inference_trace = {}
    if args.server:
        client = RemotePolicy(args.server, args.policy_timeout,transport=args.transport)
        def infer(obs, *, full_horizon=False):
            nonlocal inference_trace
            inference_trace={}
            actions,latency,result=client.infer_chunk(obs,args.instruction,full_horizon=full_horizon)
            inference_trace=result.get('model_trace',{})
            return actions,latency
    else:
        def infer(obs, *, full_horizon=False):
            nonlocal runner,inference_trace
            inference_trace={}
            if runner is None:
                from deployment.pi05_stack2.model_loader import load_policy
                from deployment.pi05_stack2.policy_runner import PolicyRunner
                policy, pre, post, _ = load_policy(args.checkpoint, args.device)
                runner = PolicyRunner(policy, pre, post, args.device)
            result=runner.infer_chunk(obs | {"task":args.instruction},full_horizon=full_horizon)
            inference_trace=runner.last_trace
            return result
    log, queue = RunLogger(), SafeActionQueue(10)
    (log.path/"config.json").write_text(json.dumps(vars(args),default=str,indent=2)+"\n")
    robot, cameras, executed, previous = None, {}, [], None
    watchdog = Watchdog(policy_timeout_s=args.policy_timeout)
    def checked(target,previous,bounds,index,state,dt_s=1/30):
        context={'observation_state':np.asarray(state).tolist(),
                 'normalized_model_output':inference_trace.get('normalized_model_output',[None]*10)[index],
                 'denormalized_model_output':inference_trace.get('denormalized_model_output',[None]*10)[index]}
        if context['denormalized_model_output'] is not None:
            context['relative_arm_action']=context['denormalized_model_output'][:5]
        return filter_action(target,previous,bounds,context=context,dt_s=dt_s)
    try:
        if dry:
            sample = args.sample or next(iter(sorted(Path("artifacts/deployment_samples").glob("*.npz"))), None)
            if sample is None: raise ValueError("Dry run requires --sample exported from a real trajectory")
            data = np.load(sample, allow_pickle=False)
            obs = make_observation(data["state"], {"observation.images.front":data["front"],
                                                  "observation.images.wrist":data["wrist"]}, args.instruction)
            actions, latency = infer(obs)
            log.log("decoded_chunk",raw_decoded_actions=actions.tolist(),latency_s=latency,motor_commands_sent=0)
            timing='SAFE_TO_PIPELINE' if latency<10/30 else 'QUEUE_STARVATION_EXPECTED'
            log.log('queue_timing',status=timing,queue_horizon_ms=1000/3,total_policy_latency_ms=latency*1000,
                    latency_margin_ms=1000/3-latency*1000,mode='synchronous_no_overlap')
            print(timing+f' horizon=333.33ms policy={latency*1000:.1f}ms',flush=True)
            profile_path = args.safety_profile or Path("artifacts/safety_profile.json")
            if not profile_path.exists():
                raise ValueError("Dry-run safety verification requires --safety-profile")
            profile = json.loads(profile_path.read_text())
            bounds = JointBounds(tuple(profile["lower"]),tuple(profile["upper"]),tuple(profile["max_delta_per_step"]),tuple(profile['max_velocity_per_s']), None if profile.get('tracking_error_limit') is None else tuple(profile['tracking_error_limit']))
            queue.push_chunk(0,actions)
            last = data["state"]
            filtered_actions = []
            while len(queue):
                last,_ = checked(queue.pop(0),last,bounds,len(filtered_actions),data['state'])
                filtered_actions.append(last.copy())
            log.log("dry_run", raw_decoded_actions=actions.tolist(), latency_s=latency,
                    filtered_actions=np.stack(filtered_actions).tolist(),executed_action=None,motor_commands_sent=0)
            print(f"decoded={actions.shape} latency={latency:.3f}s motors=0", flush=True)
            return 0
        from deployment.pi05_stack2.camera import CameraCapture
        from deployment.pi05_stack2.hardware import SO101Hardware
        profile = json.loads(args.safety_profile.read_text())
        bounds = JointBounds(tuple(profile["lower"]),tuple(profile["upper"]),tuple(profile["max_delta_per_step"]),tuple(profile['max_velocity_per_s']), None if profile.get('tracking_error_limit') is None else tuple(profile['tracking_error_limit']))
        bounds.validate()
        devices = json.loads(args.devices_json.read_text())
        if motor_output:
            from deployment.pi05_stack2.cli.preflight import validate_motor_gate
            validate_motor_gate(args.preflight_report,args.devices_json,args.calibration,args.safety_profile)
        robot = SO101Hardware(devices["follower"]["port"], args.calibration, profile["calibration_sha256"], motor_output=motor_output)
        robot.connect()
        for name in ("front","wrist"): cameras[name] = CameraCapture(devices["cameras"][name])
        time.sleep(1.0)
        last_camera_timestamps=()
        def observe():
            nonlocal last_camera_timestamps
            state, stamp = robot.read_state()
            frames = {"observation.images."+name:camera.latest() for name,camera in cameras.items()}
            last_camera_timestamps=tuple(f.timestamp_monotonic for f in frames.values())
            watchdog.check(state_ts=stamp, camera_ts=tuple(f.timestamp_monotonic for f in frames.values()))
            if np.any(state < np.asarray(bounds.lower)) or np.any(state > np.asarray(bounds.upper)):
                raise ValueError("Current pose lies outside calibrated limits")
            ranges=np.asarray(profile['state_p1_p50_p99'])
            if args.stage >= 1 and np.any((state<ranges[0])|(state>ranges[2])):
                raise ValueError('CONTRACT_ERROR: live state outside reference P1-P99; policy disabled')
            return make_observation(state,frames,args.instruction), state, stamp
        budget = stage_action_budget(args.stage,args.steps)
        if args.stage == 0:
            for _ in range(args.steps):
                _,state,stamp = observe()
                log.log("observe_only",state=state.tolist(),state_timestamp=stamp,motor_commands_sent=0)
                print(f"state={state.tolist()} motors=0",flush=True); time.sleep(1/30)
            return 0
        if motor_output:
            robot.require_write_ready()
            _,state,_=observe()
            print(f'Current pose: {state.tolist()}\nStage {args.stage}: at most {budget} filtered actions. A fresh observation is captured AFTER confirmation.')
            if input('Type EXECUTE to confirm this stage: ').strip()!='EXECUTE': return 0
        from deployment.pi05_stack2.async_runtime import AsyncRuntime
        # Prewarm outside both workers. This prediction is discarded, never queued.
        warm_obs,_,_=observe()
        infer(warm_obs,full_horizon=True)
        def infer_async(obs):
            actions,_=infer(obs,full_horizon=True)
            return actions,dict(inference_trace)
        def send(filtered):
            robot.send_target(filtered)
            executed.append(filtered.copy())
        runtime=AsyncRuntime(infer_async,lambda tick:observe()[0],bounds,
                             motor_write_enabled=motor_output,send_target=send if motor_output else None,
                             shadow_safety=False)
        result=runtime.run(max(100,args.steps+60),action_budget=max(1,budget))
        for name in ('ticks','requests','first_executions','rejections'):
            (log.path/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in result[name]))
        for rejected in result['rejections']: append_rejection(rejected)
        (log.path/'async_summary.json').write_text(json.dumps(result['summary'],indent=2))
        log.log('async_runtime',**result['summary'])
        print(json.dumps(result['summary'],indent=2))
        if result['summary']['stop_reason']:
            raise RuntimeError(result['summary']['stop_reason'])
    except KeyboardInterrupt:
        log.log("emergency_stop",reason="operator Ctrl+C",motor_commands_sent=len(executed))
        print("STOP: queue cleared; no further motor targets",flush=True)
    except Exception as exc:
        report=exc.report if isinstance(exc,SafetyRejection) else {
            'observation_state':inference_trace.get('observation_state'),
            'relative_arm_action':None,'decoded_absolute_arm_target':None,'absolute_gripper_target':None,
            'previous_target':None if previous is None else previous.tolist(),
            'current_target':None,'per_joint_delta':None,'per_joint_velocity':None,'configured_bound':None,
            'unavailable_reason':'Failure before a validated target/safety evaluation',
            'violations':[{'category':rejection_category(exc),
                           'reason':str(exc)}], 'normalized_model_output':inference_trace.get('normalized_model_output'),
            'denormalized_model_output':inference_trace.get('denormalized_model_output')}
        append_rejection(report); append_rejection(report,log.path/'safety_rejections.jsonl')
        log.log("safety_stop",error=str(exc),motor_commands_sent=len(executed)); raise
    finally:
        queue.clear()
        if args.server: client.close()
        for camera in cameras.values(): camera.close()
        if robot is not None: robot.close()
        if executed: np.save(log.path/"actions.npy",np.stack(executed))
        (log.path/"summary.json").write_text(json.dumps({"hardware_enabled":bool(motor_output),
                                                      "motor_commands_sent":len(executed),"stage":args.stage},indent=2)+"\n")
        log.close(); print(f"run_dir={log.path}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
