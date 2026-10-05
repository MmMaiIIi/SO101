"""Frozen Stack2 v1 async hardware runner.

No motor write is possible without the explicit --enable-hardware flag.
Per-robot device, calibration and safety files are supplied locally and are
never copied into the repository.
"""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from deployment.pi05_stack2.action_adapter import JointBounds
from deployment.pi05_stack2.async_runtime import AsyncRuntime
from deployment.pi05_stack2.camera import CameraCapture
from deployment.pi05_stack2.hardware import SO101Hardware
from deployment.pi05_stack2.protocol import RemotePolicy
from deployment.pi05_stack2.safety import Watchdog
from deployment.pi05_stack2.startup_recovery import recover_and_hold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen SO101 Stack2 async policy")
    parser.add_argument("--enable-hardware", action="store_true",
                        help="required to enable real motor writes; otherwise exits safely")
    parser.add_argument("--devices-json", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--safety-profile", type=Path, required=True)
    parser.add_argument("--server", required=True, help="persistent local/LAN/SSH-forwarded GPU endpoint")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--transport", choices=("png",), default="png")
    parser.add_argument("--policy-timeout", type=float, default=15.0)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--max-ticks", type=int, default=1800)
    args = parser.parse_args()
    if not args.enable_hardware:
        raise SystemExit("Motor output disabled. Pass --enable-hardware only for an authorized live rollout.")
    if args.max_ticks < 1 or args.policy_timeout <= 0:
        raise SystemExit("max-ticks and policy-timeout must be positive")

    devices = json.loads(args.devices_json.read_text(encoding="utf-8"))
    profile = json.loads(args.safety_profile.read_text(encoding="utf-8"))
    run_dir = args.output_root / ("real_stack2_" + time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    robot = SO101Hardware(devices["follower"]["port"], args.calibration,
                          profile["calibration_sha256"], motor_output=True)
    route = RemotePolicy(args.server, args.policy_timeout, transport=args.transport)
    cameras: dict[str, CameraCapture] = {}
    stop_recording = threading.Event()
    recorder = None
    video = None
    runtime = None
    previous = None
    state_now = None
    last_state_ts = None
    sent = 0
    result = None
    error = None
    started = None
    record_errors: list[str] = []
    track_bound = np.asarray(profile["max_delta_per_step"], dtype=np.float32)
    bounds = JointBounds(tuple(profile["lower"]), tuple(profile["upper"]),
                         tuple(profile["max_delta_per_step"]),
                         tuple(profile["max_velocity_per_s"]), None)
    watchdog = Watchdog()
    ticks_file = (run_dir / "ticks.jsonl").open("w", buffering=1, encoding="utf-8")
    events_file = (run_dir / "events.jsonl").open("w", buffering=1, encoding="utf-8")

    def event(name: str, **fields) -> None:
        events_file.write(json.dumps({"event": name, "time_monotonic": time.monotonic(), **fields}) + "\n")

    def emergency_stop(*_):
        raise KeyboardInterrupt("Operator emergency stop")

    signal.signal(signal.SIGINT, emergency_stop)
    signal.signal(signal.SIGTERM, emergency_stop)

    def record_video() -> None:
        try:
            epoch = time.monotonic()
            frame_index = 0
            while not stop_recording.is_set():
                front = cameras["front"].latest().image
                wrist = cameras["wrist"].latest().image
                if front.shape != (480, 640, 3) or wrist.shape != (480, 640, 3):
                    raise ValueError("Unexpected camera shape; expected 480x640 RGB")
                image = cv2.cvtColor(np.concatenate((front, wrist), axis=1), cv2.COLOR_RGB2BGR)
                cv2.putText(image, f"{args.instruction} t={time.monotonic()-epoch:.2f}s",
                            (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 1)
                video.write(image)
                frame_index += 1
                stop_recording.wait(max(0, epoch + frame_index / 30 - time.monotonic()))
        except Exception as exc:
            record_errors.append(str(exc))

    def observe(tick: int):
        nonlocal state_now, last_state_ts
        if (run_dir / "STOP").exists():
            raise KeyboardInterrupt("Operator STOP file")
        if record_errors:
            raise RuntimeError("Video capture failed: " + str(record_errors))
        begin = time.monotonic()
        try:
            state, state_stamp = robot.read_state()
        except ConnectionError as exc:
            frames = [camera.latest() for camera in cameras.values()]
            event("state_read_failed_hold", attempts=3, error=str(exc), last_state_ts=last_state_ts,
                  new_target_sent=False)
            if last_state_ts is None:
                raise
            watchdog.check(state_ts=last_state_ts,
                           camera_ts=tuple(frame.timestamp_monotonic for frame in frames))
            return None
        state_now, last_state_ts = state, state_stamp
        if state_stamp - begin > watchdog.state_timeout_s:
            raise TimeoutError("STALE_STATE: serial read delay")
        frames = {name: camera.latest() for name, camera in cameras.items()}
        watchdog.check(state_ts=state_stamp,
                       camera_ts=tuple(frame.timestamp_monotonic for frame in frames.values()))
        if not np.isfinite(state).all():
            raise ValueError("NAN_INF measured state")
        if np.any(state < np.asarray(profile["lower"])) or np.any(state > np.asarray(profile["upper"])):
            raise ValueError("POSITION_LIMIT measured state")
        return {"observation.state": state,
                **{"observation.images." + name: frame.image for name, frame in frames.items()}}

    def send(target) -> None:
        nonlocal previous, sent
        robot.send_target(target)
        previous = target.copy()
        sent += 1
        event("policy_motor_write", target=target.tolist(), state=state_now.tolist())

    def infer(observation):
        actions, latency, response = route.infer_chunk(observation, args.instruction, full_horizon=True)
        event("inference", latency_s=latency, client_trace=response.get("client_trace"),
              server_trace=response.get("server_trace"),
              model_ms=response.get("model_trace", {}).get("model_ms"))
        return actions, response.get("model_trace", {})

    def tick_sink(row):
        if row.get("goal_target") is None and previous is not None:
            row["goal_target"] = previous.tolist()
            row["tracking_error"] = (state_now - previous).tolist()
        ticks_file.write(json.dumps(row) + "\n")

    try:
        robot.connect()
        for name in ("front", "wrist"):
            cameras[name] = CameraCapture(devices["cameras"][name])
        deadline = time.monotonic() + 4
        while True:
            try:
                for camera in cameras.values():
                    camera.latest()
                break
            except TimeoutError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.02)
        video = cv2.VideoWriter(str(run_dir / "front_wrist.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                30, (1280, 480))
        if not video.isOpened():
            raise RuntimeError("Video writer unavailable")
        recorder = threading.Thread(target=record_video, name="so101-video", daemon=True)
        recorder.start()
        previous = recover_and_hold(robot, profile, event)
        observation = observe(0)
        event("start", instruction=args.instruction, state=previous.tolist(),
              percentile_slew_policy="WARNING_ONLY", absolute_limits_unchanged=True,
              transport=args.transport, read_retries=2)
        started = time.monotonic()
        runtime = AsyncRuntime(infer, observe, bounds, motor_write_enabled=True, send_target=send,
                               hold_timeout_s=None, percentile_slew_warning=True,
                               tick_sink=tick_sink, actions_per_chunk=50, chunk_size_threshold=.6)
        result = runtime.run(args.max_ticks)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        event("stop_exception", error=error)
    finally:
        if runtime is not None:
            runtime.stop_event.set()
            with runtime.condition:
                runtime.condition.notify_all()
        reason = error or (result or {}).get("summary", {}).get("stop_reason")
        torque_disabled = {}
        if robot.bus.is_connected and reason:
            for name in robot.bus.motors:
                try:
                    robot.bus.write("Torque_Enable", name, 0, num_retry=1)
                    torque_disabled[name] = robot.bus.read("Torque_Enable", name, normalize=False) == 0
                except Exception as exc:
                    torque_disabled[name] = str(exc)
        stop_recording.set()
        if recorder is not None:
            recorder.join(timeout=2)
        if video is not None:
            video.release()
        for camera in cameras.values():
            camera.close()
        robot.close()
        if runtime is None or not runtime.worker.is_alive():
            route.close()
        if result:
            for key in ("requests", "rejections", "first_executions"):
                (run_dir / (key + ".jsonl")).write_text(
                    "".join(json.dumps(row) + "\n" for row in result[key]), encoding="utf-8")
        summary = {"task_result_requires_visual_review": True,
                   "episode_started": started is not None, "policy_motor_commands": sent,
                   "duration_s": None if started is None else time.monotonic() - started,
                   "reason": reason, "runtime": None if result is None else result["summary"],
                   "torque_disabled_readback": torque_disabled,
                   "video": str(run_dir / "front_wrist.mp4"), "log": str(run_dir),
                   "instruction": args.instruction}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        events_file.close()
        ticks_file.close()
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
