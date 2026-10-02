"""SO101 teleop collector using the installed LeRobot hardware and v3 writer.

The main thread owns all hardware. A separate thread owns LeRobotDataset.
Space/Enter/Backspace never disconnect or reconfigure either arm.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import queue
import random
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from collections import deque
from quality_telemetry import (append_json, camera_snapshot, calibration_action_ranges,
                               instrument_state_read, manifest, timing_summary, write_episode_timing)


COLORS = ("red", "blue", "yellow")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class TaskScheduler:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.tasks = []
        self.index = 0
        self._shuffle()

    def _shuffle(self):
        self.tasks = [(a, b) for a in COLORS for b in COLORS if a != b]
        self.rng.shuffle(self.tasks)
        self.index = 0

    def reveal(self):
        source, target = self.tasks[self.index]
        return f"Stack the {source} block on top of the {target} block."

    def success(self):
        self.index += 1
        if self.index == len(self.tasks):
            self._shuffle()


def make_hardware(config):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    cameras = {
        name: OpenCVCameraConfig(
            index_or_path=c["path"], width=c["width"], height=c["height"],
            fps=c["fps"], fourcc=c["fourcc"],
        ) for name, c in config["cameras"].items()
    }
    robot = SOFollower(SOFollowerRobotConfig(
        port=config["follower"]["port"], id=config["follower"]["id"],
        cameras=cameras, use_degrees=True,
    ))
    leader = SOLeader(SOLeaderTeleopConfig(
        port=config["leader"]["port"], id=config["leader"]["id"], use_degrees=True,
    ))
    return robot, leader


def dataset_features(robot):
    from lerobot.utils.feature_utils import hw_to_dataset_features
    return {
        **hw_to_dataset_features(robot.observation_features, "observation", use_video=True),
        **hw_to_dataset_features(robot.action_features, "action", use_video=True),
    }


def hardware_preflight(robot, leader, max_start_gap=20):
    """Only reads motors. Never accepts or writes a replacement calibration."""
    devices = (leader, robot)
    states = []
    try:
        for device in devices:
            if not device.calibration:
                raise RuntimeError(f"Missing existing calibration: {device.calibration_fpath}")
            device.bus.connect()
            actual = {name: asdict(cal) for name, cal in device.bus.read_calibration().items()}
            saved = {name: asdict(cal) for name, cal in device.calibration.items()}
            if actual != saved:
                raise RuntimeError(f"Calibration/identity mismatch: {device.config.port}; refusing to write calibration")
            states.append(device.bus.sync_read("Present_Position", num_retry=2))
        goals = robot.bus.sync_read("Goal_Position", num_retry=2)
        for joint in JOINTS:
            limit = 20 if joint == "gripper" else max_start_gap
            gap = abs(states[0][joint] - states[1][joint])
            old_goal_gap = abs(goals[joint] - states[1][joint])
            if gap > limit or old_goal_gap > limit:
                raise RuntimeError(
                    f"Startup gap for {joint}: leader/follower={gap:.1f}, "
                    f"existing follower goal/state={old_goal_gap:.1f} (limit {limit}). "
                    "Align the arms manually using your existing procedure before starting."
                )
    finally:
        for device in devices:
            if device.bus.is_connected:
                device.bus.disconnect(disable_torque=False)


class EpisodeWriter:
    """FIFO barriers for save/discard; bounded frames and nonblocking producer."""
    def __init__(self, factory, capacity=30):
        self.jobs = queue.SimpleQueue()
        self.results = queue.SimpleQueue()
        self.slots = threading.BoundedSemaphore(capacity)
        self.ready = threading.Event()
        self.error = None
        self.features = None
        self.saved = 0
        self.last_saved_summary = None
        self.factory = factory
        self.thread = threading.Thread(target=self._run, name="lerobot-writer", daemon=False)
        self.thread.start()
        self.ready.wait()
        if self.error:
            raise RuntimeError(f"Dataset startup failed: {self.error}")

    def _run(self):
        dataset = None
        try:
            dataset = self.factory()
            self.features = dataset.features
            self.root = getattr(dataset, 'root', None)
        except Exception as exc:
            self.error = repr(exc)
        finally:
            self.ready.set()
        if dataset is None:
            return
        timing_rows = []
        pending_frames = 0
        try:
            while True:
                kind, value = self.jobs.get()
                try:
                    if kind == "close":
                        append_json(self.root, 'events.jsonl', {'event': 'stop', 'saved': self.saved, 'host_monotonic_ns': time.monotonic_ns(), 'unfinished_discarded': dataset.has_pending_frames()})
                        if dataset.has_pending_frames():
                            dataset.clear_episode_buffer()
                        break
                    if self.error:
                        continue
                    if kind == "frame":
                        telemetry = value.pop('_collector_qc', None)
                        write_ns = time.monotonic_ns()
                        dataset.add_frame(value)
                        pending_frames += 1
                        if telemetry is not None:
                            telemetry['dataset_write_monotonic_ns'] = write_ns
                            for camera in ('front','wrist'):
                                update_ns = telemetry.get(f'{camera}_update_monotonic_ns')
                                telemetry[f'{camera}_dataset_write_age_ms'] = (write_ns-update_ns)/1e6 if update_ns is not None else None
                            timing_rows.append(telemetry)
                    elif kind == "event":
                        append_json(self.root, 'events.jsonl', value)
                    elif kind == "save":
                        if timing_rows and len(timing_rows) != pending_frames:
                            raise ValueError('Timing sidecar / pending dataset frame mismatch')
                        dataset.save_episode(parallel_encoding=False)
                        if timing_rows:
                            write_episode_timing(self.root, self.saved, timing_rows)
                        if value is not None:
                            value['timing'] = timing_summary(timing_rows)
                        self.last_saved_summary = value
                        append_json(self.root, 'events.jsonl', {'event': 'save', 'episode': self.saved, 'host_monotonic_ns': time.monotonic_ns(), **(value or {})})
                        self.saved += 1
                        timing_rows = []
                        pending_frames = 0
                        self.results.put(("saved", self.saved))
                    elif kind == "discard":
                        dataset.clear_episode_buffer()
                        timing_rows = []
                        pending_frames = 0
                        append_json(self.root, 'events.jsonl', {'event': 'discard', 'host_monotonic_ns': time.monotonic_ns(), **(value or {})})
                        self.results.put(("discarded", None))
                except Exception as exc:
                    self.error = repr(exc)
                    self.results.put(("error", self.error))
                finally:
                    if kind == "frame":
                        self.slots.release()
        finally:
            try:
                dataset.finalize()
            except Exception as exc:
                self.error = repr(exc)
                self.results.put(("error", self.error))

    def frame(self, values, snapshot_owned=False):
        if self.error or not self.slots.acquire(blocking=False):
            return False
        try:
            # Images belong to camera threads; detach the snapshot before enqueueing.
            self.jobs.put(("frame", values if snapshot_owned else copy.deepcopy(values)))
        except BaseException:
            self.slots.release()
            raise
        return True

    def barrier(self, kind, value=None):
        self.jobs.put((kind, value))

    def close(self):
        self.barrier("close")
        self.thread.join()


class RecordingState:
    def __init__(self, writer, scheduler, action_ranges=None, start_motion_threshold=0.005,
                 pre_roll_frames=12):
        self.writer, self.scheduler = writer, scheduler
        self.recording = False
        self.armed = False
        self.busy = False
        self.task = None
        self.frames = 0
        self.started = 0.0
        self.saved = 0
        self.notice = "Press SPACE when scene is ready"
        self.attempt = 0
        self.episode_overruns = 0
        self.action_ranges = action_ranges or {f'{j}.pos': 100. for j in JOINTS}
        self.start_motion_threshold = start_motion_threshold
        self.pre_roll = deque(maxlen=pre_roll_frames)
        self.reference_action = None
        self.current_action = None
        self.motion_ticks = 0
        self.previous_saved_seq = {}
        self.armed_at_ns = None

    @property
    def active(self):
        return self.armed or self.recording

    def poll(self):
        while True:
            try:
                kind, value = self.writer.results.get_nowait()
            except queue.Empty:
                return
            self.busy = False
            if kind == "saved":
                self.saved = value
                self.scheduler.success()
                self.notice = "Saved. Reset blocks with follower; SPACE when ready"
                summary = getattr(self.writer, 'last_saved_summary', None)
                if summary:
                    camera_lines = []
                    for name, stats in summary.get('timing',{}).get('cameras',{}).items():
                        age = stats['age_ms']
                        fmt = lambda number: 'unknown' if number is None else f'{number:.1f}ms'
                        camera_lines.append(f'  {name} fresh: {100*stats["fresh_ratio"]:.1f}% | age p50={fmt(age["p50"])} p95={fmt(age["p95"])} max={fmt(age["max"])}')
                    self.notice = f'[SAVED] ep={value-1:04d} frames={summary["frames"]} duration={summary["duration_s"]:.2f}s overruns={summary["overruns"]}\ncamera timing:\n'+ '\n'.join(camera_lines)+'\nReset blocks with follower; SPACE when ready'
            elif kind == "discarded":
                self.notice = "Discarded. Reset blocks; SPACE retries the same hidden task"
            else:
                self.recording = False
                self.armed = False
                self.task = None
                self.notice = f"WRITER ERROR: {value}; teleop active, press ESC to exit"

    def key(self, key, now):
        if key == " " and not self.active and not self.busy and not self.writer.error:
            self.task = self.scheduler.reveal()
            self.frames = 0
            self.attempt += 1
            self.episode_overruns = 0
            self.pre_roll.clear()
            self.previous_saved_seq = {}
            self.reference_action = copy.copy(self.current_action)
            self.motion_ticks = 0
            self.armed_at_ns = time.monotonic_ns()
            if isinstance(self.writer, EpisodeWriter):
                self.writer.barrier('event', {'event': 'armed', 'attempt': self.attempt, 'episode': self.saved, 'task': self.task, 'host_monotonic_ns': self.armed_at_ns})
            self.armed = True
            self.notice = "Waiting for first movement...   BACKSPACE cancel   ESC stop"
        elif key in ("\r", "\n") and self.recording and self.frames:
            self.finish("save", "Saving in background; teleop active")
        elif key in ("\x7f", "\b") and self.active:
            if self.armed:
                if isinstance(self.writer, EpisodeWriter):
                    self.writer.barrier('event', {'event': 'cancel', 'attempt': self.attempt, 'task': self.task, 'host_monotonic_ns': time.monotonic_ns()})
                self.armed, self.task = False, None
                self.pre_roll.clear()
                self.notice = 'Cancelled. Reset blocks; SPACE retries the same hidden task'
            else:
                self.finish("discard", "Discarding in background; teleop active")

    def finish(self, kind, notice):
        summary = {'attempt': self.attempt, 'task': self.task, 'frames': self.frames,
                   'duration_s': self.frames / getattr(self, 'fps', 30), 'overruns': self.episode_overruns,
                   'armed_at_ns': self.armed_at_ns, 'start_motion_threshold': self.start_motion_threshold,
                   'pre_roll_frames': self.pre_roll.maxlen, 'reason': notice}
        self.recording, self.armed, self.task, self.busy = False, False, None, True
        self.pre_roll.clear()
        self.notice = notice
        if isinstance(self.writer, EpisodeWriter):
            self.writer.barrier(kind, summary)
        else:
            self.writer.barrier(kind)

    def record(self, obs, action, cameras, now, period, timing=None):
        self.current_action = copy.copy(action)
        if not self.active:
            return
        reason = None
        for name, camera in cameras.items():
            timestamp = camera.latest_timestamp
            if name not in obs or timestamp is None or time.perf_counter() - timestamp > 0.25:
                reason = f"{name} RGB missing/stale"
        if reason:
            self.finish("discard", f"Episode discarded: {reason}; teleop active")
            return
        from lerobot.utils.feature_utils import build_dataset_frame
        self.fps = 1 / period
        telemetry = dict(timing or {})
        telemetry.setdefault('tick_monotonic_ns', time.monotonic_ns())
        telemetry.update(camera_snapshot(cameras, telemetry['tick_monotonic_ns']))
        frame = {
            **build_dataset_frame(self.writer.features, obs, "observation"),
            **build_dataset_frame(self.writer.features, action, "action"),
            "task": self.task,
        }
        frame['_collector_qc'] = telemetry
        if self.armed:
            if self.reference_action is None:
                self.reference_action = copy.copy(action)
            score = max(abs(action[key]-self.reference_action[key])/span for key,span in self.action_ranges.items())
            self.motion_ticks = self.motion_ticks+1 if score > self.start_motion_threshold else 0
            self.pre_roll.append(copy.deepcopy(frame))
            if self.motion_ticks < 2:
                return
            self.armed, self.recording = False, True
            self.started = self.pre_roll[0]['_collector_qc']['tick_monotonic_ns']/1e9
            self.notice = 'ENTER success   BACKSPACE discard   ESC stop'
            if isinstance(self.writer, EpisodeWriter):
                self.writer.barrier('event', {'event': 'start', 'attempt': self.attempt, 'episode': self.saved,
                    'task': self.task, 'host_monotonic_ns': time.monotonic_ns(),
                    'first_frame_monotonic_ns': int(self.started*1e9), 'movement_score': score})
            for buffered in self.pre_roll:
                if not self._enqueue(buffered, snapshot_owned=True):
                    break
            self.pre_roll.clear()
        else:
            self._enqueue(frame)

    def _enqueue(self, frame, snapshot_owned=False):
        """Assign indexes/newness only to frames entering the episode, including pre-roll."""
        telemetry = frame['_collector_qc']
        telemetry['episode_index'] = self.saved
        telemetry['frame_index'] = self.frames
        for camera in ('front','wrist'):
            seq = telemetry.get(f'{camera}_frame_seq')
            telemetry[f'{camera}_is_new'] = seq is not None and seq != self.previous_saved_seq.get(camera)
        accepted = self.writer.frame(frame, snapshot_owned=snapshot_owned) if isinstance(self.writer, EpisodeWriter) else self.writer.frame(frame)
        if not accepted:
            self.finish("discard", "Episode discarded: writer overload; teleop active")
            return False
        else:
            self.frames += 1
            for camera in ('front','wrist'):
                self.previous_saved_seq[camera] = telemetry.get(f'{camera}_frame_seq')
            return True


class Terminal:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Run in an interactive WSL terminal (stdin must be a TTY)")
        self.fd = sys.stdin.fileno()
        self.previous = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        sys.stdout.write("\x1b[?25l")
        return self

    def keys(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            import os
            return os.read(self.fd, 64).decode(errors="ignore")
        return ""

    def draw(self, state, fps):
        status = "RECORDING" if state.recording else "ARMED" if state.armed else "RESET"
        elapsed = f"  {time.monotonic() - state.started:.1f} s" if state.recording else ""
        task = ""
        if state.active:
            task = state.task
            for color, code in (("red", 91), ("blue", 94), ("yellow", 93)):
                task = task.replace(color, f"\x1b[1;{code}m{color.upper()}\x1b[0m")
        text = f"[{status}]{elapsed}\nSaved: {state.saved}   Control: {fps:.1f} Hz\n{task}\n{state.notice}\nESC safe stop"
        sys.stdout.write("\x1b[H\x1b[J" + text)
        sys.stdout.flush()

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)
        sys.stdout.write("\x1b[0m\x1b[?25h\n")
        sys.stdout.flush()


def make_dataset_factory(root, repo_id, fps, features):
    def factory():
        import av
        av.logging.set_level(av.logging.ERROR)
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset.create(
            repo_id=repo_id, root=root, fps=fps, features=features,
            robot_type="so101_follower", use_videos=True, metadata_buffer_size=1,
            video_backend="pyav",
            streaming_encoding=True, encoder_queue_maxsize=30, encoder_threads=1,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=18, g=2),
        )
    return factory


def run(args):
    config = json.loads(args.devices.read_text())
    if set(config["cameras"]) != {"front", "wrist"}:
        raise ValueError("Exactly front and wrist RGB cameras are required")
    paths = [config[role]["port"] for role in ("leader", "follower")]
    paths += [cam["path"] for cam in config["cameras"].values()]
    for path in paths:
        if not Path(path).exists():
            raise RuntimeError(f"Device missing: {path}. Attach USB to WSL; do not replace with an assumed numeric node.")
    robot, leader = make_hardware(config)
    hardware_preflight(robot, leader)
    if args.check:
        print("Read-only identity, calibration and startup pose checks passed. No motor writes.")
        print(json.dumps(dataset_features(robot), indent=2))
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Use an interactive terminal")
    root = args.root or Path.home() / "datasets" / ("so101_rgb_blocks_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {root}")
    backing = Path(config.get("storage_backing_mount", "/"))
    if not backing.exists():
        raise RuntimeError(f"Cannot check physical backing disk: {backing}")
    if shutil.disk_usage(backing).free < 5 * 1024**3:
        raise RuntimeError(f"Less than 5 GiB free on backing disk {backing}; free space before collecting")
    root.parent.mkdir(parents=True, exist_ok=True)
    features = dataset_features(robot)
    action_ranges = calibration_action_ranges(robot)
    threshold = getattr(args, 'start_motion_threshold', 0.005)
    pre_roll_frames = getattr(args, 'pre_roll_frames', 12)
    session_manifest = manifest(root, config, robot, leader)
    session_manifest['episode_start'] = {'action_ranges': action_ranges, 'start_motion_threshold': threshold,
                                        'consecutive_ticks': 2, 'pre_roll_frames': pre_roll_frames}
    instrument_state_read(robot)
    writer = EpisodeWriter(make_dataset_factory(root, args.repo_id, config["fps"], features),
                           capacity=max(30, pre_roll_frames+2))
    try:
        (root/'session_manifest.json').write_text(json.dumps(session_manifest, indent=2))
    except Exception:
        writer.close()
        raise
    state = RecordingState(writer, TaskScheduler(), action_ranges, threshold, pre_roll_frames)
    from lerobot.processor import make_default_processors
    teleop_processor, action_processor, obs_processor = make_default_processors()
    fps = config["fps"]
    period = 1 / fps
    ticks = 0
    overruns = 0
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    started = time.perf_counter()
    deadline = started
    last_draw = 0
    last_record_tick = None
    last_tick_start_ns = None
    try:
        # Existing LeRobot connect/configure; calibration routines are explicitly disabled.
        leader.connect(calibrate=False)
        robot.connect(calibrate=False)
        started = deadline = time.perf_counter()
        with Terminal() as terminal:
            while not stop.is_set():
                state.poll()
                now = time.perf_counter()
                tick_start_ns = time.monotonic_ns()
                control_dt_ms = (tick_start_ns-last_tick_start_ns)/1e6 if last_tick_start_ns is not None else None
                last_tick_start_ns = tick_start_ns
                keys = terminal.keys()
                if "\x1b" in keys or "\x03" in keys:
                    break
                for key in keys:
                    state.key(key, now)
                # Same control pipeline as installed lerobot_teleoperate.teleop_loop.
                obs = robot.get_observation()
                raw_action = leader.get_action()
                action_read_done_ns = time.monotonic_ns()
                teleop_action = teleop_processor((raw_action, obs))
                to_send = action_processor((teleop_action, obs))
                sent_action = robot.send_action(to_send)
                action_send_done_ns = time.monotonic_ns()
                timing = {'tick_start_monotonic_ns': tick_start_ns, 'tick_monotonic_ns': time.monotonic_ns(),
                          'state_timestamp_ns': robot.state_read_done_ns,
                          'state_read_done_ns': robot.state_read_done_ns,
                          'action_read_done_ns': action_read_done_ns,
                          'action_send_timestamp_ns': action_send_done_ns,
                          'action_send_done_ns': action_send_done_ns, 'control_dt_ms': control_dt_ms}
                # Save the returned action actually sent, not a pre-clipping intention.
                if state.active:
                    if state.recording and last_record_tick is not None and now - last_record_tick > 2 * period:
                        state.finish("discard", "Episode discarded: control timing gap; teleop active")
                    else:
                        state.record(obs_processor(obs), sent_action, robot.cameras, time.perf_counter(), period, timing)
                    last_record_tick = now if state.recording else None
                else:
                    state.current_action = copy.copy(sent_action)
                    last_record_tick = None
                ticks += 1
                if now - last_draw > 0.25:
                    terminal.draw(state, ticks / max(now - started, period))
                    last_draw = now
                deadline += period
                delay = deadline - time.perf_counter()
                if delay > 0:
                    stop.wait(delay)
                else:
                    overruns += 1
                    if state.recording:
                        state.episode_overruns += 1
                    deadline = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop physical control promptly before waiting for any disk/video finalization.
        for device in (robot, leader):
            try:
                if device.bus.is_connected:
                    device.bus.disconnect(disable_torque=True)
            except Exception:
                logging.exception("Arm disconnect failed")
        for camera in robot.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception:
                logging.exception("Camera disconnect failed")
        writer.close()  # unfinished episode discarded; previous successful episodes finalized
        summary = {"root": str(root), "saved": writer.saved, "control_ticks": ticks,
                   "overruns": overruns, "writer_error": writer.error}
        (root / "collector_session.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    if writer.error:
        raise RuntimeError(f"Dataset writer failed: {writer.error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=Path, default=Path(__file__).with_name("devices.json"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--repo-id", default="local/so101_rgb_blocks")
    parser.add_argument("--check", action="store_true", help="read-only calibration/pose preflight; no control")
    parser.add_argument('--start-motion-threshold', type=float, default=0.005,
                        help='max action delta / existing calibration span; requires 2 consecutive ticks (default: 0.005)')
    parser.add_argument('--pre-roll-frames', type=int, default=12,
                        help='recent ARMED frames retained at movement onset (default: 12)')
    args = parser.parse_args()
    if not 0 < args.start_motion_threshold < 1 or not 1 <= args.pre_roll_frames <= 30:
        parser.error('threshold must be between 0 and 1; pre-roll frames must be 1..30')
    logging.basicConfig(level=logging.WARNING)
    run(args)


if __name__ == "__main__":
    main()
