"""Hardware-free lifecycle and real LeRobot v3/video verification."""
import argparse
import json
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

from collector import EpisodeWriter, JOINTS, RecordingState, TaskScheduler, make_dataset_factory


class FakeDataset:
    features = {}

    def __init__(self, blocked_save=None):
        self.buffer = []
        self.episodes = []
        self.blocked_save = blocked_save
        self.saving = threading.Event()
        self.finalized = False
        self.seen = []

    def add_frame(self, frame):
        self.buffer.append(frame)
        self.seen.append(frame)

    def save_episode(self, parallel_encoding=False):
        self.saving.set()
        if self.blocked_save:
            self.blocked_save.wait(5)
        self.episodes.append(self.buffer)
        self.buffer = []

    def clear_episode_buffer(self):
        self.buffer = []

    def has_pending_frames(self):
        return bool(self.buffer)

    def finalize(self):
        self.finalized = True


def await_idle(state, timeout=30):
    deadline = time.monotonic() + timeout
    while state.busy and time.monotonic() < deadline:
        state.poll()
        if state.writer.error:
            raise RuntimeError(state.writer.error)
        time.sleep(0.005)
    assert not state.busy, "writer barrier timed out"


class LifecycleTests(unittest.TestCase):
    def test_scheduler_cycles_and_retry(self):
        scheduler = TaskScheduler(7)
        tasks = []
        for _ in range(12):
            task = scheduler.reveal()
            self.assertEqual(scheduler.reveal(), task)
            tasks.append(task)
            scheduler.success()
        self.assertEqual(len(set(tasks[:6])), 6)
        self.assertEqual(len(set(tasks[6:])), 6)

    def test_save_never_blocks_control_and_reset_is_hidden(self):
        release = threading.Event()
        dataset = FakeDataset(release)
        writer = EpisodeWriter(lambda: dataset, capacity=2)
        state = RecordingState(writer, TaskScheduler(1))
        try:
            self.assertIsNone(state.task)
            state.key(" ", time.perf_counter())
            first_task = state.task
            state.armed, state.recording = False, True  # writer lifecycle test bypasses onset detector
            self.assertTrue(writer.frame({"task": first_task, "marker": 1}))
            state.frames = 1
            state.key("\r", time.perf_counter())
            self.assertTrue(dataset.saving.wait(2))
            self.assertFalse(state.recording)
            self.assertIsNone(state.task)
            ticks = 0
            for _ in range(100):
                ticks += 1  # represents unconditional existing teleop control calls
                state.poll()
                state.key(" ", time.perf_counter())
            self.assertEqual(ticks, 100)
            self.assertTrue(state.busy)
            self.assertIsNone(state.task)
            self.assertEqual(state.saved, 0)
            release.set()
            await_idle(state)
            self.assertEqual(state.saved, 1)
            state.key(" ", time.perf_counter())
            retry_task = state.task
            state.armed, state.recording = False, True
            self.assertNotEqual(retry_task, first_task)
            self.assertTrue(writer.frame({"marker": 2}))
            state.frames = 1
            state.key("\x7f", time.perf_counter())
            await_idle(state)
            self.assertIsNone(state.task)
            state.key(" ", time.perf_counter())
            self.assertEqual(state.task, retry_task)
            self.assertTrue(writer.frame({"marker": 3}))
            writer.close()  # ESC closes with an unsaved current episode
            self.assertEqual(len(dataset.episodes), 1)
            self.assertEqual(dataset.episodes[0][0]["marker"], 1)
            self.assertEqual(dataset.buffer, [])
            self.assertTrue(dataset.finalized)
        finally:
            release.set()
            if writer.thread.is_alive():
                writer.close()

    def test_overload_is_nonblocking_and_snapshot_is_detached(self):
        release = threading.Event()
        dataset = FakeDataset(release)
        writer = EpisodeWriter(lambda: dataset, capacity=1)
        try:
            writer.barrier("save")
            self.assertTrue(dataset.saving.wait(2))
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            self.assertTrue(writer.frame({"image": image}))
            image[:] = 255
            start = time.perf_counter()
            self.assertFalse(writer.frame({"image": image}))
            self.assertLess(time.perf_counter() - start, 0.05)
            release.set()
            writer.close()
            # On close, pending frames are discarded as on ESC.
            self.assertEqual(dataset.buffer, [])
            self.assertEqual(int(dataset.seen[0]["image"].sum()), 0)
        finally:
            release.set()
            if writer.thread.is_alive():
                writer.close()

    def test_stale_rgb_discards_and_does_not_advance(self):
        dataset = FakeDataset()
        writer = EpisodeWriter(lambda: dataset)
        state = RecordingState(writer, TaskScheduler(2))
        try:
            state.key(" ", time.perf_counter())
            task = state.task
            state.record({}, {}, {"front": SimpleNamespace(latest_timestamp=None)}, time.perf_counter(), 1 / 30)
            self.assertFalse(state.recording)
            self.assertIsNone(state.task)
            await_idle(state)
            state.key(" ", time.perf_counter())
            self.assertEqual(state.task, task)
            self.assertEqual(state.saved, 0)
        finally:
            writer.close()


def full_loop_roundtrip():
    """Exercise collector.run itself with fake hardware and the real dataset writer."""
    import collector
    import pyarrow.parquet as pq

    class Bus:
        is_connected = False

        def disconnect(self, disable_torque=True):
            self.is_connected = False

    class Robot:
        def __init__(self):
            self.ticks = 0
            self.sent = 0
            self.cameras = {n: SimpleNamespace(latest_timestamp=0, is_connected=False) for n in ("front", "wrist")}
            self.bus = Bus()
            self.action_features = {f"{j}.pos": float for j in JOINTS}
            self.observation_features = {**self.action_features, "front": (48, 64, 3), "wrist": (48, 64, 3)}

        def connect(self, calibrate=False):
            assert calibrate is False
            self.bus.is_connected = True

        def get_observation(self):
            self.ticks += 1
            self.state_read_done_ns = time.monotonic_ns()
            for camera in self.cameras.values():
                camera.latest_timestamp = time.perf_counter()
                camera.last_read_frame_metadata = (self.ticks, time.monotonic_ns())
            image = np.full((48, 64, 3), 90, dtype=np.uint8)
            return {**{f"{j}.pos": float(self.ticks) for j in JOINTS}, "front": image, "wrist": image}

        def send_action(self, action):
            self.sent += 1
            return {key: value - 0.25 for key, value in action.items()}

    robot = Robot()

    class Leader:
        def __init__(self):
            self.bus = Bus()

        def connect(self, calibrate=False):
            assert calibrate is False
            self.bus.is_connected = True

        def get_action(self):
            return {f"{j}.pos": float(robot.ticks) + 0.5 for j in JOINTS}

    class Terminal:
        def __init__(self):
            self.state = None
            self.stage = 0
            self.retry_task = None
            self.discarded_task = None
            self.reset_ticks = []

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def draw(self, state, fps):
            self.state = state
            if not state.active:
                assert state.task is None

        def keys(self):
            s = self.state
            if robot.ticks > 400:
                raise AssertionError("simulation stalled")
            if self.stage == 0 and robot.ticks == 10:
                self.stage = 1
                return " "
            if self.stage == 1 and robot.ticks == 70:
                self.stage = 2
                return "\r"
            if self.stage == 2 and s is not None and s.saved == 1 and robot.ticks >= 95:
                self.stage = 3
                return " "
            if self.stage == 3 and s.frames >= 12:
                self.discarded_task = s.task
                self.stage = 4
                return "\x7f"
            if self.stage == 4 and not s.busy:
                self.stage = 5
                return " "
            if self.stage == 5 and s.recording:
                self.retry_task = s.task
                assert self.retry_task == self.discarded_task
                if s.frames >= 15:
                    self.stage = 6
                    return "\r"
            if self.stage == 6 and s.saved == 2:
                return "\x1b"
            if s and not s.recording:
                self.reset_ticks.append(robot.ticks)
            return ""

    terminal = Terminal()
    with tempfile.TemporaryDirectory(prefix="so101_full_loop_") as temp:
        root = Path(temp) / "dataset"
        cfg = {"leader": {"port": __file__}, "follower": {"port": __file__},
               "cameras": {n: {"path": __file__} for n in ("front", "wrist")},
               "fps": 30, "storage_backing_mount": "/mnt/d"}
        devices = Path(temp) / "devices.json"
        devices.write_text(json.dumps(cfg))
        args = SimpleNamespace(devices=devices, root=root, repo_id="local/full_loop_validation", check=False)
        with patch.object(collector, "make_hardware", return_value=(robot, Leader())), \
             patch.object(collector, "hardware_preflight"), \
             patch.object(collector, "calibration_action_ranges", return_value={f'{j}.pos':100. for j in JOINTS}), \
             patch.object(collector, "Terminal", return_value=terminal), \
             patch.object(collector.sys.stdin, "isatty", return_value=True):
            collector.run(args)
        assert robot.ticks == robot.sent
        assert len(terminal.reset_ticks) >= 20
        tables = [pq.read_table(p).to_pandas() for p in (root / "data").rglob("*.parquet")]
        import pandas as pd
        table = pd.concat(tables)
        assert len(table) == 75, len(table)
        assert list(table.groupby("episode_index").size()) == [60, 15]
        assert all(float(a[0]) == float(s[0]) + 0.25 for a, s in zip(table["action"], table["observation.state"]))
        assert int(table.iloc[0]["observation.state"][0]) == 11
        events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        saves = [event for event in events if event['event'] == 'save']
        assert [event['episode'] for event in saves] == [0, 1]
        assert [event['frames'] for event in saves] == [60, 15]
        assert any(event['event'] == 'discard' for event in events)
        sidecars = [pq.read_table(root/'telemetry'/f'episode_{i:03d}.parquet').to_pandas() for i in (0,1)]
        assert [len(sidecar) for sidecar in sidecars] == [60, 15]
        for i, sidecar in enumerate(sidecars):
            assert list(sidecar.frame_index) == list(range(len(sidecar)))
            assert set(sidecar.episode_index) == {i}
        assert (root/'session_manifest.json').is_file()
        import sys
        sys.path.insert(0, str(Path(__file__).parent/'tools'))
        from qc_dataset import host_camera_telemetry
        telemetry, errors = host_camera_telemetry(root)
        assert not errors, errors
        assert [telemetry[i]['sample_count'] for i in (0, 1)] == [60, 15]
        result = {"hardware": "simulated", "actual_collector_run": "passed", "control_ticks": robot.ticks,
                  "sent_actions": robot.sent, "reset_ticks_with_control_active": len(terminal.reset_ticks),
                  "saved_frames": 75, "actual_sent_action_saved": True, "discard_same_task_retry": True,
                  "telemetry_events_and_manifest": "passed"}
        print(json.dumps(result, indent=2))
        return result


def dataset_roundtrip(output):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import hw_to_dataset_features
    import pyarrow.parquet as pq
    import av

    features = {
        **hw_to_dataset_features({f"{j}.pos": float for j in JOINTS}, "action"),
        **hw_to_dataset_features({**{f"{j}.pos": float for j in JOINTS},
                                  "front": (480, 640, 3), "wrist": (480, 640, 3)}, "observation"),
    }
    with tempfile.TemporaryDirectory(prefix="so101_baseline_") as temp:
        root = Path(temp) / "dataset"
        writer = EpisodeWriter(make_dataset_factory(root, "local/baseline_validation", 30, features))
        state = RecordingState(writer, TaskScheduler(3))
        tasks = []

        def frames(count, offset=0):
            for i in range(count):
                image = np.empty((480, 640, 3), dtype=np.uint8)
                image[:] = ((i * 3 + offset) % 255, 30, 210)
                frame = {"observation.images.front": image,
                         "observation.images.wrist": image[:, ::-1].copy(),
                         "observation.state": np.full(6, i, dtype=np.float32),
                         "action": np.full(6, i + 0.5, dtype=np.float32), "task": state.task}
                while not writer.frame(frame):
                    if writer.error:
                        raise RuntimeError(writer.error)
                    time.sleep(0.005)
                state.frames += 1

        try:
            state.key(" ", time.perf_counter())
            state.armed, state.recording = False, True
            tasks.append(state.task)
            frames(90)
            state.key("\r", time.perf_counter())
            await_idle(state)
            assert state.task is None and state.saved == 1
            state.key(" ", time.perf_counter())
            retry_task = state.task
            state.armed, state.recording = False, True
            frames(10, 50)
            state.key("\x7f", time.perf_counter())
            await_idle(state)
            state.key(" ", time.perf_counter())
            assert state.task == retry_task
            state.armed, state.recording = False, True
            tasks.append(state.task)
            frames(60, 80)
            state.key("\r", time.perf_counter())
            await_idle(state)
            writer.close()
            assert writer.error is None, writer.error
            info = json.loads((root / "meta/info.json").read_text())
            assert info["codebase_version"] == "v3.0"
            assert info["total_episodes"] == 2 and info["total_frames"] == 150
            tables = [pq.read_table(path).to_pandas() for path in sorted((root / "data").rglob("*.parquet"))]
            import pandas as pd
            table = pd.concat(tables, ignore_index=True)
            assert len(table) == 150
            assert list(table.groupby("episode_index").size()) == [90, 60]
            assert all(float(a[0]) == float(s[0]) + 0.5 for a, s in zip(table["action"], table["observation.state"]))
            video_results = {}
            for key in ("front", "wrist"):
                paths = sorted((root / "videos" / f"observation.images.{key}").rglob("*.mp4"))
                decoded = 0
                for path in paths:
                    with av.open(str(path)) as container:
                        for frame in container.decode(video=0):
                            rgb = frame.to_ndarray(format="rgb24")
                            assert rgb.shape == (480, 640, 3)
                            # Synthetic red/blue channel distinction survives RGB video encoding.
                            assert abs(float(rgb[..., 1].mean()) - 30) < 10
                            assert abs(float(rgb[..., 2].mean()) - 210) < 10
                            decoded += 1
                assert decoded == 150, (key, decoded)
                video_results[key] = {"decoded_frames": decoded, "resolution": [640, 480]}
            reader = LeRobotDataset("local/baseline_validation", root=root, video_backend="pyav")
            sample = reader[0]
            assert sample["task"] == tasks[0]
            assert sample["observation.images.front"].shape == (3, 480, 640)
            assert sample["observation.images.wrist"].shape == (3, 480, 640)
            result = {"synthetic_only": True, "format": info["codebase_version"], "fps": info["fps"],
                      "saved_episodes": 2, "discarded_episodes": 1, "saved_frames": 150,
                      "tasks": tasks, "videos": video_results, "LeRobotDataset_readback": "passed",
                      "lifecycle": "RESET → Space → success/save → RESET → discard/retry → success/save"}
            output.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
        finally:
            if writer.thread.is_alive():
                writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    dataset_roundtrip(args.output)
    result = json.loads(args.output.read_text())
    result["full_loop"] = full_loop_roundtrip()
    result["unit_tests_passed"] = outcome.testsRun
    args.output.write_text(json.dumps(result, indent=2))
