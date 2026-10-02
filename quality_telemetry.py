"""Lightweight host camera-update telemetry; disk writes stay on dataset writer."""
import hashlib
import importlib.metadata
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def git_commit(path):
    """Return provenance when a checkout exists; never invent a commit."""
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, timeout=2).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return None


def manifest(root, config, robot, leader):
    """Snapshot configuration and existing calibration hashes before collection."""
    import lerobot
    base = Path(__file__).parent
    calibration = {}
    for role, device in [('leader', leader), ('follower', robot)]:
        location = getattr(device, 'calibration_fpath', None)
        path = Path(location) if location else None
        calibration[role] = {'path': str(path) if path else None, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None}
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset_root': str(root),
            'LeRobot_version': importlib.metadata.version('lerobot'),
            'LeRobot_commit': git_commit(Path(lerobot.__file__).parents[2]),
            'collector_commit': git_commit(base),
            'collector_sha256': hashlib.sha256((base/'collector.py').read_bytes()).hexdigest(),
            'devices': config, 'calibration': calibration,
            'task_list': [f'Stack the {a} block on top of the {b} block.' for a in ('red','blue','yellow') for b in ('red','blue','yellow') if a != b],
            'clock': 'time.monotonic_ns: host camera update immediately after successful hardware read; NOT sensor exposure timestamp',
            'frame_sidecar': 'telemetry/episode_NNN.parquet; one row per saved LeRobot frame_index',
            'tick_definition': 'observation/action snapshot after send_action; tick_start_monotonic_ns separately records loop start; ages at snapshot and at dataset.add_frame are distinct'}


def camera_snapshot(cameras, tick_ns):
    """Flatten metadata paired by read_latest; repeated reads retain sequence and time."""
    row = {}
    for name, camera in cameras.items():
        metadata = getattr(camera, 'last_read_frame_metadata', None)
        seq, update_ns = metadata if metadata is not None else (None, None)
        row[f'{name}_frame_seq'] = seq
        row[f'{name}_update_monotonic_ns'] = update_ns
        row[f'{name}_age_ms'] = (tick_ns-update_ns)/1e6 if update_ns is not None else None
    return row


def calibration_action_ranges(robot):
    """Use LeRobot's existing normalization to express calibration spans in sent-action units."""
    ranges = {}
    for name, cal in robot.calibration.items():
        motor_id = robot.bus.motors[name].id
        lower = robot.bus._normalize({motor_id: cal.range_min})[motor_id]
        upper = robot.bus._normalize({motor_id: cal.range_max})[motor_id]
        span = abs(upper-lower)
        if span <= 0:
            raise ValueError(f'Invalid calibration span for {name}')
        ranges[f'{name}.pos'] = span
    return ranges


def instrument_state_read(robot):
    """Timestamp follower Present_Position completion without changing the read or its result."""
    robot.state_read_done_ns = None
    if not hasattr(robot.bus, 'sync_read'):
        return
    original = robot.bus.sync_read
    def sync_read(data_name, *args, **kwargs):
        result = original(data_name, *args, **kwargs)
        if data_name == 'Present_Position':
            robot.state_read_done_ns = time.monotonic_ns()
        return result
    robot.bus.sync_read = sync_read


def timing_summary(rows):
    """Small saved-episode timing statistics; runs on the existing writer thread."""
    import numpy as np
    def stats(values):
        values = [v for v in values if v is not None]
        return {'p50': float(np.percentile(values,50)), 'p95': float(np.percentile(values,95)), 'max': float(max(values))} if values else {'p50': None, 'p95': None, 'max': None}
    return {'cameras': {name: {'fresh_ratio': sum(r.get(f'{name}_is_new',False) for r in rows)/max(len(rows),1),
                               'age_ms': stats([r.get(f'{name}_age_ms') for r in rows])}
                        for name in ('front','wrist')},
            'control_dt_ms': stats([r.get('control_dt_ms') for r in rows]),
            'state_to_action_ms': stats([(r['action_send_timestamp_ns']-r['state_timestamp_ns'])/1e6 for r in rows if r.get('state_timestamp_ns') is not None and r.get('action_send_timestamp_ns') is not None])}


def write_episode_timing(root, episode, rows):
    """Write typed Parquet atomically after save; reject missing/non-contiguous rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if any(r['frame_index'] != i for i,r in enumerate(rows)):
        raise ValueError('Timing frame_index is not contiguous')
    if root is None:
        return
    folder = Path(root)/'telemetry'
    folder.mkdir(exist_ok=True)
    for row in rows:
        row['episode_index'] = episode
    # Explicit nullable types also cover a camera with all-unknown metadata.
    integer_names = ('episode_index','frame_index','tick_monotonic_ns','tick_start_monotonic_ns',
                     'state_timestamp_ns','state_read_done_ns','action_read_done_ns','action_send_timestamp_ns',
                     'action_send_done_ns','dataset_write_monotonic_ns','front_frame_seq','front_update_monotonic_ns',
                     'wrist_frame_seq','wrist_update_monotonic_ns')
    float_names = ('control_dt_ms','front_age_ms','wrist_age_ms','front_dataset_write_age_ms','wrist_dataset_write_age_ms')
    schema = pa.schema([(name,pa.int64()) for name in integer_names]+[(name,pa.float64()) for name in float_names]+[(name,pa.bool_()) for name in ('front_is_new','wrist_is_new')])
    target = folder/f'episode_{episode:03d}.parquet'
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix('.parquet.tmp')
    pq.write_table(pa.Table.from_pylist(rows,schema=schema), temporary)
    temporary.replace(target)


def append_json(root, filename, value):
    """Append worker-owned sidecars; no existing raw dataset is opened for writing."""
    if root is not None:
        with (Path(root)/filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, allow_nan=False)+'\n')
