# SO101 full-chain teleoperation data infrastructure

Reusable baseline for collecting and inspecting SO101 teleoperation datasets
across tasks. It covers leader/follower control integration, LeRobot v3 episode
recording, camera and control timing telemetry, dataset QC, and review assets.
The current task scheduler is the initial three-color RGB block-stacking pilot;
it does not change motor calibration or IDs.

## Requirements

- Ubuntu/WSL2 with the existing LeRobot 0.6.2 environment and its SO101,
  camera, OpenCV, PyArrow, and video dependencies.
- The host camera timing patch in `patches/` applied to the matching LeRobot
  0.6.2 `src/lerobot/cameras/opencv/camera_opencv.py` source. The timestamps
  are host update times after successful camera reads, not exposure times.
- Local serial and camera device permissions/configuration.

Keep your real hardware configuration private. Start from
`devices.example.json`, save a local copy as `devices.json`, and replace every
placeholder with stable device paths. `devices.json` is git-ignored.

## Collect

Run from this directory in the existing LeRobot Python environment:

```bash
python collector.py --devices devices.json --root /path/to/datasets/session
```

SPACE reveals a task and arms the episode. Recording starts after two
consecutive meaningful action ticks and includes the configured pre-roll.
ENTER saves, BACKSPACE cancels/discards the current task attempt, and ESC
stops collection. The default start threshold is `0.005` of calibrated action
range with 12 pre-roll frames; both are configurable on the command line.

## Quality checks

```bash
python tools/qc_dataset.py /path/to/datasets/session --report
```

QC outputs are written outside the immutable raw dataset by default. Use
`python tools/qc_dataset.py review /path/to/datasets/session` to edit the
manual review sidecar. The tools support LeRobot v3 datasets and report
available telemetry when a timing sidecar is present.

## Development checks

Run tests inside the compatible LeRobot environment:

```bash
python tools/test_qc.py
python test_quality_telemetry.py
python test_armed_timing.py
python verify_baseline.py --output /tmp/collector_verification.json
```
