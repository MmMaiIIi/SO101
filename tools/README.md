# Reusable LeRobot v3 QC

Run in the existing LeRobot environment:

```bash
cd /path/to/so101-collector
python tools/qc_dataset.py /path/to/dataset --report
```

Outputs default to `so101-collector/qc/<dataset-name>/`, outside immutable raw data.
`--output` overrides this but is rejected inside the dataset. Raw file SHA256s
are checked before and after processing. Nothing is trimmed, deleted or uploaded.

Artifacts: JSON report, episode CSV, manual review CSV, HTML, contact sheets,
joint/lag/boundary/camera SVG plots, metric time series, combined review MP4s.
Shared source MP4s are sliced with official metadata time intervals.
`--no-review-video` skips MP4 generation; existing review videos remain available.

```bash
python tools/qc_dataset.py review /path/to/dataset
python tools/test_qc.py
python test_quality_telemetry.py
python verify_baseline.py --output /tmp/collector_verification.json
```

Manual KEEP/REVIEW/DROP labels are preserved on reruns. Automatic WARN flags
are review suggestions, never task-success labels. Current idle warning cutoffs
are start >1 s or end >2 s. Motion/near-static thresholds adapt to each episode;
moving+same >5% triggers a conservative review suggestion. Jump candidates use
MAD/p95/span, without claiming motor safety limits. Lag minimizes command versus
later measured-state RMSE over 0–10 frames; maximum-correlation lag is also saved.
Neither command lag nor encoded video timestamps prove camera synchronization.

New collector sessions include `session_manifest.json`, `events.jsonl` and
`telemetry/episode_NNN.parquet`. Each Parquet row matches a saved dataset
episode_index/frame_index. Camera sequences/timestamps are published with the
image under the camera lock. Timestamps use `time.monotonic_ns()` immediately
after a successful background hardware read, before postprocessing; they are
host camera update timestamps, not sensor exposure timestamps. Repeated
read_latest calls retain the same sequence and update time. Tick ages use the
observation/action snapshot after send_action; tick_start is separately stored.
Dataset write age uses the writer's time immediately before dataset.add_frame.
State timing marks follower Present_Position read completion; action timing
marks leader read completion and follower send_action completion. Nothing is
shifted or compensated. QC reads the sidecars automatically and checks their
indexes/counts and timestamp consistency; legacy data remains supported.

SPACE reveals the task and enters ARMED. A fixed SPACE reference is the most
recent actual sent action (first available sent action when none exists).
Movement score is max(abs(sent-reference)/follower calibration span), using
LeRobot's normalization for degrees and gripper 0–100. Default threshold is
0.005 (0.5% span), with 2 consecutive ticks. Change it with
`--start-motion-threshold`. ARMED retains only its newest 12 frames (0.4 s at
30 Hz); change with `--pre-roll-frames`. On trigger these frames, including both
confirmation frames, become the episode start. Available static context can be
shorter if the operator moves immediately. ARMED never starts on timeout.
Backspace cancels ARMED without an episode and retries the same hidden task.
Only successful saves advance the task schedule. Disk writes and summary
percentiles remain on the existing dataset writer; no new age-based auto-discard
rule was added. RESET never enters the dataset. ESC discards an unfinished
episode and finalizes previously saved episodes.

Hardware control, calibration, motor IDs and LeRobot/PyTorch versions unchanged.
Changes were exercised with simulated hardware only; confirm the next short
real episode before scaling collection.

```bash
python test_armed_timing.py
python collector.py --start-motion-threshold 0.005 --pre-roll-frames 12
```
