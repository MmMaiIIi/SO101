# SO101 Stack2 π0.5 contract

Read this contract from the step-4228 checkpoint, its saved processors, pinned LeRobot code, and the LeRobot v3 data. Do not reorder keys or apply a second normalization.

## Observation

| Field | Contract |
|---|---|
| Model camera order | `observation.images.wrist`, then `observation.images.front` |
| Camera meaning | wrist = gripper-mounted close view; front = stationary table view |
| Raw frame | RGB, `uint8`, HWC, 640×480×3 |
| Image processor | Saved pipeline; does model resize/padding to 224×224 and maps to [-1,1] |
| Robot state | 6 float32 values: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| Arm units | Degrees |
| Gripper units | Calibrated 0–100 range |
| Language | One string passed through the saved tokenizer processor |

## Action

| Field | Contract |
|---|---|
| Dimension/order | 6, same names/order as state |
| Model arm output | First five dimensions are relative to the observation state at chunk creation |
| Model gripper output | Absolute normalized 0–100 target |
| Runtime output | Saved postprocessor denormalizes and restores the arm anchor; resulting 6D command is absolute |
| Predicted chunk | 50 actions |
| Frozen asynchronous execution | Up to 50 actions, selected by observation/control tick alignment; obsolete prefix is dropped |
| Action anchor | One unchanged observation-state anchor for every action in a chunk |

The historical synchronous `n_action_steps=10` is retained by the checkpoint but is not the cloud async action cap.

## Timing and normalization

- Dataset rate and control target: 30 Hz.
- Async refresh threshold: 0.6 of the configured 50-action chunk.
- Empty queue: safe hold of the current target. Expired actions are discarded and never replayed.
- State/action use the saved training quantile statistics; images use the saved processor operations.
- Checkpoint dtype: bfloat16.
- No lag compensation, action shift, resampling, cumulative relative integration, or hidden axis reorder.
- Camera timestamps are host `time.monotonic()` update times, not sensor exposure timestamps.

## Safety boundary

Absolute calibration/position bounds, nonfinite values, stale observations, and communication failure remain stop conditions. Dataset percentile slew statistics are diagnostics, not mechanical limits. The runtime does not silently clip a target. The single real Stack2 result and its post-task safety stop are recorded in [REAL_ROBOT_RESULT.md](REAL_ROBOT_RESULT.md).
