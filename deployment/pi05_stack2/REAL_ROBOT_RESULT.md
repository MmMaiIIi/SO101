# Real SO101 Stack2 result

| Field | Result |
|---|---|
| Task | Stack the blue block on top of the yellow block |
| Task outcome | **SUCCESS**, visually confirmed in final video frames after release and gripper withdrawal |
| Runtime termination | **SAFETY_ABORT**, after task completion |
| Checkpoint | Public step 4228, `main`, SHA-256 `f54e8cec0217bc5e2baebc095f06979675209c955093e7d5916fc5587eb098c4` |
| Control | 30 Hz; 50-action native chunks; up to 50 consumed; refill threshold 0.6 |
| Requests / motor commands | 58 inference requests / 1,187 policy motor commands |
| Duration | 44.03 s |
| Queue | 130 underflow ticks (38 initial, 92 later); zero stale replays |
| Request latency | P50 640 ms; P95 1,085 ms; P99 1,185 ms |
| Prediction age | P50 1,033 ms; P95 1,533 ms; P99 1,618 ms |
| Serial | 1,320 reads, zero extra retries in this run |

The task completed before the retreat phase proposed a shoulder target 0.14135° beyond the saved calibration lower bound. That command was rejected and never sent; runtime then stopped and disabled torque. The result therefore separates task success from runtime termination: `task_outcome=SUCCESS`, `runtime_termination=SAFETY_ABORT`.

Latency stages on the same run, based on 58 requests:

| Stage | P50 | P95 |
|---|---:|---:|
| Client serialization | 33 ms | 37 ms |
| Server body receive | 147 ms | 258 ms |
| PNG decode | 26 ms | 27 ms |
| GPU model | 191 ms | 202 ms |
| End-to-end request | 635 ms | 1,084 ms |
| Unattributed transport/response remainder | ~220 ms | ~640 ms |

The remainder is not a measured one-way download time: client/server clocks were not synchronized, so one-way transfer time cannot be isolated. Full video and raw logs remain outside Git; see the operator's local `runs/real_stack2_20261004_225212/` directory.

This one run does not establish a success rate, general Stack2 robustness, Stack3 capability, sim2real transfer, or a statistical checkpoint comparison.
