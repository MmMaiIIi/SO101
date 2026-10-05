# Asynchronous alignment and real-run note

This document preserves the implementation rationale. The pre-hardware closure results below are historical; actual physical task evidence is in [REAL_ROBOT_RESULT.md](REAL_ROBOT_RESULT.md).

## Alignment

Pinned LeRobot commit: `ff71cae1ae2d09fd035553c35da65888ed6c8304`.

- `async_inference/policy_server.py::_time_action_chunk`: action i is assigned observation time + i/30 and observation tick + i.
- `async_inference/policy_server.py::_get_action_chunk`: uses native `predict_action_chunk`.
- `async_inference/robot_client.py::_aggregate_action_queues`: expired timesteps are discarded and newer predictions replace overlapping future timesteps.
- The deployment selects `index = current_control_tick - observation_tick`; it does not cumulatively integrate relative actions or replay stale outputs.

The frozen real async deployment consumes up to 50 actions from a native 50-action prediction. Historical synchronous `n_action_steps=10` is not the asynchronous execution cap. Chunk expiration is tied to the 50-step horizon.

The control worker owns robot state acquisition, safety, and guarded motor output at 30 Hz. HTTP, PNG encoding, preprocessing and inference run in a separate worker. Queue empty means SAFE HOLD; no stale chunk is replayed. The selected v1 configuration is recorded in [`configs/deployment_v1.yaml`](configs/deployment_v1.yaml).

## Historical software timing proof

Before hardware execution, a 1,000-tick paced test used prerecorded observations and a real RTX 4090 inference server. It verified inference/control overlap, queue sequencing and no stale replay. This did not establish physical success. The measured real WAN task timing is reported separately in [REAL_ROBOT_RESULT.md](REAL_ROBOT_RESULT.md).

## Limits

One real Stack2 outcome is demonstrated, not a success-rate estimate. The saved calibration envelope remains a command bound, not an independent collision-free mechanical certification. The successful task ended with a correct post-task retreat target rejection at the unchanged shoulder-lift lower bound.
