# SO101 π0.5 Stack2 deployment

This package contains the frozen v1 software and deployment record for `WoWjx/so101-stack2-pi05-full-relative`, step 4228. A real blue-on-yellow Stack2 task was visually successful; runtime later stopped safely on an out-of-calibration retreat target. Details: [DEPLOYMENT_V1.md](DEPLOYMENT_V1.md), [BUGS_AND_FIXES.md](BUGS_AND_FIXES.md), [REAL_ROBOT_RESULT.md](REAL_ROBOT_RESULT.md).

## Install

Use an isolated Python 3.12 environment. The live run used LeRobot commit `ff71cae1ae2d09fd035553c35da65888ed6c8304`; see the repository deployment requirements. Install a CUDA PyTorch build matching the target GPU separately.

## Download checkpoint

```bash
hf download WoWjx/so101-stack2-pi05-full-relative \
  --revision main --local-dir models/so101-pi05-4228
```

Verify the model hash and keep its paired processor files together. Do not commit weights or Hugging Face cache.

## Preflight / offline inference

Read-only and offline tools remain available:

```bash
python -m deployment.pi05_stack2.cli.preflight --robot-only
python -m deployment.pi05_stack2.cli.preflight --camera-only
python -m deployment.pi05_stack2.cli.preflight --all
python -m deployment.pi05_stack2.cli.infer_offline --help
```

Hardware preflight requires local device/calibration inputs and writes no motor targets. It does not replace the staged safety checks for a new robot.

## Remote GPU inference

On the GPU server, run one persistent process:

```bash
python -m deployment.pi05_stack2.inference_server \
  --device cuda --checkpoint models/so101-pi05-4228 --host 127.0.0.1 --port 8765
```

Forward a local port over a private LAN/VPN/SSH connection; do not expose the HTTP server publicly. The demonstrated WAN route sent lossless PNG images over a persistent HTTP connection and SSH forward. Its settings and latency are recorded in `configs/deployment_v1.yaml` and `REAL_ROBOT_RESULT.md`.

## Frozen v1 behavior

- 30 Hz control; native prediction chunk is 50×6.
- Async runtime can consume up to 50 actions; historical synchronous `n_action_steps=10` is not the async cap.
- Refill when remaining chunk fraction reaches 0.6; align to observation time and discard expired actions.
- Empty queue means safe hold of current target. Never replay stale actions.
- Saved pre/postprocessors and relative-arm/absolute-gripper decoding are mandatory.
- Absolute calibration/position, NaN/Inf, stale data and communication gates remain hard stops. Percentile slew is diagnostic only; targets are never silently clipped.
- Hardware output defaults off outside the explicitly guarded real-run launcher. Do not use Stack3 experiments to edit this v1 package.

## Tests

```bash
pytest deployment/pi05_stack2/tests -q
```

Some integration tests need the pinned LeRobot environment and local checkpoint/dataset paths supplied through environment variables. Generated artifacts, hardware identities, calibration, raw datasets, full run logs, and videos are ignored by Git.

## Stack3

Branch from the `pi05-stack2-deployment-v1` tag. Stack3 instructions, result schema, and frozen-baseline comparison protocol live in `deployment/pi05_stack3/` on the experiment branch. Do not change runtime between B0 and B1.
