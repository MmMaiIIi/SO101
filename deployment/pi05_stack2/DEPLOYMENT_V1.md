# SO101 π0.5 Stack2 deployment v1

## Frozen baseline

Real SO101 Stack2 deployment has been demonstrated successfully with the public step-4228 checkpoint using remote RTX 4090 asynchronous inference.

- Checkpoint: `WoWjx/so101-stack2-pi05-full-relative`, revision `main`, step 4228.
- Checkpoint SHA-256: `f54e8cec0217bc5e2baebc095f06979675209c955093e7d5916fc5587eb098c4`.
- Runtime config: [`configs/deployment_v1.yaml`](configs/deployment_v1.yaml).
- Runtime config SHA-256: `1c1900f3545426a9d0a78187294e401fd17ff7ecab3623086158f11d8b082039`.
- Machine-readable result: [`results/real_stack2_v1.json`](results/real_stack2_v1.json).
- Full video, tick log, and request log remain in the operator's local run directory and are intentionally not committed.

The observed task outcome was `SUCCESS`: the blue block remained stably on the yellow block after release and retreat. Runtime termination was `SAFETY_ABORT`: after task completion, a retreat-phase `shoulder_lift.pos` target was 0.14135° below the configured calibration lower bound. The safety gate rejected it, and that target was not sent to the motor. **SAFETY_ABORT != task failure** when visual task completion already occurred.

This is one demonstrated real-robot task outcome. It does not establish a 100% success rate, a general Stack2 solution, Stack3 capability, sim2real performance, or statistical superiority of step 4228 over step 2500.

## Frozen execution contract

The complete settings are recorded in `configs/deployment_v1.yaml`. Important values are 30 Hz control, a 50-action native prediction chunk, up to 50 actions consumed per async chunk, 0.6 refill threshold, persistent PNG-over-SSH inference, and SAFE HOLD on queue underflow. Old synchronous `n_action_steps=10` is historical and is not the async execution cap. Stale action replay is disabled.

Arm dimensions 0–4 are decoded relative to the state at chunk creation; gripper is an absolute 0–100 target. Saved checkpoint pre/postprocessors remain authoritative. Absolute calibration/position, NaN/Inf, stale-observation, and communication gates remain hard stops. Percentile-derived slew is a warning only; it is not a mechanical limit. No action clipping is performed.

## Freeze identity

The executable baseline is frozen at the annotated Git tag `pi05-stack2-deployment-v1`. The tag target is the v1 freeze commit; the concrete commit SHA is recorded here after that commit is created. Configuration SHA-256 is recorded in the tag metadata/release notes. Stack3 work must branch from this tag and record the base deployment commit SHA, checkpoint SHA, and config SHA. Any runtime change requires an explicit `DEPLOYMENT_CHANGE` and separate ablation; do not modify the v1 branch during Stack3 experiments.

## Reproduction entry points

Use the package's isolated deployment dependencies, download the exact checkpoint and matching processor files, and follow [`README.md`](README.md). The historical live rollout launcher used operator-local camera, device, calibration, and tunnel settings; these private values are deliberately excluded. Physical device paths and credentials must be supplied locally and must not be committed.
