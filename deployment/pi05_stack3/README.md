# SO101 Stack3 inference-only baseline

This experiment uses the frozen Stack2 π0.5 deployment unchanged. It tests whether the Stack2 policy can compose learned stacking primitives; it does not train a new policy.

## Immutable baseline

```text
base_deployment_git_sha: 4689af34ea3c9886b2834531d161ee21697fe59c
freeze_metadata_commit: 62b0b7de77d55a2d7d8132291f5bf3432fd73fbe
checkpoint: WoWjx/so101-stack2-pi05-full-relative@main
checkpoint_sha256: f54e8cec0217bc5e2baebc095f06979675209c955093e7d5916fc5587eb098c4
deployment_v1_config_sha256: 1c1900f3545426a9d0a78187294e401fd17ff7ecab3623086158f11d8b082039
```

The experiment branch records this baseline in `BASELINE.json`. Do not edit `deployment/pi05_stack2/` on this branch. Any infrastructure correction requires a separately named `DEPLOYMENT_CHANGE` commit and must not be mixed into B0/B1 comparisons.

## Start gate

Use the same SO101, camera placement, lighting, cubes, reachable workspace, and frozen [`../pi05_stack2/configs/deployment_v1.yaml`](../pi05_stack2/configs/deployment_v1.yaml) for all runs. Before each trial, all three cubes must start separately on the table in the recorded reachable region, with the robot clear. A previously stacked scene is not a valid fresh Stack3 start. Stop on an existing v1 safety/system abort or physical emergency; do not change bounds or deployment settings to force a trial.

The Stack2 success run left a two-block stack and the prior run ended with motors disabled. Stack3 physical trials have not started because a fresh three-cube scene has not been confirmed. Arrange/reset the blocks locally, then run from the robot computer. No automatic scene reset is implemented.

## B0 — one full instruction, no high-level controller

Instruction:

```text
Stack the red block on top of the blue block, then stack the yellow block on top of the red block.
```

Use the frozen live runner for each attempt, passing that exact instruction. Do not manually change the instruction mid-run or invoke Stack2 subtasks. Run 3–5 smoke trials, resetting the physical scene between attempts. Record whether the first placement succeeds, whether the policy continues to the second placement, and the final stable stack order.

## B1 — explicit Stack2 composition

Use the same frozen low-level policy/runtime. Run two Stack2 instructions in sequence:

1. `Stack the red block on top of the blue block.`
2. `Stack the yellow block on top of the red block.`

Initially, transition verification requires operator confirmation that the first two-block stack is released and stable. This condition is **B1-privileged**, not autonomous: the operator may only confirm/deny completion, not move the robot or blocks during the policy episode. Reset all three cubes between B1 attempts. Run 3–5 smoke trials. Do not compare B1-privileged against B0 as an autonomous controller.

## Execution / logs

Keep the GPU inference server and SSH forward persistent. On the robot host, use the v1 runner with local-only hardware files:

```bash
python -m deployment.pi05_stack2.cli.run_async_robot \
  --enable-hardware \
  --devices-json /path/to/devices.json \
  --calibration /path/to/follower_calibration.json \
  --safety-profile /path/to/safety_profile.json \
  --server http://127.0.0.1:8765 \
  --instruction "Stack the red block on top of the blue block, then stack the yellow block on top of the red block." \
  --output-root runs/stack3/B0
```

The runner writes per-tick logs, inference trace, synchronized front/wrist video, and a summary. Record the final visual stack evaluation in `results/`; do not infer success from loop completion. The v1 absolute safety gates remain active.

For B1 run one policy call per subtask, stop the first call only after completion/stop, record the operator's stable-stack confirmation, then start the second instruction. The model server stays resident, but each client call is a distinct recorded subtask run.

## Metrics and failure taxonomy

Each attempt records overall Stack3 success; subtask-1/subtask-2 outcomes; failure stage; inference request/motor-command count; queue underflow and safe-hold duration; safety interventions; duration; subtask transitions; and whether second manipulation destroyed the first stack.

Failure labels: `NO_GRASP_1`, `BAD_PLACE_1`, `STACK1_UNSTABLE`, `FAILED_TRANSITION`, `NO_GRASP_2`, `DESTROYED_EXISTING_STACK`, `BAD_PLACE_2`, `FINAL_STACK_UNSTABLE`, `SAFETY_ABORT`, `SYSTEM_ABORT`.

Success requires all three cubes in the intended vertical order, gripper released, robot no longer supporting them, and the stack stable through a short observation window. Always report `task_outcome` separately from `runtime_termination`.

Use [`result.schema.json`](result.schema.json) for attempt records. The first 3–5 B0 and 3–5 B1-privileged trials are smoke tests to find the dominant failure mechanism; do not claim population-level performance.
