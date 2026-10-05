# SO101 StackCube simulator validation

Reference: `johnsutor/so101-nexus` commit `b497ffaaad7e82168d5f58e8c179aff6b68edf05`, package 0.8.0; MuJoCo 3.14.0; Gymnasium 1.3.0; environment `MuJoCoStackCube-v1`.

The public StackCube environment resets and renders successfully; its six-dimensional `JointPositions` state and action spaces use radians. Its `WristCamera` and `OverheadCamera` observations provide RGB 640×480 images. The deployment adapter maps overhead to trained `front`, wrist to trained `wrist`, arm radians to degrees, and simulator gripper radians to the trained 0–100 range. Physics control period is set to 1/30 s to match the dataset's 30 FPS.

The 100-step random-policy smoke test exercised reset, render, stepping and video recording. It did not execute our π0.5 checkpoint. Model-connected S0/S1/S2 run remains gated on inference resource availability; the current GPU has 6 GB total and cannot hold this 9.35 GB BF16 checkpoint. A random rollout is only an environment smoke test.

The environment's own StackCube success predicate requires top-cube placement/alignment, static settling, and release/no-grasp. Any future run must report success separately from deployment plumbing. Simulation is an interface preflight, not evidence of expected real-world success; it does not support a sim2real claim.
