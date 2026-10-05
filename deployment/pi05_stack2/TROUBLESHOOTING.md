# Troubleshooting

- **Checkpoint hash mismatch:** stop before loading. Re-download the exact revision; do not use a mutable cache path or `step-2500` by mistake.
- **Missing/unexpected model keys:** strict loading fails closed. Do not accept an initialized fallback tensor.
- **Missing processor files/tokenizer:** stop; policy and processor artifacts are a paired unit.
- **Camera key/order mismatch:** input keys are `observation.images.wrist` and `observation.images.front`; their checkpoint tensor order is wrist then front.
- **Image appears saturated:** accepted raw format is RGB HWC `uint8`; call LeRobot's `prepare_observation_for_inference` once. Do not divide by 255 twice.
- **Arm action seems too large:** saved postprocessor emits absolute degree targets for five arm joints. Check the state anchor and verify the output is not being decoded twice.
- **Gripper direction/range mismatch:** target is absolute normalized 0–100, not degrees/radians. Stop if current or predicted range is outside 0–100.
- **Simulator range error:** check radians/degrees conversion for arm joints and normalized gripper interpolation. Do not clip unexplained model outputs.
- **Policy does not fit GPU:** the current RTX 3060 Laptop GPU has 6 GB VRAM; this checkpoint has 9.35 GB of bfloat16 weights before activations. Use a separate adequately provisioned GPU server and the versioned local-network protocol; no automatic API/cloud fallback is configured.
- **Stale observation/policy timeout:** clear action queues, reacquire a fresh observation, and restart inference. Never replay an old chunk indefinitely.
