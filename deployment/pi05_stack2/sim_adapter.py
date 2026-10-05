from __future__ import annotations

import time
import numpy as np

from .observation_adapter import TimedFrame, make_observation


def make_stackcube_env(source: str = "yellow", target: str = "blue", *, seed: int = 0):
    """Build the public SO101-Nexus StackCube task with policy-compatible RGB keys."""
    import gymnasium as gym
    import so101_nexus.mujoco  # noqa: F401: register native Gymnasium environments.
    from so101_nexus.config import PhysicsConfig, RenderConfig, StackCubeConfig
    from so101_nexus.observations import JointPositions, OverheadCamera, WristCamera

    cfg = StackCubeConfig(
        cube_a_colors=source,
        cube_b_colors=target,
        obs_mode="visual",
        observations=[JointPositions(), WristCamera(640, 480), OverheadCamera(640, 480)],
        render=RenderConfig(width=640, height=480, camera="overhead"),
        physics=PhysicsConfig(timestep_s=1 / 600, control_period_s=1 / 30),
        robot_init_qpos_noise=0.0,
    )
    env = gym.make("MuJoCoStackCube-v1", config=cfg, render_mode="rgb_array", robot_init_qpos_noise=0.0)
    env.reset(seed=seed)
    return env


def sim_observation_to_policy(obs: dict, instruction: str, action_space, *, now_monotonic: float | None = None) -> dict:
    """Map Nexus radians and named RGB cameras to the trained degrees/RGB policy contract."""
    now = time.monotonic() if now_monotonic is None else now_monotonic
    state_rad = np.asarray(obs["state"], dtype=np.float32)
    if state_rad.shape != (6,):
        raise ValueError(f"Nexus visual state must be the six controlled joints, got {state_rad.shape}")
    if not np.isfinite(state_rad).all() or np.any(state_rad < action_space.low - 1e-5) or np.any(state_rad > action_space.high + 1e-5):
        raise ValueError("Nexus joint state is non-finite or outside its configured SO101 limits")
    state_degrees = np.rad2deg(state_rad).astype(np.float32)
    gripper_low, gripper_high = float(action_space.low[5]), float(action_space.high[5])
    state_degrees[5] = np.clip((state_rad[5] - gripper_low) / (gripper_high - gripper_low) * 100.0, 0, 100)
    cameras = {
        "observation.images.wrist": TimedFrame(np.asarray(obs["wrist_camera"], dtype=np.uint8), now),
        "observation.images.front": TimedFrame(np.asarray(obs["overhead_camera"], dtype=np.uint8), now),
    }
    return make_observation(state_degrees, cameras, instruction, now_monotonic=now)


def policy_degrees_to_sim_radians(action_degrees: np.ndarray, action_space) -> np.ndarray:
    """Map trained arm degrees and normalized 0..100 gripper position into Nexus joint radians."""
    target = np.asarray(action_degrees, dtype=np.float32)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("Expected finite six-dimensional policy action")
    if not 0.0 <= target[5] <= 100.0:
        raise ValueError("Policy gripper output must stay within its trained 0..100 range")
    action = np.empty(6, dtype=np.float32)
    action[:5] = np.deg2rad(target[:5])
    action[5] = action_space.low[5] + (target[5] / 100.0) * (action_space.high[5] - action_space.low[5])
    if not action_space.contains(action):
        raise ValueError("Decoded policy target is outside SO101-Nexus joint-space bounds")
    return action
