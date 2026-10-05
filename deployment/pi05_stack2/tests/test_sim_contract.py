import numpy as np
from gymnasium.spaces import Box

from deployment.pi05_stack2.config import CAMERA_ORDER
from deployment.pi05_stack2.sim_adapter import policy_degrees_to_sim_radians, sim_observation_to_policy


def test_sim_arm_degrees_and_gripper_percent_roundtrip():
    space = Box(low=np.array([-1.9, -1.7, -1.6, -1.6, -2.7, -0.17], dtype=np.float32),
                high=np.array([1.9, 1.7, 1.6, 1.6, 2.7, 1.74], dtype=np.float32))
    source = np.array([30, -25, 45, -35, 70, 62], dtype=np.float32)
    sim = policy_degrees_to_sim_radians(source, space)
    state = sim.copy()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = {"state": state, "wrist_camera": frame, "overhead_camera": frame}
    converted = sim_observation_to_policy(obs, "task", space, now_monotonic=3.0)
    np.testing.assert_allclose(converted["observation.state"], source, atol=1e-5)
    assert tuple(converted[CAMERA_ORDER[0]].shape) == (480, 640, 3)
