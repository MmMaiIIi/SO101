import numpy as np
import pytest

from deployment.pi05_stack2.action_adapter import JointBounds, filter_action, validate_action


def test_relative_arm_absolute_gripper_round_trip():
    state = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.63], dtype=np.float32)
    action = np.array([0.12, -0.18, 0.29, -0.39, 0.48, 0.71], dtype=np.float32)
    relative = action.copy()
    relative[:5] -= state[:5]
    decoded = relative.copy()
    decoded[:5] += state[:5]
    np.testing.assert_allclose(decoded, action, atol=1e-7)
    assert decoded[5] == action[5]  # absolute gripper is neither shifted nor decoded.


def test_action_rejects_shape_nonfinite_and_bound_violations():
    with pytest.raises(ValueError):
        validate_action(np.zeros(7))
    with pytest.raises(ValueError):
        validate_action(np.array([0, 0, 0, 0, 0, np.nan]))
    limits = JointBounds((-1,)*6, (1,)*6, (0.1,)*6)
    with pytest.raises(ValueError, match="Safety stop"):
        filter_action(np.array([0.2, 0, 0, 0, 0, 0]), np.zeros(6), limits)
