import numpy as np
import pytest

from deployment.pi05_stack2.config import CAMERA_ORDER, STATE_ORDER
from deployment.pi05_stack2.observation_adapter import TimedFrame, make_observation


def test_canonical_camera_and_joint_order():
    assert CAMERA_ORDER == ("observation.images.wrist", "observation.images.front")
    assert STATE_ORDER == ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                           "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")


def test_observation_shapes_and_rgb_range():
    now = 12.0
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    obs = make_observation(np.arange(6, dtype=np.float32), {
        CAMERA_ORDER[0]: TimedFrame(image, now), CAMERA_ORDER[1]: TimedFrame(image, now),
    }, "Stack the yellow block on the blue block.", now_monotonic=now)
    assert obs["observation.state"].shape == (6,)
    assert tuple(obs[CAMERA_ORDER[0]].shape) == (480, 640, 3)
    assert obs[CAMERA_ORDER[0]].dtype == np.uint8


def test_missing_or_stale_camera_fails_closed():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(KeyError):
        make_observation(np.zeros(6), {CAMERA_ORDER[0]: image}, "task")
    with pytest.raises(TimeoutError):
        make_observation(np.zeros(6), {k: TimedFrame(image, 0.0) for k in CAMERA_ORDER},
                         "task", now_monotonic=1.0)
