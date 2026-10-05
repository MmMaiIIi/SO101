import pytest

from deployment.pi05_stack2.safety import Watchdog


def test_watchdog_state_camera_and_policy_timeouts():
    w = Watchdog(0.2, 0.4, 1.0)
    w.check(state_ts=9.9, camera_ts=(9.8, 9.9), now=10.0)
    with pytest.raises(TimeoutError, match="stale robot state"):
        w.check(state_ts=9.7, camera_ts=(10.0,), now=10.0)
    with pytest.raises(TimeoutError, match="stale camera"):
        w.check(state_ts=10.0, camera_ts=(9.5,), now=10.0)
    with pytest.raises(TimeoutError, match="policy request timeout"):
        w.check(state_ts=10.0, camera_ts=(10.0,), policy_started=8.0, now=10.0)
