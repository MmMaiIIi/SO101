from __future__ import annotations

from dataclasses import dataclass
import time
import math


@dataclass
class Watchdog:
    state_timeout_s: float = 0.25
    camera_timeout_s: float = 0.50
    policy_timeout_s: float = 2.0

    def check(self, *, state_ts: float, camera_ts: tuple[float, ...], policy_started: float | None = None,
              now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        stamps = (current,state_ts,*camera_ts,*(() if policy_started is None else (policy_started,)))
        if not all(math.isfinite(x) for x in stamps):
            raise ValueError("CONTRACT_ERROR: nonfinite observation/policy timestamp")
        if any(ts > current for ts in (state_ts,*camera_ts)):
            raise ValueError("Observation timestamp is in the future")
        if current - state_ts > self.state_timeout_s:
            raise TimeoutError("STALE_STATE: stale robot state; clear action queue")
        if not camera_ts or any(current - ts > self.camera_timeout_s for ts in camera_ts):
            raise TimeoutError("STALE_CAMERA: stale camera observation; clear action queue")
        if policy_started is not None and current - policy_started > self.policy_timeout_s:
            raise TimeoutError("STOP: policy request timeout; clear action queue")
