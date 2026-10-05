from __future__ import annotations

from dataclasses import dataclass
import time
import math
from typing import Mapping

import numpy as np

from .config import CAMERA_ORDER, STATE_ORDER


@dataclass(frozen=True)
class TimedFrame:
    image: np.ndarray
    timestamp_monotonic: float


def _rgb_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    """Validate RGB HWC frames in the same raw format accepted by LeRobot's inference helper."""
    arr = np.asarray(frame)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected uint8 RGB HWC image, got {arr.dtype} {arr.shape}")
    if arr.shape != (480, 640, 3):
        raise ValueError(f"Camera contract requires 480x640x3, got {arr.shape}")
    return np.ascontiguousarray(arr)


def make_observation(
    state: np.ndarray,
    cameras: Mapping[str, TimedFrame | np.ndarray],
    instruction: str,
    *,
    now_monotonic: float | None = None,
    camera_max_age_s: float = 0.5,
) -> dict:
    """Build the exact observation keys, rejecting missing/stale data before inference."""
    now = time.monotonic() if now_monotonic is None else now_monotonic
    if not math.isfinite(now): raise ValueError('CONTRACT_ERROR: nonfinite observation clock')
    s = np.asarray(state, dtype=np.float32)
    if s.shape != (6,) or not np.isfinite(s).all():
        raise ValueError(f"Expected finite 6D state ordered {STATE_ORDER}, got {s.shape}")
    obs = {"observation.state": s.copy()}
    for key in CAMERA_ORDER:
        if key not in cameras:
            raise KeyError(f"Missing required camera {key}")
        item = cameras[key]
        if isinstance(item, TimedFrame):
            age = now - item.timestamp_monotonic
            if not math.isfinite(age) or age < 0 or age > camera_max_age_s:
                raise TimeoutError(f"{key} stale: age={age:.3f}s limit={camera_max_age_s:.3f}s")
            image = item.image
        else:
            image = item
        obs[key] = _rgb_hwc_uint8(image)
    if not instruction.strip():
        raise ValueError("Language instruction must not be empty")
    return obs
