from __future__ import annotations

from collections import deque
import numpy as np
import time


class SafeActionQueue:
    """Small testable FIFO; production policy uses PI05Policy's native 10-action queue."""
    def __init__(self, capacity: int = 10):
        if capacity < 1:
            raise ValueError("capacity must be >=1")
        self._items: deque[tuple[int, np.ndarray]] = deque(maxlen=capacity)
        self._last_sequence = -1
        self.capacity = capacity
        self._expires_at = 0.0

    def push_chunk(self, sequence: int, actions: np.ndarray) -> None:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 6 or not np.isfinite(arr).all():
            self.clear()
            raise ValueError("Action chunk must be finite [time,6]")
        if len(arr) > self.capacity:
            self.clear()
            raise ValueError("Action chunk exceeds queue capacity")
        if sequence <= self._last_sequence:
            self.clear()
            raise ValueError("Stale or duplicate action sequence")
        self.clear()
        self._last_sequence = sequence
        self._expires_at = time.monotonic() + len(arr)/30
        self._items.extend((sequence, action.copy()) for action in arr)

    def pop(self, sequence: int) -> np.ndarray:
        if time.monotonic() > self._expires_at:
            self.clear()
            raise RuntimeError('Action queue expired; stale targets cleared')
        if not self._items or self._items[0][0] != sequence:
            self.clear()
            raise RuntimeError("Action queue empty or wrong sequence")
        return self._items.popleft()[1]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
