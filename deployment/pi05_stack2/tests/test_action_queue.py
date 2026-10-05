import numpy as np
import pytest

from deployment.pi05_stack2.action_queue import SafeActionQueue


def test_fifo_and_stale_sequence_rejection():
    q = SafeActionQueue(10)
    chunk = np.arange(30, dtype=np.float32).reshape(5, 6)
    q.push_chunk(4, chunk)
    np.testing.assert_array_equal(q.pop(4), chunk[0])
    with pytest.raises(ValueError, match="Stale"):
        q.push_chunk(4, chunk)
    q.push_chunk(5, chunk)
    assert len(q) == 5


def test_queue_clears_on_bad_chunk():
    q = SafeActionQueue()
    with pytest.raises(ValueError):
        q.push_chunk(1, np.full((3, 6), np.nan))
    assert len(q) == 0
