import numpy as np
import pytest
from deployment.pi05_stack2.config import CAMERA_ORDER
from deployment.pi05_stack2.protocol import encode_request, decode_request
from deployment.pi05_stack2.action_queue import SafeActionQueue


def test_lossless_camera_transport_and_contract_rejection():
    image = np.zeros((480,640,3), dtype=np.uint8)
    image[:,:,0] = 211
    obs = {"observation.state": np.arange(6, dtype=np.float32), **{k:image for k in CAMERA_ORDER}}
    request = encode_request(obs, 3, "test-session", "Stack the yellow block on top of the blue block.")
    decoded = decode_request(request)
    np.testing.assert_array_equal(decoded[CAMERA_ORDER[0]], image)
    np.testing.assert_array_equal(decoded["observation.state"], obs["observation.state"])
    request["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="mismatch"):
        decode_request(request)


def test_old_chunk_rejected_even_after_queue_drains():
    queue = SafeActionQueue(10)
    queue.push_chunk(1, np.zeros((1,6), dtype=np.float32))
    queue.pop(1)
    with pytest.raises(ValueError, match="Stale"):
        queue.push_chunk(1, np.zeros((1,6), dtype=np.float32))
    with pytest.raises(ValueError, match="capacity"):
        queue.push_chunk(2, np.zeros((11,6), dtype=np.float32))
