import numpy as np
import pytest
from deployment.pi05_stack2.hardware import SO101Hardware
from deployment.pi05_stack2.cli.run_robot import stage_action_budget
from deployment.pi05_stack2.action_queue import SafeActionQueue


def test_hardware_disabled_rejects_write_before_bus_access():
    backend = SO101Hardware.__new__(SO101Hardware)
    backend.motor_output = False
    with pytest.raises(RuntimeError,match="guard disabled"):
        backend.send_target(np.zeros(6))


def test_stages_cap_first_physical_actions():
    assert [stage_action_budget(s,999) for s in range(5)] == [0,0,1,5,10]


def test_invalid_chunk_clears_preexisting_targets():
    queue = SafeActionQueue(10)
    queue.push_chunk(1,np.zeros((10,6)))
    with pytest.raises(ValueError): queue.push_chunk(2,np.full((10,6),np.nan))
    assert len(queue) == 0
