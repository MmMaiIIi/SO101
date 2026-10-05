import threading,time
import numpy as np
import pytest
from deployment.pi05_stack2.async_runtime import AlignedQueue,Candidate,AsyncRuntime
from deployment.pi05_stack2.action_adapter import JointBounds,filter_action,SafetyRejection
from deployment.pi05_stack2.config import MODEL_SHA256
from deployment.pi05_stack2.protocol import CONTRACT_VERSION


def candidate(seq=0,origin=0):
    return Candidate(origin,origin/30,seq,MODEL_SHA256,CONTRACT_VERSION,1.,.0,.25,
                     np.repeat(np.arange(50,dtype=np.float32)[:,None],6,axis=1),{})


def test_late_chunk_skips_elapsed_prefix_and_latest_wins():
    q=AlignedQueue(); q.accept(candidate(),7)
    target,c,i,_=q.pop(7)
    assert i==7 and target[0]==7
    q.accept(candidate(1,6),8)
    target,c,i,_=q.pop(8)
    assert c.inference_seq==1 and i==2 and target[0]==2
    with pytest.raises(ValueError,match='regression'): q.accept(candidate(),9)
    with pytest.raises(ValueError,match='replay'): q.pop(8)


def test_each_chunk_keeps_ten_execution_budget_and_cannot_wrap():
    q=AlignedQueue(); q.accept(candidate(),8)
    indices=[q.pop(t)[2] for t in range(8,18)]
    assert indices==list(range(8,18))
    assert q.pop(18) is None
    assert q.pop(51) is None
    assert not q.accept(candidate(1,1),60)
    assert q.pop(60) is None


def test_tracking_error_never_substituted_for_command_slew():
    b=JointBounds((-100,)*6,(100,)*6,(1,)*6,(30,)*6)
    action=np.full(6,10.5); previous=np.full(6,10.); measured=np.full(6,2.)
    filter_action(action,previous,b,measured_state=measured)
    with pytest.raises(SafetyRejection) as e:
        filter_action(action,measured,b,measured_state=measured,startup=True)
    assert e.value.report['transition_type']=='STARTUP_TRANSITION'
    b2=JointBounds((-100,)*6,(100,)*6,(1,)*6,(30,)*6,(5,)*6)
    with pytest.raises(SafetyRejection) as e:
        filter_action(action,previous,b2,measured_state=measured)
    assert {v['category'] for v in e.value.report['violations']}=={'TRACKING_ERROR_LIMIT'}


def test_saved_processor_float_roundoff_does_not_expand_physical_caps():
    b=JointBounds((-180,)*6,(180,)*6,(5.4505462646484375,)*6)
    prev=np.full(6,-99,np.float32)
    filter_action(prev+np.float32(5.4505462646484375+7.62939453125e-6),prev,b)
    with pytest.raises(SafetyRejection): filter_action(prev+5.4515,prev,b)
    assert b.max_delta_per_step[0]==5.4505462646484375


def test_blocked_inference_never_blocks_control_and_hold_stops():
    seen=[]
    def infer(obs):
        seen.append(threading.current_thread().name)
        time.sleep(.25)
        return np.zeros((50,6),np.float32),{}
    r=AsyncRuntime(infer,lambda tick:{'observation.state':np.zeros(6)},
        JointBounds((-1,)*6,(1,)*6,(1,)*6),hold_timeout_s=.07)
    result=r.run(30)['summary']
    assert result['watchdog_activations']==1 and result['control_ticks']<10
    assert result['motor_commands_sent']==0
    assert result['control_work_ms']['P99']<20
    assert seen==['so101-inference']
