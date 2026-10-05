import numpy as np
import pytest
from deployment.pi05_stack2.action_adapter import JointBounds,filter_action,SafetyRejection
from deployment.pi05_stack2.action_queue import SafeActionQueue
from deployment.pi05_stack2.safety import Watchdog
from deployment.pi05_stack2.cli.preflight import validate_motor_gate


def test_structured_rejection_keeps_exact_unclipped_target():
    bounds=JointBounds((-1,)*5+(0,),(1,)*5+(100,),(1,)*6,(10,)*6)
    action=np.array([2,0,0,0,0,101.],np.float32)
    with pytest.raises(SafetyRejection) as failure:
        filter_action(action,np.zeros(6),bounds,context={'observation_state':[0]*6})
    r=failure.value.report
    assert r['current_target']==action.tolist()
    assert {v['category'] for v in r['violations']}=={'POSITION_LIMIT','GRIPPER_LIMIT','DELTA_LIMIT','VELOCITY_LIMIT'}
    assert next(v for v in r['violations'] if v['category']=='POSITION_LIMIT')['violation_amount']==1


def test_nonfinite_clock_and_action_fail_closed():
    from deployment.pi05_stack2.action_adapter import rejection_category
    assert rejection_category(TimeoutError('timed out'))=='POLICY_TIMEOUT'
    assert rejection_category(ConnectionError('connection closed'))=='NETWORK_OR_DEVICE_ERROR'
    assert rejection_category(TimeoutError('STALE_CAMERA'))=='STALE_CAMERA'
    for ts in (float('nan'),float('inf')):
        with pytest.raises(ValueError): Watchdog().check(state_ts=ts,camera_ts=(10,),now=10)
    with pytest.raises(SafetyRejection) as failure:
        filter_action(np.full(6,np.nan),np.zeros(6),JointBounds((-1,)*6,(1,)*6,(1,)*6))
    assert failure.value.report['violations'][0]['category']=='NAN_INF'


def test_stale_sequence_clears_pending_actions():
    q=SafeActionQueue(10); q.push_chunk(3,np.zeros((10,6)))
    with pytest.raises(ValueError): q.push_chunk(2,np.zeros((1,6)))
    assert len(q)==0


def test_motor_gate_requires_actual_full_preflight():
    with pytest.raises(ValueError,match='preflight-report'):
        validate_motor_gate(None,None,None,None)


def test_binary_transport_preserves_both_real_pixel_orders():
    from deployment.pi05_stack2.protocol import encode_binary_request,decode_binary_request
    from deployment.pi05_stack2.config import CAMERA_ORDER
    rng=np.random.default_rng(4228)
    obs={'observation.state':np.arange(6,dtype=np.float32),
         **{k:rng.integers(0,256,(480,640,3),dtype=np.uint8) for k in CAMERA_ORDER}}
    payload=encode_binary_request(obs,0,'test','Stack the yellow block on top of the blue block.')
    _,decoded=decode_binary_request(payload)
    for k in obs: np.testing.assert_array_equal(decoded[k],obs[k])
    with pytest.raises(ValueError): decode_binary_request(payload[:-1])


def test_queue_has_finite_time_to_live(monkeypatch):
    import deployment.pi05_stack2.action_queue as module
    monkeypatch.setattr(module.time,'monotonic',lambda:10.)
    q=SafeActionQueue(10); q.push_chunk(0,np.zeros((10,6)))
    monkeypatch.setattr(module.time,'monotonic',lambda:11.)
    with pytest.raises(RuntimeError,match='expired'): q.pop(0)
    assert len(q)==0


def test_nan_camera_stamp_rejected():
    from deployment.pi05_stack2.observation_adapter import TimedFrame,make_observation
    from deployment.pi05_stack2.config import CAMERA_ORDER
    frames={k:TimedFrame(np.zeros((480,640,3),np.uint8),float('nan')) for k in CAMERA_ORDER}
    with pytest.raises(TimeoutError): make_observation(np.zeros(6),frames,'Stack yellow on blue.')


def test_explicit_motor_flag_cannot_bypass_unresolved_profile(tmp_path):
    import json,time
    from deployment.pi05_stack2.cli.preflight import digest
    devices=tmp_path/'devices.json'; devices.write_text('{}')
    calibration=tmp_path/'cal.json'; calibration.write_text('{}')
    profile=tmp_path/'profile.json'; profile.write_text('{"software_gate":"BLOCKED"}')
    report=tmp_path/'preflight.json'
    report.write_text(json.dumps({'status':'PASS','timestamp_unix_s':time.time(),
        'artifact_hashes':{'devices':digest(devices),'calibration':digest(calibration),'safety_profile':digest(profile)},
        'gates':{k:'PASS' for k in ('calibration','camera_mapping','joint_mapping','contract','safety','state_distribution')}}))
    with pytest.raises(ValueError,match='GT oracle'):
        validate_motor_gate(report,devices,calibration,profile)
