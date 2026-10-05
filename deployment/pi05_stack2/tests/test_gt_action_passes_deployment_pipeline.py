"""Real saved processors and GT replay; never relax safety to manufacture a pass."""
import os
from pathlib import Path
import json
import numpy as np
import pytest


def test_gt_action_passes_deployment_pipeline():
    checkpoint=os.environ.get('SO101_TEST_CHECKPOINT')
    dataset=os.environ.get('SO101_TEST_DATASET')
    if not checkpoint or not dataset: pytest.skip('Requires actual saved checkpoint processors and real GT parquet')
    from tools.gt_deployment_oracle import saved_action_roundtrip
    from lerobot.processor.factory import load_pretrained_policy_processors
    import pyarrow.parquet as pq
    from deployment.pi05_stack2.action_adapter import JointBounds,filter_action,SafetyRejection
    from deployment.pi05_stack2.action_queue import SafeActionQueue
    pre,post=load_pretrained_policy_processors(Path(checkpoint),preprocessor_overrides={'device_processor':{'device':'cpu'}})
    profile=json.loads(Path('artifacts/safety_profile.json').read_text())
    bounds=JointBounds(tuple(profile['lower']),tuple(profile['upper']),tuple(profile['max_delta_per_step']))
    rows=[]
    for p in sorted((Path(dataset)/'data').rglob('*.parquet')):
        rows.extend(pq.read_table(p,columns=['episode_index','frame_index','action','observation.state']).to_pylist())
    rows.sort(key=lambda r:(r['episode_index'],r['frame_index']))
    accepted=rejected=0
    for offset in range(0,len(rows)-50,50):
        batch=rows[offset:offset+50]
        if batch[0]['episode_index']!=batch[-1]['episode_index']: continue
        state=np.array(batch[0]['observation.state'],np.float32)
        targets=np.array([r['action'] for r in batch],np.float32)
        decoded,normalized,relative=saved_action_roundtrip(pre,post,state,targets)
        np.testing.assert_allclose(decoded,targets,atol=3e-5,rtol=0)
        np.testing.assert_allclose(relative[:,:5],targets[:,:5]-state[:5],atol=1e-6)
        np.testing.assert_array_equal(relative[:,5],targets[:,5])
        # Golden: every future target uses the same anchor, never cumsum(relative).
        np.testing.assert_allclose(decoded[:,:5],relative[:,:5]+state[:5],atol=3e-5)
        q=SafeActionQueue(10); q.push_chunk(offset,decoded[:10])
        previous=state if offset==0 or rows[offset-1]['episode_index']!=batch[0]['episode_index'] else np.array(rows[offset-1]['action'])
        for i in range(10):
            value=q.pop(offset)
            try:
                filtered,_=filter_action(value,previous,bounds)
                np.testing.assert_array_equal(filtered,value); accepted+=1
            except SafetyRejection as e:
                # Unsafe source GT is explicitly reported, not clipped or called deployable.
                assert e.report['violations']; rejected+=1
            previous=value
    assert accepted>1000
    assert accepted+rejected>1000
