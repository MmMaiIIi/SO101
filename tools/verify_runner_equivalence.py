"""Seeded real-model regression: reused observation preprocessing preserves native queue semantics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from deployment.pi05_stack2.model_loader import load_policy
from deployment.pi05_stack2.policy_runner import PolicyRunner


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--samples',type=Path,default=Path('artifacts/deployment_samples'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/runner_equivalence.json'))
    args=ap.parse_args()
    policy,pre,post,_=load_policy(args.checkpoint,'cuda'); runner=PolicyRunner(policy,pre,post,'cuda')
    checks=[]
    for path in sorted(args.samples.glob('*.npz'))[::10]:
        d=np.load(path,allow_pickle=False)
        obs={'observation.state':d['state'],'observation.images.front':d['front'],'observation.images.wrist':d['wrist'],
             'task':str(d['instruction'])}
        runner.reset(); torch.manual_seed(1000)
        original=np.stack([runner.infer_one(obs)[0] for _ in range(10)])
        runner.reset(); torch.manual_seed(1000)
        optimized,_=runner.infer_chunk(obs)
        np.testing.assert_allclose(optimized,original,atol=1e-5,rtol=0)
        runner.reset(); torch.manual_seed(1000)
        full,_=runner.infer_chunk(obs,full_horizon=True)
        assert full.shape==(50,6) and policy.config.n_action_steps==10
        np.testing.assert_allclose(full[:10],original,atol=1e-5,rtol=0)
        checks.append({'sample':path.name,'max_absolute_difference':float(np.max(np.abs(original-optimized))),
                       'native_async_prefix_difference':float(np.max(np.abs(full[:10]-original))),
                       'native_prediction_length':len(full),'n_action_steps_unchanged':10})
    report={'pass':True,'checks':checks,'horizon_unchanged':10,'hardware_commands_sent':0}
    args.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report))


if __name__=='__main__': main()
