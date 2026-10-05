from __future__ import annotations

import argparse
from pathlib import Path

from deployment.pi05_stack2.model_loader import load_policy


def main() -> int:
    p = argparse.ArgumentParser(description="Offline dataset inference; never opens hardware")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--episode", type=int, action="append", required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("models/so101-pi05-4228"))
    p.add_argument("--output", type=Path, default=Path("artifacts/offline_inference"))
    p.add_argument("--device",default="cuda")
    args = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from deployment.pi05_stack2.offline import run_episode_replay
    import torch
    torch.manual_seed(1000)
    policy, pre, post, report = load_policy(args.checkpoint,args.device)
    ds = LeRobotDataset(repo_id="local/so101-stack2-replay",root=args.dataset_root,episodes=args.episode,video_backend="pyav")
    args.output.mkdir(parents=True, exist_ok=True)
    for ep in args.episode:
        run_episode_replay(ds, ep, policy, pre, post, args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
