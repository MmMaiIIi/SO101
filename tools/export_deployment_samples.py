"""Export real trajectory observations for deployment regression without changing raw data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--output", type=Path, default=Path("artifacts/deployment_samples"))
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from deployment.pi05_stack2.offline import _as_rgb_hwc
    dataset = LeRobotDataset(repo_id="local/stack2-contract", root=args.dataset_root,
                            episodes=args.episodes, video_backend="pyav")
    episode_ids = np.asarray(dataset.hf_dataset["episode_index"])
    all_actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for ep in args.episodes:
        indices = np.flatnonzero(episode_ids == ep)
        if not len(indices):
            raise ValueError(f"Missing episode {ep}")
        for frac in (0.0, 0.25, 0.5, 0.75, 0.95):
            position = int(frac * (len(indices) - 1))
            index = int(indices[position])
            sample = dataset[index]
            gt_indices = indices[np.minimum(np.arange(position, position + 10), len(indices)-1)]
            out = args.output / f"ep{ep:03d}_frame{position:06d}.npz"
            np.savez_compressed(out, state=np.asarray(sample["observation.state"], dtype=np.float32),
                                wrist=_as_rgb_hwc(sample["observation.images.wrist"]),
                                front=_as_rgb_hwc(sample["observation.images.front"]),
                                ground_truth=all_actions[gt_indices], instruction=str(sample["task"]))
            rows.append({"episode": ep, "frame_index": position, "timestamp_s": float(sample["timestamp"]),
                         "instruction": str(sample["task"]), "file": out.name,
                         "split": "unverified: exact published training/test episode IDs unavailable"})
    (args.output / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Exported {len(rows)} real observations; raw dataset unchanged")


if __name__ == "__main__":
    main()
