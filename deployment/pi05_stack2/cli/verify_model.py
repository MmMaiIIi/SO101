from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from deployment.pi05_stack2.model_loader import load_strict_cpu
from deployment.pi05_stack2.observation_adapter import CAMERA_ORDER
from deployment.pi05_stack2.policy_runner import PolicyRunner


def main() -> int:
    p = argparse.ArgumentParser(description="Strictly load the checkpoint and run one no-motor forward")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("artifacts/model_forward_4228.json"))
    args = p.parse_args()
    report = {"hardware_commands_sent": 0, "test_split_verified": False}
    t0 = time.perf_counter()
    try:
        policy, pre, post, model_report = load_strict_cpu(args.checkpoint)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo_id=None, root=args.dataset_root)
        ep = np.asarray(ds.hf_dataset["episode_index"])
        indices = np.flatnonzero(ep == args.episode)
        if args.frame >= len(indices):
            raise IndexError(f"episode {args.episode} does not contain frame {args.frame}")
        sample = ds[int(indices[args.frame])]
        raw = {"observation.state": np.asarray(sample["observation.state"], dtype=np.float32)}
        for key in CAMERA_ORDER:
            image = sample[key].detach().cpu().numpy()
            if image.shape[0] == 3:
                image = np.moveaxis(image, 0, -1)
            if image.dtype != np.uint8:
                image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
            raw[key] = image
        task = sample.get("task", "")
        if not isinstance(task, str):
            task = str(ds.meta.tasks[int(sample["task_index"])]["task"])
        runner = PolicyRunner(policy, pre, post, "cpu")
        action, latency = runner.infer_one(raw | {"task": task})
        report.update({"load": model_report, "forward_ok": True, "forward_device": "cpu",
                       "forward_latency_s": latency, "total_elapsed_s": time.perf_counter()-t0,
                       "input_episode": args.episode, "input_frame": args.frame,
                       "input_task": task, "input_images": {k: list(raw[k].shape) for k in CAMERA_ORDER},
                       "action_shape": list(action.shape), "action_finite": bool(np.isfinite(action).all()),
                       "action_values": action.tolist(), "action_gripper_in_0_100": bool(0 <= action[5] <= 100),
                       "test_split_verified": False,
                       "note": "Input is a real recorded observation, but the published artifacts omit the official train/val/test episode ID manifest."})
    except Exception as exc:
        report.update({"forward_ok": False, "error_type": type(exc).__name__, "error": str(exc),
                       "total_elapsed_s": time.perf_counter()-t0})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
