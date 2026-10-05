"""Real-data CUDA forward, decoded action comparisons and resource report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from deployment.pi05_stack2.model_loader import load_policy
from deployment.pi05_stack2.policy_runner import PolicyRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("artifacts/cuda_inference"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1000)
    start = time.perf_counter()
    policy, pre, post, report = load_policy(args.checkpoint, "cuda")
    report["load_latency_s"] = time.perf_counter() - start
    runner = PolicyRunner(policy, pre, post, "cuda")
    rows = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    for path in sorted(args.samples.glob("*.npz")):
        sample = np.load(path, allow_pickle=False)
        obs = {"observation.state": sample["state"], "observation.images.wrist": sample["wrist"],
               "observation.images.front": sample["front"], "task": str(sample["instruction"])}
        actions, latency = runner.infer_chunk(obs)
        gt = sample["ground_truth"]
        np.savez_compressed(args.output / path.name, predicted=actions, ground_truth=gt, state=sample["state"])
        rows.append({"sample": path.name, "instruction": str(sample["instruction"]), "latency_s": latency,
                     "shape": list(actions.shape), "finite": bool(np.isfinite(actions).all()),
                     "first_action_mae": float(np.mean(np.abs(actions[0]-gt[0]))),
                     "ten_action_mae": float(np.mean(np.abs(actions-gt))),
                     "min": actions.min(0).tolist(), "max": actions.max(0).tolist(),
                     "gripper_range": [float(actions[:,5].min()), float(actions[:,5].max())],
                     "split": "not asserted as held-out; exact test membership unavailable"})
        fig, axes = plt.subplots(6, 1, figsize=(9, 10), sharex=True)
        for j, ax in enumerate(axes):
            ax.plot(gt[:,j], label="recorded"); ax.plot(actions[:,j], label="policy")
            if j == 0: ax.legend()
        fig.tight_layout(); fig.savefig(args.output / (path.stem + ".png")); plt.close(fig)
        montage = Image.new("RGB", (1280, 480))
        montage.paste(Image.fromarray(sample["front"]), (0,0)); montage.paste(Image.fromarray(sample["wrist"]), (640,0))
        montage.save(args.output / (path.stem + ".jpg"))
        print(json.dumps(rows[-1]), flush=True)
    report.update({"samples": rows, "sample_count": len(rows),
                   "cold_inference_latency_s": rows[0]["latency_s"],
                   "warm_latency_median_s": float(np.median([r["latency_s"] for r in rows[1:]])),
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "no_nan_inf": all(r["finite"] for r in rows), "motor_commands_sent": 0})
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
