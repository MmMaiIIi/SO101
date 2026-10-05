from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import ACTION_ORDER, CAMERA_ORDER
from .policy_runner import PolicyRunner


def _as_rgb_hwc(value) -> np.ndarray:
    arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"Expected one image tensor, got shape {arr.shape}")
    if arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            raise ValueError("Dataset image must be uint8 [0,255] or float [0,1]")
        arr = np.rint(arr * 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def run_episode_replay(dataset, episode: int, policy, preprocessor, postprocessor,
                       output_root: Path, model_report: dict) -> Path:
    """Run strict prerecorded inference with no hardware output; save predictions and plots."""
    from .observation_adapter import make_observation

    runner = PolicyRunner(policy,preprocessor,postprocessor,model_report.get("device","cpu"))
    runner.reset()
    ep_ids = np.asarray(dataset.hf_dataset["episode_index"])
    indices = np.flatnonzero(ep_ids == episode).tolist()
    if not indices:
        raise ValueError(f"Episode {episode} is absent from dataset")
    rows, latencies, keyframes = [], [], {}
    selected = [0,len(indices)//4,len(indices)//2,3*len(indices)//4,len(indices)-1]
    for position,ix in enumerate(indices):
        sample = dataset[ix]
        state = np.asarray(sample["observation.state"], dtype=np.float32)
        cameras = {key: _as_rgb_hwc(sample[key]) for key in CAMERA_ORDER}
        if position in selected:
            keyframes[position] = cameras["observation.images.front"].copy()
        task = sample.get("task", sample.get("task_index", ""))
        if not isinstance(task, str):
            tasks = getattr(dataset.meta, "tasks", None)
            if tasks is None:
                raise ValueError("Task instruction could not be resolved from dataset metadata")
            task = str(tasks[int(task)]["task"])
        transition = {"observation.state": state, **cameras}
        # Feed through the same saved processors/model/postprocessor as deployment.
        action, latency = runner.infer_one(transition | {"task": task})
        gt = np.asarray(sample["action"], dtype=np.float32)
        rows.append((int(sample.get("frame_index", len(rows))), state, gt, action,
                     float(sample.get("timestamp", len(rows) / 30))))
        latencies.append(latency)
    out = output_root / f"episode_{episode:06d}"
    out.mkdir(parents=True, exist_ok=True)
    frame, state, gt, pred, ts = zip(*rows)
    np.savez_compressed(out / "actions.npz", frame_index=frame, timestamp_s=ts,
                        state=np.stack(state), ground_truth=np.stack(gt), prediction=np.stack(pred))
    metrics = {"episode": episode, "task": task, "frames": len(rows),
               "mean_latency_s": float(np.mean(latencies)), "p95_latency_s": float(np.percentile(latencies, 95)),
               "first_action_mae": float(np.mean(np.abs(pred[0] - gt[0]))),
               "chunk_mae": float(np.mean(np.abs(np.stack(pred) - np.stack(gt)))),
               "predicted_min": np.stack(pred).min(axis=0).tolist(),
               "predicted_max": np.stack(pred).max(axis=0).tolist(),
               "nan_or_inf": not bool(np.isfinite(np.stack(pred)).all()),
               "action_order": list(ACTION_ORDER), "hardware_commands_sent": 0,
               "checkpoint_sha256": model_report["model_sha256"]}
    metrics["split"] = "real-data replay; exact train/validation/test membership was not published"
    (out / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
        for j, ax in enumerate(axes):
            ax.plot(ts, np.stack(gt)[:, j], label="recorded")
            ax.plot(ts, np.stack(pred)[:, j], label="policy")
            ax.set_ylabel(ACTION_ORDER[j].replace(".pos", ""))
            if j == 0: ax.legend(loc="upper right")
        axes[-1].set_xlabel("episode time (s)")
        fig.tight_layout()
        fig.savefig(out / "action_comparison.png", dpi=140)
        plt.close(fig)
        from PIL import Image, ImageDraw
        thumbs = []
        for i in selected:
            im = Image.fromarray(keyframes[i])
            im.thumbnail((320, 240))
            thumbs.append(im.copy())
        sheet = Image.new("RGB", (320 * len(thumbs), 270), "white")
        draw = ImageDraw.Draw(sheet)
        for j, im in enumerate(thumbs):
            sheet.paste(im, (j*320, 0))
            draw.text((j*320+4, 244), f"t={ts[selected[j]]:.2f}s", fill="black")
        sheet.save(out / "front_montage.jpg", quality=92)
    except ImportError:
        metrics["plots"] = "skipped: matplotlib/Pillow unavailable"
    return out
