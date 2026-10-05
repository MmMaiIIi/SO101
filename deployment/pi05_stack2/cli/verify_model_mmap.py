from __future__ import annotations

"""Strictly load Pi0.5 weights through file-backed CPU mappings."""

import argparse
import json
import struct
import time
from pathlib import Path

import torch

DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def mapped_state_dict(path: Path) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Map safetensors storage without eagerly copying its multi-GB payload."""
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    payload_offset = 8 + header_size
    backing = torch.from_file(str(path), shared=False, size=path.stat().st_size, dtype=torch.uint8)
    tensors: dict[str, torch.Tensor] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = DTYPES[meta["dtype"]]
        start, end = meta["data_offsets"]
        view = backing[payload_offset + start : payload_offset + end].view(dtype)
        tensors[name] = view.reshape(meta["shape"])
    return backing, tensors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401
    from lerobot.processor.factory import load_pretrained_policy_processors
    from lerobot.processor.relative_action_processor import bind_relative_anchor

    t0 = time.perf_counter()
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = "meta"
    config.text_tokenizer_name = str(args.checkpoint / "tokenizer")
    with torch.device("meta"):
        policy = PI05Policy(config)

    backing, raw = mapped_state_dict(args.checkpoint / "model.safetensors")
    fixed = policy._fix_pytorch_state_dict_keys(raw, policy.config)
    remapped = {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed.items()}
    remapped = policy._prepare_pretrained_state_dict(remapped)
    result = policy.load_state_dict(remapped, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict load mismatch: {result}")
    # Some Transformers modules have non-persistent scalar buffers that are
    # intentionally absent from the checkpoint. Recreate these from the
    # module's recorded constructor scalar, matching its normal CPU init.
    for module in policy.modules():
        for name, value in list(module._buffers.items()):
            if value is not None and value.device.type == "meta":
                scalar = getattr(module, f"scalar_{name}", None)
                if scalar is not None:
                    concrete = torch.tensor(scalar, dtype=torch.get_default_dtype())
                elif name == "position_ids":
                    concrete = torch.arange(value.shape[-1], dtype=torch.long).expand(value.shape)
                elif name in ("inv_freq", "original_inv_freq") and hasattr(module, "compute_default_rope_parameters"):
                    concrete = module.compute_default_rope_parameters(module.config, device="cpu")[0]
                    if name == "original_inv_freq":
                        concrete = concrete.clone()
                else:
                    raise RuntimeError(f"Uninitialized non-persistent buffer: {type(module).__name__}.{name}")
                module._buffers[name] = concrete
    policy.config.device = "cpu"
    pre, post = load_pretrained_policy_processors(
        args.checkpoint, preprocessor_overrides={"device_processor": {"device": "cpu"}}
    )
    bind_relative_anchor(policy, pre)
    policy.eval()
    if any(p.device.type != "cpu" for p in policy.parameters()):
        raise RuntimeError("one or more parameters were not assigned to CPU")

    report = {
        "strict_load": True,
        "load_method": "file-backed-mmap + meta model + strict assign",
        "checkpoint": "${SO101_PI05_CHECKPOINT_4228}",
        "state_dict_keys": len(remapped),
        "missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys,
        "parameter_count": sum(p.numel() for p in policy.parameters()),
        "parameter_dtype_counts": {},
        "elapsed_s": time.perf_counter() - t0,
        "forward_performed": False,
    }
    for p in policy.parameters():
        dtype = str(p.dtype)
        report["parameter_dtype_counts"][dtype] = report["parameter_dtype_counts"].get(dtype, 0) + p.numel()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset_root is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import numpy as np
        from deployment.pi05_stack2.config import CAMERA_ORDER
        from deployment.pi05_stack2.offline import _as_rgb_hwc
        from deployment.pi05_stack2.policy_runner import PolicyRunner

        dataset = LeRobotDataset(repo_id="local/so101-stack2", root=args.dataset_root, episodes=[args.episode])
        sample = dataset[0]
        state = np.asarray(sample["observation.state"], dtype=np.float32)
        transition = {"observation.state": state}
        images = {}
        for key in CAMERA_ORDER:
            images[key] = _as_rgb_hwc(sample[key])
            transition[key] = images[key]
        task = str(sample.get("task", ""))
        runner = PolicyRunner(policy, pre, post, "cpu")
        action, latency = runner.infer_one(transition | {"task": task})
        report.update({
            "forward_performed": True,
            "sample_episode": args.episode,
            "sample_instruction": task,
            "sample_state_shape": list(state.shape),
            "sample_camera_shapes": {k: list(v.shape) for k, v in images.items()},
            "inference_latency_s": latency,
            "predicted_action_shape": list(action.shape),
            "predicted_action_min": action.tolist(),
            "predicted_action_max": action.tolist(),
            "predicted_action_finite": bool(np.isfinite(action).all()),
            "ground_truth_action": np.asarray(sample["action"], dtype=np.float32).tolist(),
        })
        from PIL import Image
        montage = Image.new("RGB", (640, 240), "white")
        for i, key in enumerate(CAMERA_ORDER):
            im = Image.fromarray(images[key]); im.thumbnail((320, 240))
            montage.paste(im, (i * 320, 0))
        montage.save(args.report.parent / "strict_load_sample_montage.jpg", quality=92)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    del backing
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
