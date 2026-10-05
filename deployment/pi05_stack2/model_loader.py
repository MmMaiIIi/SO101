from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import time
from typing import Any


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: Path, *, expected_model_sha256: str) -> dict[str, Any]:
    manifest_path = path / "MODEL_MANIFEST.json"
    config_path = path / "config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Checkpoint must include MODEL_MANIFEST.json and config.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = path / "model.safetensors"
    actual = sha256_file(model_path)
    if actual != expected_model_sha256:
        raise ValueError(f"Checkpoint model hash mismatch: {actual}")
    if manifest.get("files", {}).get("model.safetensors", {}).get("sha256") != actual:
        raise ValueError("MODEL_MANIFEST model hash does not match the downloaded weight file")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("type") != "pi05" or cfg.get("chunk_size") != 50 or cfg.get("n_action_steps") != 10:
        raise ValueError("Checkpoint config does not match the frozen SO101 π0.5 deployment contract")
    names = cfg.get("action_feature_names", [])
    expected = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    if names != expected:
        raise ValueError(f"Unexpected action ordering: {names}")
    return {"model_sha256": actual, "model_bytes": model_path.stat().st_size,
            "config": cfg, "manifest": manifest,
            "processor_files_present": all((path / n).is_file() for n in (
                "policy_preprocessor.json", "policy_postprocessor.json",
                "policy_preprocessor_step_3_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))}


def _mapped_safetensors(path: Path):
    """Map tensor bytes from safetensors without an eager multi-GB CPU copy."""
    import torch

    dtypes = {"BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16,
              "I32": torch.int32, "I64": torch.int64, "F16": torch.float16,
              "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64}
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    payload_offset = 8 + header_size
    backing = torch.from_file(str(path), shared=False, size=path.stat().st_size, dtype=torch.uint8)
    state = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        state[name] = backing[payload_offset + start:payload_offset + end].view(dtypes[meta["dtype"]]).reshape(meta["shape"])
    return backing, state


def load_strict_cpu(checkpoint: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Strictly load every checkpoint tensor using file-backed CPU storage.

    This is useful on machines where the checkpoint is larger than available
    RAM. It keeps weights mapped from the original file and rejects any key
    mismatch; it is not a quantized or partial load.
    """
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401: register saved pipeline step.
    from lerobot.processor.factory import load_pretrained_policy_processors
    from lerobot.processor.relative_action_processor import bind_relative_anchor

    from .config import MODEL_SHA256
    verified = verify_checkpoint(checkpoint, expected_model_sha256=MODEL_SHA256)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "meta"
    config.text_tokenizer_name = str(checkpoint / "tokenizer")
    t0 = time.perf_counter()
    with torch.device("meta"):
        policy = PI05Policy(config)
    backing, original_state = _mapped_safetensors(checkpoint / "model.safetensors")
    fixed = policy._fix_pytorch_state_dict_keys(original_state, policy.config)
    remapped = {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed.items()}
    remapped = policy._prepare_pretrained_state_dict(remapped)
    loaded = policy.load_state_dict(remapped, strict=True, assign=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"Strict weight load mismatch: missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}")
    for module in policy.modules():
        for name, value in list(module._buffers.items()):
            if value is None or value.device.type != "meta":
                continue
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
    policy._checkpoint_mmap_backing = backing
    pre, post = load_pretrained_policy_processors(
        checkpoint, preprocessor_overrides={"device_processor": {"device": "cpu"}}
    )
    bind_relative_anchor(policy, pre)
    policy.eval()
    verified.update({"strict_load": True, "load_state_dict_calls": 1,
                     "missing_keys": loaded.missing_keys, "unexpected_keys": loaded.unexpected_keys,
                     "load_method": "file-backed-mmap + meta model + strict assign",
                     "parameter_count": sum(p.numel() for p in policy.parameters()),
                     "parameter_dtype_counts": {}, "strict_cpu_load_seconds": time.perf_counter() - t0})
    for p in policy.parameters():
        name = str(p.dtype)
        verified["parameter_dtype_counts"][name] = verified["parameter_dtype_counts"].get(name, 0) + p.numel()
    if torch.cuda.is_available():
        verified["cuda_available"] = True
        verified["cuda_device"] = torch.cuda.get_device_name(0)
    else:
        verified["cuda_available"] = False
    return policy, pre, post, verified


def load_policy(checkpoint: Path, device: str = "cuda"):
    """Strict CPU load, then move the complete paired policy to the chosen device."""
    import torch
    from lerobot.processor.factory import load_pretrained_policy_processors
    from lerobot.processor.relative_action_processor import bind_relative_anchor

    policy, pre, post, report = load_strict_cpu(checkpoint)
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.reset_peak_memory_stats(selected)
        policy.to(selected)
        policy.config.device = str(selected)
        pre, post = load_pretrained_policy_processors(
            checkpoint, preprocessor_overrides={"device_processor": {"device": str(selected)}}
        )
        bind_relative_anchor(policy, pre)
        report["cuda_allocated_bytes_after_load"] = torch.cuda.memory_allocated(selected)
        report["cuda_total_bytes"] = torch.cuda.get_device_properties(selected).total_memory
    report["device"] = str(selected)
    return policy, pre, post, report
