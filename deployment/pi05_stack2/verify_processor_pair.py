from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
import lerobot.policies.pi05.processor_pi05  # noqa: F401
from lerobot.processor.factory import load_pretrained_policy_processors
from lerobot.processor.relative_action_processor import to_absolute_actions, to_relative_actions
from lerobot.policies import prepare_observation_for_inference


def main() -> int:
    if len(sys.argv) not in (3,4):
        raise SystemExit("usage: verify_processor_pair.py CHECKPOINT OUTPUT_JSON [REAL_SAMPLES_DIRECTORY]")
    checkpoint, output = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cpu"
    cfg.text_tokenizer_name = str(checkpoint / "tokenizer")
    pre, post = load_pretrained_policy_processors(
        checkpoint, preprocessor_overrides={"device_processor": {"device": "cpu"}}
    )
    rng = np.random.default_rng(4228)
    state = torch.tensor(rng.uniform(-100, 100, size=(1, 6)), dtype=torch.float32)
    action = torch.tensor(rng.uniform(-100, 100, size=(1, 10, 6)), dtype=torch.float32)
    mask = [True, True, True, True, True, False]
    encoded = to_relative_actions(action, state, mask)
    decoded = to_absolute_actions(encoded, state, mask)
    max_error = float((decoded - action).abs().max())
    if max_error > 1e-5:
        raise RuntimeError(f"Relative arm / absolute gripper round-trip failed: {max_error}")
    images = {key: np.zeros((480, 640, 3), dtype=np.uint8) for key in
              ("observation.images.wrist", "observation.images.front")}
    raw = {"observation.state": state[0].numpy(), **images}
    model_input = prepare_observation_for_inference(raw, torch.device("cpu"),
                                                    "Stack the yellow block on top of the blue block.", "so101")
    prepared = pre(model_input)
    def describe(x):
        if isinstance(x, torch.Tensor): return {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
        if isinstance(x, dict): return {str(k): describe(v) for k, v in x.items()}
        return type(x).__name__
    # Use an arbitrary finite normalized model output to verify the saved inverse pipeline and anchor.
    dummy = torch.zeros((1, 6), dtype=torch.float32)
    restored = post(dummy)
    restored_action = restored
    if isinstance(restored_action, torch.Tensor):
        restored_action = restored_action.detach().float().cpu()
    if not torch.isfinite(restored_action).all():
        raise RuntimeError("Saved postprocessor produced non-finite action")
    report = {"checkpoint_revision": cfg.repo_id, "model_config_type": cfg.type,
              "feature_order": list(cfg.input_features), "action_names": cfg.action_feature_names,
              "normalization": {k: str(v) for k,v in cfg.normalization_mapping.items()},
              "chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps,
              "processor_pair_loaded": True, "preprocessed_observation": describe(prepared),
              "postprocessor_dummy_finite": True, "relative_arm_absolute_gripper_roundtrip_max_error": max_error,
              "contract_verified": True}
    if len(sys.argv) == 4:
        errors = []
        for path in sorted(Path(sys.argv[3]).glob("*.npz")):
            sample = np.load(path, allow_pickle=False)
            actual_state = torch.from_numpy(sample["state"].copy()).unsqueeze(0)
            actual_action = torch.from_numpy(sample["ground_truth"].copy()).unsqueeze(0)
            encoded = to_relative_actions(actual_action,actual_state,mask)
            recovered = to_absolute_actions(encoded,actual_state,mask)
            err = float((recovered-actual_action).abs().max())
            if err > 3e-5 or not torch.equal(encoded[:,:,5],actual_action[:,:,5]):
                raise RuntimeError(f"Real-data arm/gripper round-trip failed: {path.name}: {err}")
            errors.append(err)
        if not errors: raise RuntimeError("No real samples found")
        report["real_sample_count"] = len(errors)
        report["real_sample_relative_roundtrip_max_error"] = max(errors)
        report["gripper_remains_absolute_in_all_real_samples"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
