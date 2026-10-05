from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path


EXPECTED = {
    "4228": {
        "revision": "main",
        "hub_commit": "85fc6c4990878185ad0248056cde8facee39d774",
        "weight_commit": "f71e25179dd39e3b986c53a89eaefa5503296390",
        "sha256": "f54e8cec0217bc5e2baebc095f06979675209c955093e7d5916fc5587eb098c4",
        "step": 4228,
    },
    "2500": {
        "revision": "step-2500",
        "hub_commit": "f3cc7ff73c6c203c4d2520d0b4b5a1fc122e8245",
        "weight_commit": "f3cc7ff73c6c203c4d2520d0b4b5a1fc122e8245",
        "sha256": "ca6731a374963c641f3ece8d624f2088cba08ac7279ec55c213cc233c2261759",
        "step": 2500,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(key: str, spec: dict, root: Path, output: Path) -> dict:
    manifest = json.loads((root / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    expected_files = manifest["files"]
    results = {}
    for name, info in expected_files.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {name}")
        actual = sha256(path)
        if path.stat().st_size != info["size"] or actual != info["sha256"]:
            raise ValueError(f"Manifest mismatch for {name}")
        results[name] = {"size_bytes": path.stat().st_size, "sha256": actual}
    model = results.get("model.safetensors")
    if model is None or model["sha256"] != spec["sha256"]:
        raise ValueError("Model weight checksum did not match the requested published revision")
    actual_files = [p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts]
    incomplete = [str(p.relative_to(root)) for p in root.rglob("*.incomplete")]
    if incomplete:
        raise ValueError(f"Incomplete HF transfers remain: {incomplete}")
    tracked_bytes = sum(p.stat().st_size for p in actual_files)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    result = {
        "repo_id": "WoWjx/so101-stack2-pi05-full-relative",
        "revision": spec["revision"],
        "resolved_hub_commit": spec["hub_commit"],
        "weight_file_commit": spec["weight_commit"],
        "training_step": spec["step"],
        "local_directory": "${SO101_PI05_CHECKPOINT_4228}" if key == "4228" else "${SO101_PI05_CHECKPOINT_2500}",
        "tracked_file_count": len(actual_files),
        "tracked_total_bytes": tracked_bytes,
        "lfs_weight_present_and_complete": model["size_bytes"] == 9354050784,
        "incomplete_transfer_files": incomplete,
        "config_present": True,
        "paired_processors_present": all((root / n).is_file() for n in (
            "policy_preprocessor.json", "policy_postprocessor.json",
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors")),
        "model_type": config.get("type"),
        "model_sha256": model["sha256"],
        "manifest_file_hashes_verified": results,
        "readme_only_head_commit_relation": (
            "main points to a documentation-only commit after the weight upload commit; "
            "the model tensor file is byte-identical to that upload revision."
            if key == "4228" else None
        ),
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-4228", type=Path, required=True)
    parser.add_argument("--checkpoint-2500", type=Path, required=True)
    args = parser.parse_args()
    EXPECTED["4228"]["directory"] = args.checkpoint_4228
    EXPECTED["2500"]["directory"] = args.checkpoint_2500
    root = Path(__file__).resolve().parents[1] / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    for key, spec in EXPECTED.items():
        report = verify(key, spec, spec["directory"], root / f"model_verification_{key}.json")
        print(key, report["model_sha256"], report["tracked_file_count"], report["tracked_total_bytes"])
