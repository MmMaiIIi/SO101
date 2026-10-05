from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .logging_utils import RunLogger
from .protocol import RemotePolicy


def main():
    ap = argparse.ArgumentParser(description="Prerecorded robot-client preflight; no serial/motor access")
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument('--transport',choices=['png','binary'],default='png')
    args = ap.parse_args()
    paths = sorted(args.samples.glob("*.npz"))[:args.limit]
    if not paths or args.limit < 1:
        raise SystemExit("No replay samples selected")
    print("DRY RUN — NO MOTOR COMMANDS WILL BE SENT", flush=True)
    client = RemotePolicy(args.server, args.timeout,transport=args.transport)
    log = RunLogger()
    latencies, chunks = [], []
    try:
        for path in paths:
            data = np.load(path, allow_pickle=False)
            observation = {"observation.state": data["state"], "observation.images.front": data["front"],
                           "observation.images.wrist": data["wrist"]}
            actions, elapsed, result = client.infer_chunk(observation, str(data["instruction"]))
            log.log("dry_run_chunk", sample=path.name, roundtrip_latency_s=elapsed,
                    raw_decoded_action=actions.tolist(), executed_action=None, motor_commands_sent=0,
                    observation_sequence=result["observation_sequence"], checkpoint_sha256=result["checkpoint_sha256"])
            latencies.append(elapsed); chunks.append(actions)
            print(f"{path.name}: {elapsed:.3f}s actions={actions.shape} motors=0", flush=True)
        np.save(log.path / "actions.npy", np.stack(chunks))
        summary = {"queries": len(chunks), "median_roundtrip_s": float(np.median(latencies)),
                   "max_roundtrip_s": float(max(latencies)), "finite": bool(np.isfinite(chunks).all()),
                   "hardware_enabled": False, "motor_commands_sent": 0}
        (log.path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"run_dir={log.path}", flush=True)
    finally:
        client.close()
        log.close()


if __name__ == "__main__":
    main()
