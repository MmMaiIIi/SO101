"""Derive a calibration envelope and teleop delta caps without modifying either source."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    calibration = json.loads(args.calibration.read_text())
    actions, states, episodes = [], [], []
    for path in sorted((args.dataset_root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action", "observation.state"]).to_pydict()
        actions.extend(table["action"]); states.extend(table["observation.state"])
        episodes.extend(zip(table["episode_index"], table["frame_index"]))
    a, s = np.asarray(actions), np.asarray(states)
    order = np.lexsort((np.array(episodes)[:,1], np.array(episodes)[:,0]))
    a, s = a[order], s[order]
    ep = np.array(episodes)[order,0]
    deltas = np.abs(np.diff(a, axis=0))[ep[1:] == ep[:-1]]
    if a.shape[1] != 6 or not np.isfinite(a).all() or not np.isfinite(s).all():
        raise ValueError("Expected finite SO101 six-dimensional trajectories")
    lower, upper, least_step = [], [], []
    for name in names:
        c = calibration[name]
        if c["range_min"] >= c["range_max"]:
            raise ValueError("Invalid calibration range")
        if name == "gripper":
            lower.append(0.0); upper.append(100.0)
            least_step.append(100 / (c["range_max"] - c["range_min"]))
        else:
            half_span = (c["range_max"] - c["range_min"]) * 180 / 4095
            lower.append(-half_span); upper.append(half_span); least_step.append(360 / 4095)
    p995 = np.percentile(deltas, 99.5, axis=0)
    caps = np.maximum(least_step, np.minimum(deltas.max(0), 1.5 * p995))
    report = {"joint_order": [n+".pos" for n in names], "lower": lower, "upper": upper,
              "max_delta_per_step": caps.tolist(), "control_hz": 30,
              "max_velocity_per_s": (caps*30).tolist(),
              "derivation": "saved calibration envelope, not certified mechanical stops; delta cap=max(encoder quantum,min(observed max,1.5*p99.5 absolute per-frame teleop delta))",
              "hardware_limit": {"status":"REQUIRES_HARDWARE_VERIFICATION"},
              "calibration_limit": {"lower":lower,"upper":upper,"live_match":"REQUIRES_HARDWARE_VERIFICATION"},
              "runtime_delta_limit":caps.tolist(),"runtime_velocity_limit":(caps*30).tolist(),
              "software_gate":"UNVERIFIED_GT_ORACLE",
              "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
              "sample_frames": len(a), "episodes": len(set(ep)),
              "data_split": "recorded teleop distribution; not asserted as training split",
              "delta_p99_5": p995.tolist(), "state_p1_p50_p99": np.percentile(s,[1,50,99],axis=0).tolist()}
    def stats(value):
        return dict(zip(['min','p0_1','p1','p50','p99','p99_9','max'],
                        np.percentile(value,[0,.1,1,50,99,99.9,100],axis=0).tolist()))
    report['data_operating_envelope']={'state':stats(s),'absolute_action_target':stats(a),
                                      'absolute_per_step_action_delta':stats(deltas)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"profile={args.output}; frames={len(a)}; episodes={len(set(ep))}; hardware writes=0")


if __name__ == "__main__":
    main()
