from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deployment.pi05_stack2.logging_utils import RunLogger
from deployment.pi05_stack2.sim_adapter import make_stackcube_env, policy_degrees_to_sim_radians, sim_observation_to_policy


def main() -> int:
    p = argparse.ArgumentParser(description="Run the pinned SO101-Nexus StackCube environment")
    p.add_argument("--mode", choices=("random", "policy"), default="random")
    p.add_argument("--checkpoint", type=Path, default=Path("models/so101-pi05-4228"))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", choices=("red", "blue", "yellow"), default="yellow")
    p.add_argument("--target", choices=("red", "blue", "yellow"), default="blue")
    p.add_argument("--output-root", type=Path, default=Path("runs"))
    args = p.parse_args()
    if args.source == args.target:
        raise SystemExit("source and target colors must differ")
    if args.steps < 1:
        raise SystemExit("steps must be positive")
    runner = None
    if args.mode == "policy":
        from deployment.pi05_stack2.model_loader import load_strict_cpu
        from deployment.pi05_stack2.policy_runner import PolicyRunner
        policy, pre, post, model_report = load_strict_cpu(args.checkpoint)
        runner = PolicyRunner(policy, pre, post, "cpu")
    else:
        model_report = {"mode": "random action_space baseline", "policy_connected": False}

    env = make_stackcube_env(args.source, args.target, seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    run = RunLogger(args.output_root)
    frames = []
    rng = np.random.default_rng(args.seed)
    try:
        run.log("sim_start", seed=args.seed, environment="MuJoCoStackCube-v1", repo_commit="b497ffaaad7e82168d5f58e8c179aff6b68edf05",
                source=args.source, target=args.target, policy_connected=runner is not None, model=model_report)
        for step in range(args.steps):
            if runner is None:
                action = env.action_space.sample()
            else:
                transition = sim_observation_to_policy(obs, f"Stack the {args.source} block on top of the {args.target} block.", env.action_space)
                action_deg, latency = runner.infer_one(transition | {"task": f"Stack the {args.source} block on top of the {args.target} block."})
                action = policy_degrees_to_sim_radians(action_deg, env.action_space)
                run.log("policy_action", step=step, latency_s=latency, action_degrees=action_deg.tolist(), action_radians=action.tolist())
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            if step == 0:
                run.log("observation_contract", state_dim=int(obs["state"].shape[0]),
                        state_units="sim radians converted to trained degrees", camera_keys=sorted(k for k in obs if "camera" in k),
                        image_shapes={k: list(v.shape) for k,v in obs.items() if "camera" in k})
            if terminated or truncated:
                obs, info = env.reset(seed=args.seed + step + 1)
        import cv2
        video_path = run.path / "video.mp4"
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        summary = {"environment": "MuJoCoStackCube-v1", "seed": args.seed, "steps": args.steps,
                   "policy_connected": runner is not None, "video": video_path.name,
                   "task_success": bool(info.get("success", False)),
                   "interpretation": "deployment-interface preflight only; no sim2real or physical success claim"}
        (run.path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        env.close()
        run.close()
    print(run.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
