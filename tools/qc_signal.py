"""Deterministic signal metrics. Heuristics describe anomalies, not motor safety."""
import re
import numpy as np


TASK_RE = re.compile(r"^Stack the (red|blue|yellow) block on top of the (red|blue|yellow) block\.$")


def parse_task(text):
    """Parse the exact canonical task; reject unknown colors and source=target."""
    match = TASK_RE.fullmatch(str(text))
    return match.groups() if match and match[1] != match[2] else None


def finite_values(values):
    """Count nonfinite scalar entries, preserving row/joint locations."""
    a = np.asarray(values, dtype=float)
    locations = np.argwhere(~np.isfinite(a))
    return {"ok": not len(locations), "nonfinite_count": len(locations), "locations": locations[:30].tolist()}


def timestamp_check(values, fps):
    """Check finite, strictly increasing episode timestamps and nominal frame grid."""
    a = np.asarray(values, dtype=float)
    finite = bool(np.isfinite(a).all())
    monotonic = finite and bool(np.all(np.diff(a) > 0))
    error = float(np.max(np.abs(a - np.arange(len(a)) / fps))) if len(a) and finite else None
    return {"finite": finite, "strictly_monotonic": monotonic, "grid_max_error_s": error,
            "grid_ok": error is not None and error < 1e-4,
            "first_s": float(a[0]) if len(a) else None, "last_s": float(a[-1]) if len(a) else None}


def longest_run(mask):
    """Longest consecutive True edge run; zero when there are no matching edges."""
    longest = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        longest = max(current, longest)
    return longest


def lag_estimate(action, state, fps, max_lag=10):
    """Compare action[t] to state[t+lag]; best lag minimizes RMSE, also report max-corr lag."""
    a, s = np.asarray(action, float), np.asarray(state, float)
    rows = []
    for lag in range(min(max_lag + 1, len(a) - 2)):
        x, y = (a[:-lag], s[lag:]) if lag else (a, s)
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        if len(x) < 3:
            continue
        variable = np.std(x) > 1e-8 and np.std(y) > 1e-8
        rows.append({"lag": lag, "corr": float(np.corrcoef(x, y)[0, 1]) if variable else None,
                     "rmse": float(np.sqrt(np.mean((x - y) ** 2))), "bias": float(np.mean(y - x))})
    if not rows:
        return {"identifiable": False, "curve": []}
    best = min(rows, key=lambda row: (row["rmse"], row["lag"]))
    correlated = [row for row in rows if row["corr"] is not None]
    best_corr = max(correlated, key=lambda row: (row["corr"], -row["lag"])) if correlated else None
    return {"best_lag_frames": best["lag"], "best_lag_ms": best["lag"] / fps * 1000,
            "corr": best["corr"], "RMSE": best["rmse"], "bias": best["bias"],
            "best_corr_lag_frames": best_corr["lag"] if best_corr else None,
            "identifiable": bool(correlated), "curve": rows}


def normalized_motion(action, state):
    """Joint motion normalized by each joint's observed IQR, combining state and action."""
    a, s = np.asarray(action, float), np.asarray(state, float)
    values = np.concatenate([a, s])
    scales = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    scales = np.maximum(scales, np.nanstd(values, axis=0) * 0.1)
    scales = np.where(scales > 1e-6, scales, 1)
    delta = np.maximum(np.abs(np.diff(a, axis=0)), np.abs(np.diff(s, axis=0))) / scales
    return np.r_[0., np.sqrt(np.mean(delta ** 2, axis=1))], scales


def joint_stats(action, state, names, fps):
    """Descriptive joint ranges/deltas, adaptive isolated-jump candidates, and lag curves."""
    out = {}
    for j, name in enumerate(names):
        a, s = action[:, j], state[:, j]
        if not np.isfinite(a).all() or not np.isfinite(s).all():
            out[name] = {"nonfinite": True}
            continue
        da, ds = np.abs(np.diff(a)), np.abs(np.diff(s))
        row = {"lag": lag_estimate(a, s, fps)}
        for key, values, delta in (("action", a, da), ("state", s, ds)):
            median = float(np.median(delta)) if len(delta) else 0
            mad = float(np.median(np.abs(delta - median))) if len(delta) else 0
            p95 = float(np.percentile(delta, 95)) if len(delta) else 0
            span = float(np.ptp(values))
            threshold = max(median + 12 * 1.4826 * mad, 4 * p95, 0.1 * span, 1e-6)
            candidates = np.flatnonzero(delta > threshold) + 1
            row[key] = {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
                        "std": float(values.std()), "max_abs_delta": float(delta.max()) if len(delta) else 0,
                        "p95_abs_delta": p95, "jump_threshold": threshold,
                        "jump_candidate_frames": candidates.tolist(),
                        "longest_exact_stationary_s": longest_run(delta == 0) / fps}
        out[name] = row
    return out


def boundary_stats(motion, fps):
    """Estimate idle padding and last-2s motion; cannot infer task completion or RESET semantics."""
    positive = motion[motion > 1e-8]
    threshold = float(np.percentile(positive, 20) * 0.5) if len(positive) else 1e-8
    idle = motion <= threshold
    first = next((i for i, v in enumerate(idle) if not v), len(idle))
    last = next((i for i, v in enumerate(idle[::-1]) if not v), len(idle))
    tail = motion[-int(2 * fps):]
    moving_threshold = float(np.median(positive)) if len(positive) else 1e-8
    return {"start_idle_s": first / fps, "end_idle_s": last / fps,
            "idle_motion_threshold": threshold, "tail_moving_fraction": float(np.mean(tail > moving_threshold)),
            "tail_motion_mean": float(np.mean(tail)), "tail_motion_max": float(np.max(tail)),
            "last_frame_moving": bool(motion[-1] > moving_threshold),
            "reset_leak": "not inferable from signals alone; inspect contact sheet/video",
            "post_save_motion": "unobservable: RESET frames are not in the dataset"}


def length_outliers(lengths):
    """Median/MAD batch length outliers; robust z>3.5, with IQR fallback for zero MAD."""
    a = np.asarray(lengths, float)
    median = float(np.median(a))
    mad = float(np.median(np.abs(a - median)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float((np.percentile(a, 75) - np.percentile(a, 25)) / 1.349)
    flags = []
    for x in a:
        z = (float(x) - median) / scale if scale > 1e-9 else 0
        flags.append({"robust_z": z, "flag": "too_long" if z > 3.5 else "too_short" if z < -3.5 else None})
    return {"median": median, "MAD": mad, "scale": scale, "episodes": flags}
