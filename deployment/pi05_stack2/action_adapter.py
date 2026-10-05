from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class JointBounds:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    max_delta_per_step: tuple[float, ...]
    max_velocity_per_s: tuple[float, ...] | None = None
    tracking_error_limit: tuple[float, ...] | None = None

    def validate(self) -> None:
        if not (len(self.lower) == len(self.upper) == len(self.max_delta_per_step) == 6):
            raise ValueError("Safety bounds must cover the six SO101 action dimensions")
        if not np.isfinite([self.lower,self.upper,self.max_delta_per_step]).all():
            raise ValueError("Safety bounds must be finite")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("Each joint lower bound must be below its upper bound")
        if any(d <= 0 for d in self.max_delta_per_step):
            raise ValueError("Each per-step delta limit must be positive")
        if self.max_velocity_per_s is not None and (len(self.max_velocity_per_s) != 6 or
                not np.isfinite(self.max_velocity_per_s).all() or min(self.max_velocity_per_s) <= 0):
            raise ValueError("Velocity limits must be positive finite six-dimensional values")
        if self.tracking_error_limit is not None and (len(self.tracking_error_limit)!=6 or
                not np.isfinite(self.tracking_error_limit).all() or min(self.tracking_error_limit)<=0):
            raise ValueError('Tracking error limits must be positive finite six-dimensional values')


class SafetyRejection(ValueError):
    """Machine-readable rejection; never clips or repairs the requested action."""
    def __init__(self, report):
        self.report = report
        super().__init__("Safety stop: " + ", ".join(v["category"] + ":" + str(v.get("joint", ""))
                                                       for v in report["violations"]))


def rejection_category(exc):
    """Keep sensor staleness, transport failure and schema failure distinct."""
    for category in ('STALE_STATE','STALE_CAMERA','NAN_INF'):
        if category in str(exc): return category
    if 'NaN' in str(exc) or 'Inf' in str(exc): return 'NAN_INF'
    if isinstance(exc, TimeoutError): return 'POLICY_TIMEOUT'
    if isinstance(exc, (ConnectionError, OSError)): return 'NETWORK_OR_DEVICE_ERROR'
    return 'CONTRACT_ERROR'


def append_rejection(report, path="artifacts/safety_rejections.jsonl"):
    """Append JSON-safe provenance, including unavailable intermediate values explicitly."""
    import json
    from pathlib import Path
    def clean(v):
        if isinstance(v, np.ndarray): return clean(v.tolist())
        if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(v): return str(v)
        if isinstance(v, np.generic): return v.item()
        return v
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f: f.write(json.dumps(clean(report), allow_nan=False) + "\n")


def validate_action(action: np.ndarray) -> np.ndarray:
    """Validate a postprocessed absolute six-dimensional SO101 target."""
    a = np.asarray(action, dtype=np.float32)
    if a.shape != (6,):
        raise ValueError(f"Expected a 6D action, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("Action contains NaN or Inf")
    return a


def filter_action(action: np.ndarray, previous: np.ndarray, bounds: JointBounds, *,
                  dt_s: float = 1/30, context: dict | None = None,
                  measured_state: np.ndarray | None = None, startup: bool = False) -> tuple[np.ndarray, list[str]]:
    """Reject out-of-range/oversized commands; never silently clip a model target."""
    bounds.validate()
    from .config import ACTION_ORDER
    report = {"observation_state": None, "normalized_model_output": None,
              "denormalized_model_output": None, "relative_arm_action": None,
              "decoded_absolute_arm_target": None, "absolute_gripper_target": None,
              **(context or {}), "previous_target": np.asarray(previous).tolist(),
              "current_target": np.asarray(action).tolist(), "dt_s": dt_s, "violations": []}
    try:
        target, prev = validate_action(action), validate_action(previous)
    except ValueError as exc:
        category = "NAN_INF" if "NaN" in str(exc) else "CONTRACT_ERROR"
        report["violations"].append({"category":category,"reason":str(exc)})
        raise SafetyRejection(report) from exc
    if not np.isfinite(dt_s) or dt_s <= 0:
        report["violations"].append({"category":"CONTRACT_ERROR","reason":"Invalid control dt"})
        raise SafetyRejection(report)
    lo, hi, delta = map(lambda x: np.asarray(x, dtype=np.float32),
                        (bounds.lower, bounds.upper, bounds.max_delta_per_step))
    change = target - prev
    velocity = change / dt_s
    # Quantile encode/decode can round each float32 endpoint by a few ULPs.
    # This arithmetic error budget does not alter any physical cap or target.
    roundoff=4*np.spacing(np.maximum(1,np.maximum(np.abs(target),np.abs(prev))).astype(np.float32))
    tracking = None if measured_state is None or startup else validate_action(measured_state)-prev
    report.update(decoded_absolute_arm_target=target[:5].tolist(), absolute_gripper_target=float(target[5]),
                  per_joint_delta=change.tolist(), per_joint_velocity=velocity.tolist(),
                  transition_type='STARTUP_TRANSITION' if startup else 'CONTINUOUS_COMMAND',
                  command_delta=change.tolist(),command_velocity=velocity.tolist(),
                  float32_roundoff_tolerance=roundoff.tolist(),
                  tracking_error=None if tracking is None else tracking.tolist(),
                  tracking_error_status='UNRESOLVED_LIMIT' if bounds.tracking_error_limit is None else 'CONFIGURED',
                  configured_bound={"lower":lo.tolist(),"upper":hi.tolist(),"delta":delta.tolist(),
                                    "velocity":bounds.max_velocity_per_s})
    for j in range(6):
        def add(category, value, limit, amount):
            report["violations"].append({"category":category,"dimension":j,"joint":ACTION_ORDER[j],
                                         "value":float(value),"bound":limit,"violation_amount":float(amount)})
        if target[j] < lo[j] or target[j] > hi[j]:
            add("GRIPPER_LIMIT" if j == 5 else "POSITION_LIMIT", target[j], [float(lo[j]),float(hi[j])],
                max(lo[j]-target[j],target[j]-hi[j]))
        if float(abs(change[j]))-float(delta[j]) > float(roundoff[j]):
            add("DELTA_LIMIT",abs(change[j]),float(delta[j]),abs(change[j])-delta[j])
        if bounds.max_velocity_per_s is not None and float(abs(velocity[j]))-bounds.max_velocity_per_s[j] > float(roundoff[j])/dt_s:
            add("VELOCITY_LIMIT",abs(velocity[j]),bounds.max_velocity_per_s[j],abs(velocity[j])-bounds.max_velocity_per_s[j])
        if tracking is not None and bounds.tracking_error_limit is not None and abs(tracking[j])>bounds.tracking_error_limit[j]:
            add('TRACKING_ERROR_LIMIT',abs(tracking[j]),bounds.tracking_error_limit[j],abs(tracking[j])-bounds.tracking_error_limit[j])
    if report["violations"]: raise SafetyRejection(report)
    return target, []
