"""Bounded inward recovery for tiny saved-calibration edge drift.

The helper never writes an out-of-envelope goal and restores the motor's
original P gain after recovery. It is only used by the explicit live runner.
"""
from __future__ import annotations

import time

import numpy as np


def recover_and_hold(robot, profile: dict, event) -> np.ndarray:
    state, _ = robot.read_state()
    lower = np.asarray(profile["lower"], dtype=np.float32)
    upper = np.asarray(profile["upper"], dtype=np.float32)
    quantization = 360.0 / 4095.0
    outside = np.maximum(np.maximum(lower - state, state - upper), 0)
    if not np.isfinite(state).all():
        raise ValueError("NAN_INF startup state")
    indices = np.flatnonzero(outside)
    event("startup_state", state=state.tolist(), outside=outside.tolist())
    if any(j not in (1, 2) or outside[j] > 4 * quantization + 1e-5 for j in indices):
        raise ValueError("STARTUP_RANGE exceeds supported small shoulder/elbow drift")

    target = state.copy()
    for j in indices:
        target[j] = upper[j] - quantization if state[j] > upper[j] else lower[j] + quantization
    if not np.all((target >= lower) & (target <= upper)):
        raise ValueError("COMMAND_TARGET_OUTSIDE startup recovery target")

    robot.send_target(target)
    robot.bus.enable_torque(num_retry=1)
    robot.require_write_ready()
    for j in indices:
        name = list(robot.bus.motors)[j]
        direction = -1 if state[j] > upper[j] else 1
        edge = upper[j] if direction < 0 else lower[j]
        inward_endpoint = edge + direction * 3.0
        original_p = robot.bus.read("P_Coefficient", name, normalize=False)
        event("STARTUP_INWARD_RECOVERY", joint=name, state=state.tolist(),
              inward_target=float(inward_endpoint), original_P=original_p)
        try:
            robot.bus.write("Lock", name, 0)
            robot.bus.write("P_Coefficient", name, 32)
            robot.bus.write("Lock", name, 1)
            started = time.monotonic()
            stable_since = None
            tick = 0
            first_target = float(target[j])
            initial = float(state[j])
            completed = False
            while time.monotonic() - started < 6:
                current, _ = robot.read_state()
                present = float(current[j])
                inward_goal = first_target + direction * min(abs(inward_endpoint - first_target),
                                                              (time.monotonic() - started) * 0.8)
                if not lower[j] < inward_goal < upper[j]:
                    raise ValueError("COMMAND_TARGET_OUTSIDE recovery")
                if direction * (present - initial) < -quantization:
                    raise ValueError("Recovery moved outward")
                if abs(present - target[j]) > profile["max_delta_per_step"][j]:
                    raise ValueError("TRACKING_ERROR recovery")
                status = robot.bus.read("Status", name, normalize=False)
                temperature = robot.bus.read("Present_Temperature", name, normalize=False)
                current_value = robot.bus.read("Present_Current", name, normalize=False)
                protection = robot.bus.read("Protection_Current", name, normalize=False)
                if status or temperature >= 60 or abs(current_value) >= protection:
                    raise ValueError("Motor protection during recovery")
                if robot.bus.read("Torque_Enable", name, normalize=False) != 1:
                    raise ValueError("Torque lost during recovery")
                if direction * (present - edge) >= 0.5:
                    stable_since = stable_since or time.monotonic()
                    if time.monotonic() - stable_since >= 0.6:
                        completed = True
                        break
                else:
                    stable_since = None
                    robot.bus.write("Goal_Position", name, float(inward_goal))
                    target[j] = inward_goal
                event("recovery_tick", joint=name, present=present, goal=float(target[j]),
                      current=current_value, temperature=temperature, status=status)
                tick += 1
                time.sleep(max(0, started + tick / 30 - time.monotonic()))
            if not completed:
                raise TimeoutError("Bounded inward recovery failed")
        finally:
            robot.bus.write("Lock", name, 0)
            robot.bus.write("P_Coefficient", name, original_p)
            robot.bus.write("Lock", name, 1)
        event("recovery_handoff", joint=name, state=current.tolist(), held_target=target.tolist(),
              P_restored=original_p, torque=1)
    return target
