from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import threading
import numpy as np
from .config import ACTION_ORDER


class SO101Hardware:
    """Read-only connection; only send_target can write, behind an explicit guard.

    Avoids SOFollower.connect/configure: no PID, ID, calibration or torque writes.
    """
    def __init__(self, port: str, calibration_path: Path, expected_calibration_hash: str, *, motor_output: bool = False):
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        raw = calibration_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_calibration_hash:
            raise ValueError("Safety-profile calibration hash mismatch")
        calibration = {name: MotorCalibration(**value) for name,value in json.loads(raw).items()}
        motors = {}
        for i, key in enumerate(ACTION_ORDER):
            name = key.removesuffix(".pos")
            if calibration[name].id != i + 1:
                raise ValueError("Motor ID differs from the verified six-joint contract")
            motors[name] = Motor(i+1, "sts3215", MotorNormMode.RANGE_0_100 if i == 5 else MotorNormMode.DEGREES)
        self.bus = FeetechMotorsBus(port=port, motors=motors, calibration=calibration)
        self.motor_output = motor_output
        self.owner_thread = threading.get_ident()
        self.read_attempts=0; self.read_calls=0
        txrx=self.bus.sync_reader.txRxPacket
        def counted_read():
            self._assert_owner(); self.read_attempts+=1
            return txrx()
        self.bus.sync_reader.txRxPacket=counted_read
        for method in ('sync_read', 'sync_write', 'read', 'write'):
            original = getattr(self.bus, method)
            def owned(*args, _method=original, **kwargs):
                self._assert_owner()
                return _method(*args, **kwargs)
            setattr(self.bus, method, owned)

    def _assert_owner(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('Feetech bus accessed outside control-owner thread')

    def connect(self):
        self._assert_owner()
        self.bus.connect()  # read-only handshake; do not call robot.configure/calibrate
        if not self.bus.is_calibrated:
            self.close()
            raise ValueError("Motor calibration differs from saved calibration; no changes were made")

    def read_state(self):
        self._assert_owner()
        self.read_calls+=1
        values = self.bus.sync_read("Present_Position", num_retry=2)
        state = np.array([values[k.removesuffix(".pos")] for k in ACTION_ORDER], dtype=np.float32)
        return state, time.monotonic()

    def require_write_ready(self):
        from lerobot.motors.feetech import OperatingMode
        if not self.motor_output:
            raise RuntimeError("Motor output is disabled")
        for name in self.bus.motors:
            if self.bus.read("Operating_Mode", name, normalize=False) != OperatingMode.POSITION.value:
                raise ValueError("Follower is not already in position mode; no configuration was changed")
            if not self.bus.read("Torque_Enable", name, normalize=False):
                raise ValueError("Follower torque is not already enabled; no torque setting was changed")

    def send_target(self, action):
        if not self.motor_output:
            raise RuntimeError("NO MOTOR COMMANDS: hardware guard disabled")
        from .action_adapter import validate_action
        action = validate_action(action)
        self.bus.sync_write("Goal_Position", {k.removesuffix(".pos"):float(v) for k,v in zip(ACTION_ORDER,action)})

    def close(self):
        self._assert_owner()
        if self.bus.is_connected:
            self.bus.disconnect(disable_torque=False)
