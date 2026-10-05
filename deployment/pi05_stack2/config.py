from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


REPO_ID = "WoWjx/so101-stack2-pi05-full-relative"
REVISION = "main"
MODEL_SHA256 = "f54e8cec0217bc5e2baebc095f06979675209c955093e7d5916fc5587eb098c4"
ACTION_ORDER = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)
STATE_ORDER = ACTION_ORDER
# This ordering is taken from config.json's input_features and is the order
# consumed by PI05Policy._preprocess_images in the pinned LeRobot revision.
CAMERA_ORDER = ("observation.images.wrist", "observation.images.front")


@dataclass(frozen=True)
class DeploymentConfig:
    checkpoint: Path = field(default_factory=lambda: Path(os.environ.get("SO101_PI05_CHECKPOINT", "models/so101-pi05-4228")))
    device: str = field(default_factory=lambda: os.environ.get("SO101_PI05_DEVICE", "cuda"))
    instruction: str = "Stack the yellow block on top of the blue block."
    control_hz: float = 30.0
    action_chunk_size: int = 50
    execute_horizon: int = 10
    state_timeout_s: float = 0.25
    camera_timeout_s: float = 0.50
    policy_timeout_s: float = 2.0
    hardware_enabled: bool = False

    def validate(self) -> None:
        if self.action_chunk_size != 50 or self.execute_horizon != 10:
            raise ValueError("Checkpoint contract fixes chunk_size=50 and n_action_steps=10")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if self.hardware_enabled:
            raise ValueError("This deployment scaffold never enables hardware through config")
