from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .action_adapter import validate_action


class PolicyRunner:
    """Uses the checkpoint's saved LeRobot processors and native 10-action queue."""
    def __init__(self, policy: Any, preprocessor: Any, postprocessor: Any, device: str):
        self.policy = policy.eval()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = torch.device(device)

    @torch.inference_mode()
    def infer_one(self, transition: dict) -> tuple[np.ndarray, float]:
        transition = dict(transition)
        task = str(transition.pop("task", ""))
        t0 = time.perf_counter()
        from lerobot.policies import prepare_observation_for_inference
        prepared_raw = prepare_observation_for_inference(transition, self.device, task, "so101")
        prepared = self.preprocessor(prepared_raw)
        action = self.policy.select_action(prepared)
        action = self.postprocessor(action)
        if isinstance(action, torch.Tensor):
            action = action.detach().float().cpu().numpy()
        action = np.asarray(action).reshape(-1)
        return validate_action(action), time.perf_counter() - t0

    def reset(self) -> None:
        self.policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            reset = getattr(processor, "reset", None)
            if reset:
                reset()

    @torch.inference_mode()
    def infer_chunk(self, transition: dict, *, full_horizon: bool = False) -> tuple[np.ndarray, float]:
        """Decode the ten native queued actions against one unchanged observation anchor."""
        self.reset()
        start = time.perf_counter()
        from lerobot.policies import prepare_observation_for_inference
        from lerobot.lerobot_types import TransitionKey
        raw = dict(transition); task = str(raw.pop("task", ""))
        anchor_state = np.asarray(raw["observation.state"]).copy().tolist()
        def stamp():
            if self.device.type == "cuda": torch.cuda.synchronize(self.device)
            return time.perf_counter()
        prepared = self.preprocessor(prepare_observation_for_inference(raw,self.device,task,"so101"))
        t3 = stamp()
        # Same observation and fixed anchor: preprocessing once is equivalent to ten identical calls.
        if full_horizon:
            # Pinned native async PolicyServer._get_action_chunk uses predict_action_chunk, not select_action.
            full=self.policy.predict_action_chunk(prepared)
            normalized=[full[:,i,:] for i in range(full.shape[1])]
        else:
            normalized = [self.policy.select_action(prepared) for _ in range(self.policy.config.n_action_steps)]
        t4 = stamp()
        denormalized = []
        def capture(index, value):
            if index == 0: denormalized.append(value[TransitionKey.ACTION].detach().clone())
        self.postprocessor.after_step_hooks.append(capture)
        try: actions = [self.postprocessor(a).detach().float().cpu().numpy().reshape(-1) for a in normalized]
        finally: self.postprocessor.after_step_hooks.remove(capture)
        t5 = stamp()
        self.last_trace = {"normalized_model_output":[a.detach().float().cpu().numpy().reshape(-1).tolist() for a in normalized],
                           "denormalized_model_output":[a.float().cpu().numpy().reshape(-1).tolist() for a in denormalized],
                           "observation_state":anchor_state,
                           "preprocessing_ms":(t3-start)*1000,"model_ms":(t4-t3)*1000,
                           "postprocessing_ms":(t5-t4)*1000,
                           "T3_processor_complete":t3,"T4_model_complete":t4,"T5_postprocess_complete":t5}
        return np.stack([validate_action(a) for a in actions]), time.perf_counter() - start
