from __future__ import annotations
import threading
import time
import numpy as np
from .observation_adapter import TimedFrame


class CameraCapture:
    """Stamp a frame only after successful cap.read(), never at consumer read time."""
    def __init__(self, config: dict):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(config["path"])
        if not self.cap.isOpened():
            raise ConnectionError(f"Cannot open configured camera {config['path']}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.get("fourcc", "MJPG")))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.stop = threading.Event(); self.lock = threading.Lock()
        self.frame = None; self.sequence = 0
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        while not self.stop.is_set():
            ok, image = self.cap.read()
            if not ok:
                time.sleep(0.01); continue
            timestamp = time.monotonic()
            rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
            with self.lock:
                self.frame = TimedFrame(rgb, timestamp); self.sequence += 1

    def latest(self):
        with self.lock:
            if self.frame is None:
                raise TimeoutError("Camera has not returned a frame")
            return TimedFrame(self.frame.image.copy(), self.frame.timestamp_monotonic)

    def close(self):
        self.stop.set(); self.thread.join(timeout=1)
        self.cap.release()
