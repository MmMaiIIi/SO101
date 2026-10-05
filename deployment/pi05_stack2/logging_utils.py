from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid


class RunLogger:
    def __init__(self, root: Path = Path("runs")):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.path = root / f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self._lock = threading.Lock()
        self._events = (self.path / "events.jsonl").open("a", encoding="utf-8")

    def log(self, event: str, **fields) -> None:
        row = {"event": event, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **fields}
        with self._lock:
            self._events.write(json.dumps(row, allow_nan=False) + "\n")
            self._events.flush()

    def close(self) -> None:
        self._events.close()
