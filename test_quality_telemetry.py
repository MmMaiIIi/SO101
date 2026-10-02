"""Hardware-free regression checks for timestamp pairing and append-only sidecars."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from quality_telemetry import camera_snapshot, append_json


class TelemetryTests(unittest.TestCase):
    def test_pair_old_frame_across_capture_update(self):
        class Camera:
            last_read_frame_metadata = (1, 10_000_000_000)
            frame_seq = 2
            frame_update_monotonic_ns = 11_000_000_000
        camera = Camera()
        sample = camera_snapshot({'wrist': camera}, 11_000_000_000)
        self.assertEqual(sample['wrist_update_monotonic_ns'], 10_000_000_000)
        self.assertEqual(sample['wrist_age_ms'], 1000.)

    def test_unknown_is_not_read_timestamp(self):
        self.assertIsNone(camera_snapshot({'front': object()}, 10_000_000_000)['front_age_ms'])

    def test_append_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            append_json(root, 'events.jsonl', {'event': 'start', 'attempt': 1})
            append_json(root, 'events.jsonl', {'event': 'discard', 'attempt': 1})
            self.assertEqual([json.loads(x)['event'] for x in (root/'events.jsonl').read_text().splitlines()], ['start','discard'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
