from __future__ import annotations

import subprocess
import sys


def test_async_hardware_launcher_requires_explicit_enable_flag():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deployment.pi05_stack2.cli.run_async_robot",
            "--devices-json",
            "does-not-exist.json",
            "--calibration",
            "does-not-exist.json",
            "--safety-profile",
            "does-not-exist.json",
            "--server",
            "http://127.0.0.1:8765",
            "--instruction",
            "test",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Pass --enable-hardware" in result.stderr
    assert "does-not-exist" not in result.stderr
