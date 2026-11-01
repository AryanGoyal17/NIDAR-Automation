"""
cv/utils/run_info.py — What a run records about where and how it ran
=====================================================================

Power plan and AC state, git commit and dirty flag, library versions and UTC
time stamps, for training and evaluation records. ``sim/generator.py`` has its
own copy of the power-plan reader (left unchanged there).
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

LIBRARIES = ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "PyYAML")


def utc_now() -> str:
    """UTC ISO 8601 with milliseconds and ``Z``."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def power_state() -> str:
    """Active power plan and AC state (Windows), e.g. ``power plan Balanced, plugged in (AC)``."""
    if platform.system() != "Windows":
        return "TBD (power plan only read on Windows)"
    try:
        out = subprocess.run(
            ["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=20
        ).stdout.strip()
        plan = out.split("(")[-1].rstrip(")") if "(" in out else out
        status = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Battery).BatteryStatus",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"not read ({exc})"
    # Win32_Battery.BatteryStatus: 1 = discharging, 2 = on AC; empty = no battery
    ac = {"1": "on battery", "2": "plugged in (AC)"}.get(status, f"battery status {status!r}")
    return f"power plan {plan}, {ac}"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_state() -> dict[str, object]:
    """Commit hash and whether the working tree has uncommitted changes."""
    return {
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "dirty": bool(_git("status", "--porcelain")),
    }


def library_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
