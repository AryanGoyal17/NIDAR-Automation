"""
tests/test_cv/test_offline.py — Ultralytics runs offline, without usage events

Runs in a fresh interpreter so the import order is the one the CLIs use.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("ultralytics")


@pytest.mark.parametrize("module", ["cv.eval.evaluate"])
def test_cli_modules_disable_events(module: str) -> None:
    code = (
        f"import {module}, os\n"
        "from cv.utils.offline import events_enabled\n"
        "from ultralytics.utils import ONLINE\n"
        "assert os.environ['YOLO_OFFLINE'] == '1'\n"
        "assert not ONLINE and not events_enabled()\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


def test_amp_check_patched() -> None:
    code = (
        "from cv.utils.offline import ultralytics_offline, patch_amp_check\n"
        "ultralytics_offline()\n"
        "import ultralytics.engine.trainer as t\n"
        "patch_amp_check()\n"
        "assert t.check_amp.__qualname__.startswith('local_check_amp')\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
