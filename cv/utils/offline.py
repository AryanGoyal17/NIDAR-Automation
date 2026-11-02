"""
cv/utils/offline.py — Run Ultralytics without network calls or usage events
===========================================================================

CLAUDE.md forbids runtime downloads and telemetry. Ultralytics sends anonymous
usage events when ``settings.json`` has ``sync: true`` and it is online, and
its training start-up downloads ``yolo26n.pt`` for the AMP check. This module:

- sets ``YOLO_OFFLINE=1`` before Ultralytics is imported, so its ``ONLINE``
  flag is False and the event collector is disabled (the user's global
  ``settings.json`` is left unchanged);
- replaces the trainer's AMP check with the same FP32-vs-AMP comparison run on
  local weights.

Import and call :func:`ultralytics_offline` before the first Ultralytics import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

AMP_CHECK_WEIGHTS = Path("models/yolov8n.pt")


def ultralytics_offline() -> None:
    """
    Set YOLO_OFFLINE=1. If Ultralytics was already imported (for example by a
    test), its ONLINE flag is fixed, so the event collector is switched off directly.
    """
    os.environ["YOLO_OFFLINE"] = "1"
    if "ultralytics.utils.events" in sys.modules or "ultralytics.utils" in sys.modules:
        from ultralytics.utils.events import events

        events.enabled = False


def events_enabled() -> bool:
    """Whether Ultralytics would send usage events in this process."""
    from ultralytics.utils.events import events

    return bool(events.enabled)


def local_check_amp(weights: str | Path = AMP_CHECK_WEIGHTS):
    """An AMP check with the Ultralytics comparison (FP32 vs AMP boxes on bus.jpg) on local weights."""

    def check(model) -> bool:
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import ASSETS
        from ultralytics.utils.torch_utils import autocast

        device = next(model.parameters()).device
        if device.type in {"cpu", "mps"}:
            return False
        m = YOLO(str(weights))
        batch = [ASSETS / "bus.jpg"] * 8
        imgsz = max(256, int(model.stride.max() * 4))
        a = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data
        with autocast(enabled=True, device=device.type):
            b = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)

    return check


def patch_amp_check(weights: str | Path = AMP_CHECK_WEIGHTS) -> None:
    """Make the Ultralytics trainer use :func:`local_check_amp` (no yolo26n.pt download)."""
    import ultralytics.engine.trainer as trainer

    trainer.check_amp = local_check_amp(weights)
