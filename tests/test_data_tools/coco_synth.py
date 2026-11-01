"""
tests/test_data_tools/coco_synth.py — Synthetic COCO fixtures (no real download)
================================================================================

Hand-made person annotations, a tiny val2017-style zip of generated JPEGs and a
stub person detector, shared by the COCO tool tests through ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from data_tools.coco_common import SizeLimits

REPO = Path(__file__).resolve().parents[2]
LIMITS = SizeLimits(imgsz=416, min_height_px=10.0, min_width_px=6.0)
NEG_FILL_PERSON = 230  # the stub detector "sees a person" in images of this grey level


def rect_poly(x: float, y: float, w: float, h: float) -> list[list[float]]:
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def person(ann_id: int, bbox: list[float], crowd: int = 0, rle: bool = False) -> dict[str, Any]:
    seg: Any = {"counts": [0, 10], "size": [10, 10]} if rle else rect_poly(*bbox)
    return {"id": ann_id, "category_id": 1, "bbox": bbox, "iscrowd": crowd, "segmentation": seg}


# id -> (width, height, persons, captions, grey fill)
SPEC: dict[int, tuple[int, int, list[dict[str, Any]], list[str], int]] = {
    1: (640, 480, [person(11, [100, 100, 50, 200])], ["a man on a street"], 90),
    2: (
        640,
        480,
        [person(21, [100, 100, 50, 200]), person(22, [10, 10, 5, 5])],
        ["two people"],
        90,
    ),
    3: (
        640,
        480,
        [person(31, [100, 100, 50, 200]), person(32, [300, 50, 200, 200], crowd=1)],
        ["crowd"],
        90,
    ),
    4: (640, 480, [person(41, [200, 100, 60, 180], rle=True)], ["a skier"], 90),
    5: (640, 480, [], ["a bowl of fruit"], 60),
    6: (640, 480, [], ["a dog and a man on a couch"], 60),  # caption filter
    7: (640, 480, [], ["an empty kitchen"], NEG_FILL_PERSON),  # detector filter
    8: (640, 480, [], ["a red bus"], 70),
    9: (640, 480, [], ["a train station"], 80),
}


def jpeg(width: int, height: int, fill: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    image = np.clip(fill + rng.integers(-5, 6, (height, width, 3)), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def stub_detector(images: list[np.ndarray]) -> list[float]:
    """Person confidence 0.9 for the bright test image, 0.1 otherwise; never used for labels."""
    return [0.9 if abs(float(im.mean()) - NEG_FILL_PERSON) < 10 else 0.1 for im in images]
