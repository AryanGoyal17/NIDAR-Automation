"""
data_tools/coco_common.py — Shared helpers for the COCO val2017 tools
=====================================================================

COCO sanity set, not arena. Config, annotation index, zip reader, split
hash, box conversion, size limits at imgsz, hashes and the disk guard used by
``get_coco``, ``coco_split``, ``coco_person_sets``, ``make_person_cutouts``,
``leakage`` and ``coco_sheets``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv.config import load_config

DEFAULT_COCO_CONFIG = Path("configs/coco.yaml")
GROUPS = ("cutout_source", "anchor", "sanity")
ZIP_IMAGE_DIR = "val2017"
GB = 10**9  # decimal gigabyte, as in the owner's disk rules


class DiskSpaceError(RuntimeError):
    """A step would leave less free space than ``disk.min_free_gb``."""


class ConfirmationNeeded(RuntimeError):
    """The estimated size is above ``disk.confirm_above_gb`` and --yes was not given."""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SizeLimits:
    """Smallest kept box at training resolution (from ``arena.yaml labels``)."""

    imgsz: int
    min_height_px: float
    min_width_px: float

    @classmethod
    def from_arena(cls, path: str | Path) -> SizeLimits:
        labels = load_config(path).labels
        return cls(
            imgsz=int(labels.imgsz),
            min_height_px=float(labels.min_box_height_px),
            min_width_px=float(labels.min_box_width_px),
        )

    def scale(self, width: int, height: int) -> float:
        """Letterbox scale from the source image to imgsz."""
        return self.imgsz / max(width, height)


@dataclass
class CocoConfig:
    """Effective COCO config (``configs/coco.yaml``) plus the arena size limits."""

    raw: dict[str, Any]
    limits: SizeLimits
    path: str = str(DEFAULT_COCO_CONFIG)

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_COCO_CONFIG) -> CocoConfig:
        raw = load_config(path).to_dict()
        return cls(raw=raw, limits=SizeLimits.from_arena(raw["arena_config"]), path=str(path))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def hash(self) -> str:
        """sha256 of the effective config and the size limits."""
        payload = {"raw": self.raw, "limits": self.limits.__dict__}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #


@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int
    persons: list[dict[str, Any]] = field(
        default_factory=list
    )  # person annotations, crowd included
    captions: list[str] = field(default_factory=list)

    @property
    def crowd(self) -> list[dict[str, Any]]:
        return [a for a in self.persons if a.get("iscrowd", 0)]

    @property
    def people(self) -> list[dict[str, Any]]:
        """Non-crowd person annotations."""
        return [a for a in self.persons if not a.get("iscrowd", 0)]


def load_index(instances: str | Path, captions: str | Path | None = None) -> dict[int, CocoImage]:
    """Images by id with their person annotations (and captions, if given)."""
    data = json.loads(Path(instances).read_text(encoding="utf-8"))
    person_ids = [c["id"] for c in data["categories"] if c["name"] == "person"]
    if len(person_ids) != 1:
        raise ValueError(f"expected one 'person' category, found {len(person_ids)}")
    person = person_ids[0]
    index = {
        int(im["id"]): CocoImage(
            int(im["id"]), im["file_name"], int(im["width"]), int(im["height"])
        )
        for im in data["images"]
    }
    for ann in data["annotations"]:
        if ann["category_id"] == person:
            index[int(ann["image_id"])].persons.append(ann)
    if captions is not None:
        cap = json.loads(Path(captions).read_text(encoding="utf-8"))
        for ann in cap["annotations"]:
            if int(ann["image_id"]) in index:
                index[int(ann["image_id"])].captions.append(ann["caption"])
    for im in index.values():
        im.persons.sort(key=lambda a: a["id"])
    return index


def is_polygon(ann: dict[str, Any]) -> bool:
    """True if the segmentation is a polygon list (not RLE)."""
    seg = ann.get("segmentation")
    return isinstance(seg, list) and len(seg) > 0


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #


def unit_hash(seed: int, key: str | int) -> float:
    """Uniform value in [0, 1): first 64 bits of sha256("<seed>:<key>") / 2^64."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_group(image_id: int, seed: int, fractions: dict[str, float]) -> str:
    """Group of one image id; depends only on the seed and the id."""
    total = sum(fractions[g] for g in GROUPS)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    u = unit_hash(seed, image_id)
    edge = 0.0
    for g in GROUPS:
        edge += fractions[g]
        if u < edge:
            return g
    return GROUPS[-1]


def load_split(path: str | Path) -> dict[str, set[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {g: set(data["groups"][g]) for g in GROUPS}


# --------------------------------------------------------------------------- #
# Boxes
# --------------------------------------------------------------------------- #


def xywh_to_yolo(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    """COCO [x, y, w, h] (pixels, top-left) to normalised YOLO (cx, cy, w, h), clipped to the image."""
    x, y, w, h = box
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    return (
        (x1 + x2) / 2 / width,
        (y1 + y2) / 2 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def overlap_fraction(box: list[float], other: list[float]) -> float:
    """Intersection area of two xywh boxes / area of ``box``."""
    x1, y1 = max(box[0], other[0]), max(box[1], other[1])
    x2 = min(box[0] + box[2], other[0] + other[2])
    y2 = min(box[1] + box[3], other[1] + other[3])
    area = box[2] * box[3]
    if area <= 0:
        return 1.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / area


# --------------------------------------------------------------------------- #
# Images and hashes
# --------------------------------------------------------------------------- #


class ImageZip:
    """val2017.zip opened once; members read by name, bytes unchanged."""

    def __init__(self, path: str | Path, image_dir: str = ZIP_IMAGE_DIR) -> None:
        self.path = Path(path)
        self.image_dir = image_dir
        self._zip = zipfile.ZipFile(self.path)

    def member(self, file_name: str) -> str:
        return f"{self.image_dir}/{file_name}"

    def read(self, file_name: str) -> bytes:
        return self._zip.read(self.member(file_name))

    def size(self, file_name: str) -> int:
        return self._zip.getinfo(self.member(file_name)).file_size

    def decode(self, file_name: str) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(self.read(file_name), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {self.member(file_name)}")
        return image

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> ImageZip:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def dhash(image: np.ndarray) -> int:
    """64-bit difference hash: 9x8 grey thumbnail, bit = left pixel brighter than right."""
    grey = image if image.ndim == 2 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, :-1] > small[:, 1:]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def percentiles(values: list[float]) -> dict[str, float | None]:
    """min, p10, median, p90, max (None for an empty list)."""
    if not values:
        return {k: None for k in ("n", "min", "p10", "median", "p90", "max")} | {"n": 0}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "min": round(float(a.min()), 2),
        "p10": round(float(np.percentile(a, 10)), 2),
        "median": round(float(np.median(a)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
        "max": round(float(a.max()), 2),
    }


# --------------------------------------------------------------------------- #
# Files, disk, provenance
# --------------------------------------------------------------------------- #


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temporary file in the same folder, then rename over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_bytes(path, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def free_bytes(path: str | Path) -> int:
    """Free bytes on the drive holding ``path`` (or its nearest existing parent)."""
    p = Path(path).resolve()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def check_disk(path: str | Path, needed_bytes: int, cfg: CocoConfig, step: str) -> int:
    """Raise DiskSpaceError if writing ``needed_bytes`` would leave less than min_free_gb."""
    free = free_bytes(path)
    min_free = float(cfg["disk"]["min_free_gb"]) * GB
    if free - needed_bytes < min_free:
        raise DiskSpaceError(
            f"{step}: {needed_bytes / GB:.2f} GB needed, {free / GB:.2f} GB free; "
            f"less than {min_free / GB:.1f} GB would remain"
        )
    return free


def dir_bytes(path: str | Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
