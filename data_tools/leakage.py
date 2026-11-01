"""
data_tools/leakage.py — Check that no protected image leaks into training
=========================================================================

Compares training sources with protected (held-out) sets:

- training sources: every image under each ``--train`` directory, every image
  named in each ``--train-list`` file (an Ultralytics list file, one path per
  line; relative paths are taken relative to the list file), and the COCO
  source photos of each ``--cutouts`` directory (from its ``manifest.json``,
  read from ``val2017.zip``);
- protected sets: each ``--protected`` directory; by default the COCO sanity
  set (COCO sanity set, not arena) and, if they exist, ``data/test_proxy`` and
  ``data/test_real``.

Errors (exit 1):

- the same sha256 on both sides;
- the same COCO image id on both sides (from a ``<12-digit id>.jpg`` file name,
  a ``<id>_<ann>.png`` cutout name, or a cutout manifest);
- with ``--split``: a ``cutout_source`` id among the images of a ``--train`` or
  ``--protected`` directory (cutout_source photos never appear in anchor or
  sanity), or a cutout whose source id is not in ``cutout_source``.

Warnings (listed, exit code unchanged): near matches, 64-bit dHash Hamming
distance <= ``leakage.dhash_max_distance`` (4), between a training source and
a protected image that are not already an exact match (resized or
re-encoded copies, near-duplicate photos).

Usage::

    python -m data_tools.leakage --train data/coco_person/anchor/images --cutouts data/coco_cutouts
    python -m data_tools.leakage --train data/train/images --protected data/test_real --split none
    python -m data_tools.leakage --train-list data/yolo_ds/e2/train.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv.logger import get_logger
from data_tools.coco_common import CocoConfig, ImageZip, dhash, load_split, sha256_bytes

log = get_logger("leakage", log_file=None)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_NAME = re.compile(r"^(\d{12})(?:_(\d+))?\.(?:jpe?g|png)$", re.IGNORECASE)
CUTOUT_MANIFEST = "manifest.json"


@dataclass
class ImageRecord:
    source: str  # file path, or "<cutouts dir>:<cutout file> <- zip:<member>"
    sha256: str
    dhash: int
    coco_id: int | None
    is_cutout: bool = False


def _decode(data: bytes, where: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode {where}")
    return image


def read_list(list_file: str | Path) -> list[Path]:
    """Image paths of an Ultralytics list file (blank lines skipped; relative to the file)."""
    list_file = Path(list_file)
    paths = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            path = Path(line)
            paths.append(path if path.is_absolute() else list_file.parent / path)
    return paths


def scan_dir(directory: str | Path) -> list[ImageRecord]:
    """sha256, dHash and COCO id (from the file name) of every image under a directory."""
    paths = [p for p in sorted(Path(directory).rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    return scan_paths(p for p in paths if p.is_file())


def scan_paths(paths: Iterable[str | Path]) -> list[ImageRecord]:
    """sha256, dHash and COCO id (from the file name) of each given image file."""
    records = []
    for path in map(Path, paths):
        data = path.read_bytes()
        m = COCO_NAME.match(path.name)
        records.append(
            ImageRecord(
                source=str(path),
                sha256=sha256_bytes(data),
                dhash=dhash(_decode(data, str(path))),
                coco_id=int(m.group(1)) if m else None,
                is_cutout=bool(m and m.group(2)),
            )
        )
    return records


def scan_cutout_sources(cutouts_dir: str | Path, images: ImageZip) -> list[ImageRecord]:
    """One record per distinct source photo of a cutout directory, read from the zip."""
    manifest = json.loads((Path(cutouts_dir) / CUTOUT_MANIFEST).read_text(encoding="utf-8"))
    seen: dict[int, ImageRecord] = {}
    for c in manifest["cutouts"]:
        sid = int(c["source_image_id"])
        if sid in seen:
            continue
        data = images.read(c["source_file"])
        seen[sid] = ImageRecord(
            source=f"{cutouts_dir}:{c['file']} <- zip:{images.member(c['source_file'])}",
            sha256=sha256_bytes(data),
            dhash=dhash(_decode(data, c["source_file"])),
            coco_id=sid,
            is_cutout=True,
        )
    return list(seen.values())


def hamming_pairs(
    a: list[int], b: list[int], max_distance: int, chunk: int = 256
) -> list[tuple[int, int, int]]:
    """(i, j, distance) for every pair with Hamming distance <= max_distance."""
    if not a or not b:
        return []
    av = np.asarray(a, dtype=np.uint64)
    bv = np.asarray(b, dtype=np.uint64)
    out = []
    for start in range(0, len(av), chunk):
        x = av[start : start + chunk, None] ^ bv[None, :]
        bits = np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(axis=-1)
        for i, j in zip(*np.nonzero(bits <= max_distance), strict=True):
            out.append((start + int(i), int(j), int(bits[i, j])))
    return out


def check(
    train: list[ImageRecord],
    protected: list[ImageRecord],
    split: dict[str, set[int]] | None,
    max_distance: int,
) -> dict[str, Any]:
    """Errors (exact matches, split violations) and warnings (near matches)."""
    errors: list[str] = []
    warnings: list[str] = []
    by_sha: dict[str, list[ImageRecord]] = {}
    by_id: dict[int, list[ImageRecord]] = {}
    for r in protected:
        by_sha.setdefault(r.sha256, []).append(r)
        if r.coco_id is not None:
            by_id.setdefault(r.coco_id, []).append(r)

    exact_pairs: set[tuple[int, int]] = set()
    pidx = {id(r): k for k, r in enumerate(protected)}
    for i, r in enumerate(train):
        for p in by_sha.get(r.sha256, []):
            errors.append(f"same sha256 {r.sha256[:16]}: {r.source} == {p.source}")
            exact_pairs.add((i, pidx[id(p)]))
        if r.coco_id is not None:
            for p in by_id.get(r.coco_id, []):
                if (i, pidx[id(p)]) not in exact_pairs:
                    errors.append(f"same COCO id {r.coco_id}: {r.source} == {p.source}")
                    exact_pairs.add((i, pidx[id(p)]))

    if split is not None:
        cutout_source = split["cutout_source"]
        for r in train + protected:
            if r.coco_id is None:
                continue
            if r.is_cutout and r.coco_id not in cutout_source:
                errors.append(f"cutout source id {r.coco_id} is not in cutout_source: {r.source}")
            elif not r.is_cutout and r.coco_id in cutout_source:
                errors.append(
                    f"cutout_source image id {r.coco_id} used as a full image: {r.source}"
                )

    for i, j, d in hamming_pairs(
        [r.dhash for r in train], [r.dhash for r in protected], max_distance
    ):
        if (i, j) not in exact_pairs:
            warnings.append(
                f"near match (dHash distance {d}): {train[i].source} ~ {protected[j].source}"
            )
    return {
        "train_images": len(train),
        "protected_images": len(protected),
        "errors": errors,
        "warnings": warnings,
    }


def default_protected(cfg: CocoConfig) -> list[Path]:
    candidates = [
        Path(cfg["paths"]["person_sets"]) / "sanity" / "images",
        Path("data/test_proxy"),
        Path("data/test_real"),
    ]
    return [p for p in candidates if p.exists()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="configs/coco.yaml")
    ap.add_argument(
        "--train", action="append", default=[], help="training image directory (repeatable)"
    )
    ap.add_argument(
        "--train-list", action="append", default=[], help="training list file (repeatable)"
    )
    ap.add_argument(
        "--cutouts", action="append", default=[], help="cutout directory with manifest.json"
    )
    ap.add_argument(
        "--protected", action="append", default=[], help="protected directory (repeatable)"
    )
    ap.add_argument(
        "--split", default=None, help="split.json (default from config; 'none' to skip)"
    )
    ap.add_argument(
        "--zip", default=None, help="val2017.zip for cutout sources (default from config)"
    )
    ap.add_argument("--json", default=None, help="write the result to this file")
    args = ap.parse_args(argv)
    cfg = CocoConfig.from_yaml(args.config)

    protected_dirs = [Path(p) for p in args.protected] or default_protected(cfg)
    if not protected_dirs:
        log.error("no protected directory given or found")
        return 2
    if not args.train and not args.train_list and not args.cutouts:
        log.error("no training source given (--train, --train-list or --cutouts)")
        return 2
    train = [r for d in args.train for r in scan_dir(d)]
    train += scan_paths(p for f in args.train_list for p in read_list(f))
    if args.cutouts:
        with ImageZip(args.zip or cfg["paths"]["images_zip"]) as images:
            train += [r for d in args.cutouts for r in scan_cutout_sources(d, images)]
    protected = [r for d in protected_dirs for r in scan_dir(d)]
    split_path = args.split or cfg["paths"]["split"]
    split = None if split_path == "none" else load_split(split_path)
    result = check(train, protected, split, int(cfg["leakage"]["dhash_max_distance"]))
    result["train_dirs"] = args.train
    result["train_lists"] = args.train_list
    result["cutout_dirs"] = args.cutouts
    result["protected_dirs"] = [str(p) for p in protected_dirs]
    result["split"] = split_path

    print(
        f"training sources: {result['train_images']} images from {args.train + args.train_list + args.cutouts}"
    )
    print(f"protected:        {result['protected_images']} images from {result['protected_dirs']}")
    for w in result["warnings"]:
        print(f"WARNING {w}")
    for e in result["errors"]:
        print(f"ERROR   {e}")
    print(f"{len(result['errors'])} errors, {len(result['warnings'])} near-match warnings")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
