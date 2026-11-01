"""
data_tools/draw_boxes.py — Contact sheets from a generated run
==============================================================

Draws the boxes of the WRITTEN label files on the WRITTEN images (never
in-memory results), so the sheet shows exactly what training will read.
meta.jsonl only supplies the caption (mode, outcome, hard labels, low light);
without it the caption is the frame id.

Tiles are chosen with a seeded shuffle, taking modes in turn so each mode is
represented.

Usage::

    python -m data_tools.draw_boxes --run data/sim_raw/run1 --out sheet.png --tiles 24 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv.logger import get_logger

log = get_logger("draw_boxes", log_file=None)

BOX_COLOR = (0, 255, 0)  # BGR green
CAPTION_BG = (32, 32, 32)
CAPTION_FG = (255, 255, 255)


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """(class, cx, cy, w, h) per line of a YOLO label file."""
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 5:
            out.append((int(fields[0]), *(float(v) for v in fields[1:])))
    return out


def draw_tile(
    image_path: Path, label_path: Path, caption: str, tile_width: int = 320
) -> np.ndarray:
    """One image with its label-file boxes and a caption bar, scaled to ``tile_width``."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {image_path}")
    h, w = img.shape[:2]
    for _, cx, cy, bw, bh in read_boxes(label_path):
        x1, y1 = round((cx - bw / 2) * w), round((cy - bh / 2) * h)
        x2, y2 = round((cx + bw / 2) * w) - 1, round((cy + bh / 2) * h) - 1
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 1)
    scale = tile_width / w
    tile = cv2.resize(img, (tile_width, round(h * scale)), interpolation=cv2.INTER_AREA)
    bar = np.full((18, tile_width, 3), CAPTION_BG, dtype=np.uint8)
    cv2.putText(bar, caption, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CAPTION_FG, 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def caption_for(fid: str, meta: dict[str, Any] | None) -> str:
    if meta is None:
        return fid
    hard = sum(lb["hard"] for lb in meta["labels"])
    parts = [fid, meta["mode"], meta["outcome"], f"{len(meta['labels'])} box"]
    if hard:
        parts.append(f"{hard} hard")
    if meta["low_light"]:
        parts.append("low light")
    if meta["retries"]:
        parts.append(f"r{meta['retries']}")
    return " ".join(parts)


def choose_frames(run: Path, tiles: int, seed: int) -> tuple[list[str], dict[str, dict]]:
    """Frame ids for the sheet (modes in turn, seeded) and the meta lines by id."""
    metas: dict[str, dict] = {}
    meta_path = run / "meta.jsonl"
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            m = json.loads(line)
            metas[m["frame_id"]] = m
    ids = sorted(p.stem for p in (run / "images").glob("*.jpg"))
    groups: dict[str, list[str]] = defaultdict(list)
    for fid in ids:
        groups[metas[fid]["mode"] if fid in metas else "unknown"].append(fid)
    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    chosen: list[str] = []
    order = sorted(groups)
    while len(chosen) < tiles and any(groups.values()):
        for key in order:
            if groups[key] and len(chosen) < tiles:
                chosen.append(groups[key].pop())
    return chosen, metas


def draw_sheet(
    run: str | Path,
    out: str | Path,
    tiles: int = 24,
    seed: int = 0,
    cols: int = 6,
    tile_width: int = 320,
) -> list[str]:
    """Write a contact sheet of ``tiles`` frames; returns the chosen frame ids."""
    run = Path(run)
    chosen, metas = choose_frames(run, tiles, seed)
    if not chosen:
        raise ValueError(f"no images in {run / 'images'}")
    cells = [
        draw_tile(
            run / "images" / f"{fid}.jpg",
            run / "labels" / f"{fid}.txt",
            caption_for(fid, metas.get(fid)),
            tile_width,
        )
        for fid in chosen
    ]
    th, tw = cells[0].shape[:2]
    rows = (len(cells) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        sheet[r * th : r * th + cell.shape[0], c * tw : c * tw + cell.shape[1]] = cell
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), sheet):
        raise OSError(f"cannot write {out}")
    log.info(f"contact sheet: {out} ({len(chosen)} tiles from {run})")
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contact sheet from a generated run")
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--out", required=True, help="Output PNG")
    parser.add_argument("--tiles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=320)
    args = parser.parse_args(argv)
    draw_sheet(args.run, args.out, args.tiles, args.seed, args.cols, args.tile_width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
