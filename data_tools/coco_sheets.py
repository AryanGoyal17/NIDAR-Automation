"""
data_tools/coco_sheets.py — Contact sheets for COCO cutouts and YOLO person sets
================================================================================

COCO sanity set, not arena. Two sheets, both drawn from the written files:

- ``--cutouts DIR``: seeded cutouts from ``DIR/manifest.json`` on a
  checkerboard (shows the alpha), with a thin frame at the PNG border (a tight
  crop touches it on all four sides) and height / aspect captions;
- ``--yolo DIR``: seeded images of a YOLO set (``DIR/images``, ``DIR/labels``)
  with the boxes read from the label files (green), negatives marked.

Usage::

    python -m data_tools.coco_sheets --cutouts data/coco_cutouts --out outputs/scratch/coco/cutouts.png
    python -m data_tools.coco_sheets --yolo data/coco_person/anchor --out outputs/scratch/coco/anchor.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from data_tools.coco_common import unit_hash

TILE_W, TILE_H, CAPTION_H, COLS = 300, 260, 22, 4
TITLE_H = 30


def _fit(image: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float, int, int]:
    """Letterbox into w x h; returns (canvas, scale, x offset, y offset)."""
    s = min(w / image.shape[1], h / image.shape[0])
    nw, nh = max(1, round(image.shape[1] * s)), max(1, round(image.shape[0] * s))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, image.shape[2]), 40, np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas, s, ox, oy


def _checker(w: int, h: int, cell: int = 12) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    v = np.where(((yy // cell + xx // cell) % 2) == 0, 200, 150).astype(np.uint8)
    return np.dstack([v, v, v])


def _tile(body: np.ndarray, caption: str) -> np.ndarray:
    tile = np.full((TILE_H + CAPTION_H, TILE_W, 3), 25, np.uint8)
    tile[:TILE_H] = body
    cv2.putText(tile, caption, (6, TILE_H + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    return tile


def _sheet(tiles: list[np.ndarray], title: str) -> np.ndarray:
    rows = (len(tiles) + COLS - 1) // COLS
    th = TILE_H + CAPTION_H
    sheet = np.full((TITLE_H + rows * (th + 4), COLS * (TILE_W + 4), 3), 15, np.uint8)
    cv2.putText(sheet, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for k, t in enumerate(tiles):
        r, c = divmod(k, COLS)
        y, x = TITLE_H + r * (th + 4), c * (TILE_W + 4)
        sheet[y : y + th, x : x + TILE_W] = t
    return sheet


def cutout_sheet(cutouts_dir: Path, n: int, seed: int) -> np.ndarray:
    manifest = json.loads((cutouts_dir / "manifest.json").read_text(encoding="utf-8"))
    chosen = sorted(manifest["cutouts"], key=lambda c: unit_hash(seed, c["file"]))[:n]
    tiles = []
    for c in chosen:
        bgra = cv2.imread(str(cutouts_dir / "images" / c["file"]), cv2.IMREAD_UNCHANGED)
        fitted, s, ox, oy = _fit(bgra, TILE_W - 8, TILE_H - 8)
        alpha = fitted[:, :, 3:4].astype(np.float32) / 255.0
        bg = _checker(fitted.shape[1], fitted.shape[0])
        comp = (fitted[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
        body = np.full((TILE_H, TILE_W, 3), 40, np.uint8)
        body[4:-4, 4:-4] = comp
        x0, y0 = 4 + ox, 4 + oy
        x1 = x0 + round(bgra.shape[1] * s) - 1
        y1 = y0 + round(bgra.shape[0] * s) - 1
        cv2.rectangle(body, (x0, y0), (x1, y1), (0, 200, 255), 1)
        tiles.append(
            _tile(body, f"id {c['source_image_id']} h={c['height']} w/h={c['aspect_w_over_h']:.2f}")
        )
    title = f"COCO person cutouts ({len(manifest['cutouts'])} total, {n} seeded) - COCO sanity set, not arena"
    return _sheet(tiles, title)


def yolo_sheet(set_dir: Path, n: int, seed: int, negatives: int = 2) -> np.ndarray:
    labels = {p.stem: p for p in (set_dir / "labels").glob("*.txt")}
    images = sorted(p for p in (set_dir / "images").iterdir() if p.stem in labels)
    pos = [p for p in images if labels[p.stem].read_text().strip()]
    neg = [p for p in images if not labels[p.stem].read_text().strip()]

    def order(paths: list[Path]) -> list[Path]:
        return sorted(paths, key=lambda p: unit_hash(seed, p.name))

    chosen = order(pos)[: n - min(negatives, len(neg))] + order(neg)[:negatives]
    tiles = []
    for p in chosen[:n]:
        image = cv2.imread(str(p))
        h, w = image.shape[:2]
        fitted, s, ox, oy = _fit(image, TILE_W, TILE_H)
        rows = [line.split() for line in labels[p.stem].read_text().splitlines() if line.strip()]
        for _, cx, cy, bw, bh in rows:
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            p1 = (round(ox + (cx - bw / 2) * s), round(oy + (cy - bh / 2) * s))
            p2 = (round(ox + (cx + bw / 2) * s), round(oy + (cy + bh / 2) * s))
            cv2.rectangle(fitted, p1, p2, (0, 255, 0), 1)
        kind = f"{len(rows)} persons" if rows else "NEGATIVE"
        tiles.append(_tile(fitted, f"{p.stem.lstrip('0')} {w}x{h} {kind}"))
    title = f"COCO {set_dir.name} set, label-file boxes ({n} seeded) - COCO sanity set, not arena"
    return _sheet(tiles, title)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cutouts", type=Path)
    src.add_argument("--yolo", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tiles", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    sheet = (
        cutout_sheet(args.cutouts, args.tiles, args.seed)
        if args.cutouts
        else yolo_sheet(args.yolo, args.tiles, args.seed)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
