"""
data_tools/make_person_cutouts.py — RGBA person cutouts from COCO val2017
=========================================================================

COCO sanity set, not arena. Cuts people out of the ``cutout_source`` group only
(never anchor or sanity images) for a later compositor. A non-crowd person
annotation becomes a cutout if (``configs/coco.yaml cutouts``):

- its segmentation is a polygon (persons stored as RLE are skipped and counted;
  polygons are rasterised with ``cv2.fillPoly``, pycocotools is not needed);
- its COCO box does not touch the image border (``border_margin_px``);
- its COCO box is at least ``min_source_height_px`` tall in the source photo;
- no other person box (crowd regions included) covers more than
  ``max_overlap_fraction`` of its box;
- its mask covers at least ``min_mask_fraction`` of its box.

Each cutout is a PNG (BGRA, alpha 255 inside the mask and 0 outside, colour
zeroed outside the mask), cropped tight to the mask. Alpha is binary: edge
feathering is the compositor's job. ``data/coco_cutouts/manifest.json`` lists
each cutout with its source image id, COCO box, cutout box, size and aspect,
and the counts per rejection reason.

The size is estimated before writing (PNG size measured on a seeded sample,
scaled by raw RGBA bytes). Above ``disk.confirm_above_gb`` the script stops
(exit 5) until re-run with ``--yes``; it stops (exit 4) if less than
``disk.min_free_gb`` would remain free.

Usage::

    python -m data_tools.make_person_cutouts
    python -m data_tools.make_person_cutouts --plan
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv.logger import get_logger
from data_tools.coco_common import (
    GB,
    CocoConfig,
    CocoImage,
    ConfirmationNeeded,
    DiskSpaceError,
    ImageZip,
    atomic_write_bytes,
    atomic_write_json,
    check_disk,
    free_bytes,
    git_commit,
    is_polygon,
    load_index,
    load_split,
    overlap_fraction,
    percentiles,
    sha256_bytes,
    sha256_file,
    unit_hash,
)

log = get_logger("make_person_cutouts", log_file=None)

NOTE = "COCO person cutouts (cutout_source group only); COCO sanity set, not arena"
PNG_COMPRESSION = 6
SIZE_SAMPLE = 50  # cutouts encoded to estimate the PNG size


@dataclass(frozen=True)
class CutoutRules:
    min_source_height_px: float
    border_margin_px: float
    max_overlap_fraction: float
    min_mask_fraction: float
    wide_aspect: float

    @classmethod
    def from_config(cls, cfg: CocoConfig) -> CutoutRules:
        c = cfg["cutouts"]
        return cls(
            float(c["min_source_height_px"]),
            float(c["border_margin_px"]),
            float(c["max_overlap_fraction"]),
            float(c["min_mask_fraction"]),
            float(c["wide_aspect"]),
        )


def polygon_mask(segmentation: list[list[float]], width: int, height: int) -> np.ndarray:
    """uint8 mask (0/255) of a COCO polygon list; each polygon filled on its own."""
    mask = np.zeros((height, width), np.uint8)
    for poly in segmentation:
        if len(poly) < 6:
            continue
        pts = np.round(np.asarray(poly, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def reject_reasons(ann: dict[str, Any], im: CocoImage, rules: CutoutRules) -> list[str]:
    """Rejection reasons that need no mask (empty list = passes these rules)."""
    reasons = []
    if not is_polygon(ann):
        reasons.append("rle_segmentation")
    x, y, w, h = (float(v) for v in ann["bbox"])
    if h < rules.min_source_height_px:
        reasons.append("too_short")
    m = rules.border_margin_px
    if x <= m or y <= m or x + w >= im.width - m or y + h >= im.height - m:
        reasons.append("touches_border")
    others = [o for o in im.persons if o["id"] != ann["id"]]
    if any(overlap_fraction([x, y, w, h], o["bbox"]) > rules.max_overlap_fraction for o in others):
        reasons.append("overlap")
    return reasons


def cutout_from_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """BGRA crop tight to the mask; colour zeroed outside it. Returns (crop, [x, y, w, h])."""
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
    crop = image[y : y + h, x : x + w].copy()
    alpha = mask[y : y + h, x : x + w]
    crop[alpha == 0] = 0
    return np.dstack([crop, alpha]), [int(x), int(y), int(w), int(h)]


def encode_png(bgra: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def plan_cutouts(
    ids: set[int], index: dict[int, CocoImage], rules: CutoutRules
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], int]:
    """Accepted cutouts (no pixels yet), first-reason and all-reason counts, candidates."""
    accepted: list[dict[str, Any]] = []
    first: Counter[str] = Counter()
    every: Counter[str] = Counter()
    candidates = 0
    for image_id in sorted(ids):
        im = index[image_id]
        for ann in im.people:
            candidates += 1
            reasons = reject_reasons(ann, im, rules)
            mask_fraction = None
            if "rle_segmentation" not in reasons:
                mask = polygon_mask(ann["segmentation"], im.width, im.height)
                x, y, w, h = (float(v) for v in ann["bbox"])
                x0, y0 = int(np.floor(x)), int(np.floor(y))
                x1, y1 = int(np.ceil(x + w)), int(np.ceil(y + h))
                inside = int(np.count_nonzero(mask[y0:y1, x0:x1]))
                mask_fraction = inside / max(w * h, 1e-9)
                if not mask.any():
                    reasons.append("empty_mask")
                elif mask_fraction < rules.min_mask_fraction:
                    reasons.append("mask_fraction")
            if reasons:
                first[reasons[0]] += 1
                every.update(reasons)
                continue
            mx, my, mw, mh = cv2.boundingRect(cv2.findNonZero(mask))
            accepted.append(
                {
                    "file": f"{image_id:012d}_{ann['id']}.png",
                    "source_image_id": image_id,
                    "source_file": im.file_name,
                    "ann_id": int(ann["id"]),
                    "coco_bbox_xywh": [float(v) for v in ann["bbox"]],
                    "cutout_box_xywh": [int(mx), int(my), int(mw), int(mh)],
                    "width": int(mw),
                    "height": int(mh),
                    "aspect_w_over_h": round(mw / mh, 4),
                    "mask_fraction_of_coco_box": round(mask_fraction, 4),
                }
            )
    return accepted, first, every, candidates


def make_cutout(images: ImageZip, im: CocoImage, entry: dict[str, Any]) -> np.ndarray:
    ann = next(a for a in im.people if a["id"] == entry["ann_id"])
    mask = polygon_mask(ann["segmentation"], im.width, im.height)
    bgra, box = cutout_from_mask(images.decode(im.file_name), mask)
    if box != entry["cutout_box_xywh"]:
        raise RuntimeError(
            f"cutout box changed for {entry['file']}: {box} != {entry['cutout_box_xywh']}"
        )
    return bgra


def estimate_bytes(
    accepted: list[dict[str, Any]], index: dict[int, CocoImage], images: ImageZip, seed: int
) -> tuple[int, int]:
    """(raw RGBA bytes of all cutouts, PNG estimate from a seeded sample)."""
    raw = sum(e["width"] * e["height"] * 4 for e in accepted)
    if not accepted:
        return 0, 0
    sample = sorted(accepted, key=lambda e: unit_hash(seed, f"size:{e['file']}"))[:SIZE_SAMPLE]
    png = sum(len(encode_png(make_cutout(images, index[e["source_image_id"]], e))) for e in sample)
    sample_raw = sum(e["width"] * e["height"] * 4 for e in sample)
    return raw, int(raw * png / sample_raw)


def run(
    cfg: CocoConfig, plan_only: bool = False, yes: bool = False, overwrite: bool = False
) -> dict[str, Any]:
    paths = cfg["paths"]
    out = Path(paths["cutouts"])
    rules = CutoutRules.from_config(cfg)
    split = load_split(paths["split"])
    index = load_index(paths["instances"])
    accepted, first, every, candidates = plan_cutouts(split["cutout_source"], index, rules)

    with ImageZip(paths["images_zip"]) as images:
        raw, estimate = estimate_bytes(accepted, index, images, int(cfg["split"]["seed"]))
        print(
            f"{len(accepted)} cutouts of {candidates} non-crowd persons; estimated PNG size "
            f"{estimate / GB:.3f} GB (raw RGBA {raw / GB:.3f} GB); free {free_bytes(out) / GB:.2f} GB"
        )
        if plan_only:
            return {"accepted": len(accepted), "estimated_bytes": estimate}
        check_disk(out, estimate, cfg, "write cutouts")
        if estimate > float(cfg["disk"]["confirm_above_gb"]) * GB and not yes:
            raise ConfirmationNeeded(
                f"estimated {estimate / GB:.3f} GB is above {cfg['disk']['confirm_above_gb']} GB; "
                "re-run with --yes after the owner agrees"
            )
        if out.exists() and any(out.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{out} is not empty; use --overwrite to rebuild it")
            shutil.rmtree(out)
        for e in accepted:
            data = encode_png(make_cutout(images, index[e["source_image_id"]], e))
            atomic_write_bytes(out / "images" / e["file"], data)
            e["sha256"] = sha256_bytes(data)

    heights = [e["height"] for e in accepted]
    aspects = [e["aspect_w_over_h"] for e in accepted]
    wide = sum(1 for a in aspects if a > rules.wide_aspect)
    manifest = {
        "note": NOTE,
        "rules": rules.__dict__,
        "config_hash": cfg.hash(),
        "git_commit": git_commit(),
        "split_sha256": sha256_file(paths["split"]),
        "alpha": "binary 0/255 from the COCO polygon; colour zeroed where alpha is 0",
        "counts": {
            "source_images": len(split["cutout_source"]),
            "candidate_persons": candidates,
            "accepted": len(accepted),
            "rejected_first_reason": dict(first.most_common()),
            "rejected_any_reason": dict(every.most_common()),
            "rle_skipped": every["rle_segmentation"],
            f"wide_aspect_over_{rules.wide_aspect:g}": wide,
        },
        "height_px": percentiles(heights),
        "width_px": percentiles([e["width"] for e in accepted]),
        "aspect_w_over_h": percentiles(aspects),
        "wide_share": round(wide / len(accepted), 4) if accepted else None,
        "estimated_bytes": estimate,
        "cutouts": accepted,
    }
    atomic_write_json(out / "manifest.json", manifest)
    print(f"\n== cutouts ({NOTE}) ==")
    print(
        f"  accepted {len(accepted)} of {candidates} non-crowd persons in {len(split['cutout_source'])} images"
    )
    print(f"  rejected (first failing rule): {dict(first.most_common())}")
    print(f"  rejected (every failing rule): {dict(every.most_common())}")
    print(f"  height px: {manifest['height_px']}")
    print(f"  aspect w/h: {manifest['aspect_w_over_h']}")
    print(f"  wide (w/h > {rules.wide_aspect:g}): {wide} ({manifest['wide_share']})")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="configs/coco.yaml")
    ap.add_argument("--plan", action="store_true", help="plan and estimate only, write nothing")
    ap.add_argument("--yes", action="store_true", help="write even above disk.confirm_above_gb")
    ap.add_argument("--overwrite", action="store_true", help="rebuild an existing cutout folder")
    args = ap.parse_args(argv)
    try:
        run(
            CocoConfig.from_yaml(args.config),
            plan_only=args.plan,
            yes=args.yes,
            overwrite=args.overwrite,
        )
    except DiskSpaceError as exc:
        log.error(str(exc))
        return 4
    except ConfirmationNeeded as exc:
        log.error(str(exc))
        return 5
    except FileExistsError as exc:
        log.error(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
