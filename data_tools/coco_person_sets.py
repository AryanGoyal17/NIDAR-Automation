"""
data_tools/coco_person_sets.py — COCO anchor and sanity YOLO person sets
========================================================================

COCO sanity set, not arena. Builds two single-class (``0 = person``) YOLO sets
from the ``anchor`` and ``sanity`` groups of ``data/coco/split.json``:

    data/coco_person/anchor/{images,labels}/   real photos mixed into training
    data/coco_person/sanity/{images,labels}/   held out, never trained on
    data/coco_person/anchor.yaml, sanity.yaml  Ultralytics data files, names {0: person}

Rules (``configs/coco.yaml``; size limits from ``arena.yaml labels``):

- An image is used only if EVERY non-crowd person annotation in it is kept.
  A person whose box, letterboxed to imgsz (scale = imgsz / max(w, h)), is
  under ``min_box_height_px`` tall or ``min_box_width_px`` wide excludes the
  whole image, as does any crowd region in the person category. So no image is
  written with a visible person left unlabelled (as far as COCO labels them).
- Persons whose polygon is stored as RLE are kept: only the box is needed.
- Labels come only from COCO boxes (xywh -> normalised YOLO). The pretrained
  class names are ignored.
- Negatives: images with no person annotation at all, in seeded order, dropped
  if a caption names a person word or if the pretrained detector finds a person
  (conf >= detector.conf). The detector only excludes, it never labels.
- Images are copies of the zip members' original JPEG bytes (sha256 unchanged).

Before copying, the size of both sets is estimated from the zip; above
``disk.confirm_above_gb`` the script stops (exit 5) until re-run with ``--yes``,
and it stops (exit 4) if less than ``disk.min_free_gb`` would remain free.

Usage::

    python -m data_tools.coco_person_sets            # plan, check disk, copy
    python -m data_tools.coco_person_sets --plan     # plan and estimate only
    python -m data_tools.coco_person_sets --yes --overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cv.logger import get_logger
from data_tools.coco_common import (
    GB,
    CocoConfig,
    CocoImage,
    ConfirmationNeeded,
    DiskSpaceError,
    ImageZip,
    SizeLimits,
    atomic_write_bytes,
    atomic_write_json,
    check_disk,
    free_bytes,
    git_commit,
    load_index,
    load_split,
    percentiles,
    sha256_bytes,
    sha256_file,
    unit_hash,
    xywh_to_yolo,
)

log = get_logger("coco_person_sets", log_file=None)

SETS = ("anchor", "sanity")
NOTE = "COCO sanity set, not arena"

# A detector maps a batch of BGR images to the highest person confidence in each (0 if none)
PersonDetector = Callable[[Sequence[np.ndarray]], list[float]]


@dataclass
class ImageDecision:
    """Whether a person image is kept, why not, and its labels."""

    image_id: int
    kept: bool
    reasons: list[str] = field(default_factory=list)
    boxes: list[dict[str, Any]] = field(default_factory=list)


def classify_person_image(im: CocoImage, limits: SizeLimits) -> ImageDecision:
    """All-or-nothing: keep the image only if every non-crowd person passes the size rule."""
    reasons: set[str] = set()
    if im.crowd:
        reasons.add("person_crowd")
    scale = limits.scale(im.width, im.height)
    boxes = []
    for ann in im.people:
        x, y, w, h = (float(v) for v in ann["bbox"])
        h416, w416 = h * scale, w * scale
        if h416 < limits.min_height_px:
            reasons.add("person_too_short")
        if w416 < limits.min_width_px:
            reasons.add("person_too_narrow")
        boxes.append(
            {
                "ann_id": int(ann["id"]),
                "bbox_xywh": [x, y, w, h],
                "yolo": [round(v, 6) for v in xywh_to_yolo([x, y, w, h], im.width, im.height)],
                "height_px_at_imgsz": round(h416, 2),
                "width_px_at_imgsz": round(w416, 2),
                "rle": not isinstance(ann.get("segmentation"), list),
            }
        )
    return ImageDecision(im.id, kept=not reasons, reasons=sorted(reasons), boxes=boxes)


def caption_regex(words: Sequence[str]) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


def caption_mentions_person(im: CocoImage, pattern: re.Pattern[str]) -> str | None:
    """The first matching person word in the image's captions, or None."""
    for cap in im.captions:
        m = pattern.search(cap)
        if m:
            return m.group(1).lower()
    return None


def yolo_person_detector(weights: str, conf: float, imgsz: int, device: Any) -> PersonDetector:
    """Pretrained Ultralytics detector restricted to COCO class 0 (person)."""
    from ultralytics import YOLO

    model = YOLO(weights)

    def detect(images: Sequence[np.ndarray]) -> list[float]:
        kwargs: dict[str, Any] = {"conf": conf, "imgsz": imgsz, "classes": [0], "verbose": False}
        if device is not None:
            kwargs["device"] = device
        results = model.predict(list(images), **kwargs)
        return [float(r.boxes.conf.max()) if len(r.boxes) else 0.0 for r in results]

    return detect


def screen_negatives(
    candidates: list[CocoImage],
    seed: int,
    set_name: str,
    pattern: re.Pattern[str],
    detector: PersonDetector,
    detector_conf: float,
    images: ImageZip,
    batch: int = 16,
) -> list[dict[str, Any]]:
    """Caption and detector verdicts for every person-free candidate, in seeded order."""
    ordered = sorted(candidates, key=lambda im: unit_hash(seed, f"negative:{set_name}:{im.id}"))
    rows = [
        {"image_id": im.id, "caption_word": caption_mentions_person(im, pattern)} for im in ordered
    ]
    for start in range(0, len(ordered), batch):
        chunk = ordered[start : start + batch]
        confs = detector([images.decode(im.file_name) for im in chunk])
        for row, c in zip(rows[start : start + batch], confs, strict=True):
            row["detector_conf"] = round(c, 4)
            row["detector_person"] = c >= detector_conf
    return rows


def plan_set(
    set_name: str,
    ids: set[int],
    index: dict[int, CocoImage],
    cfg: CocoConfig,
    detector: PersonDetector,
    images: ImageZip,
) -> dict[str, Any]:
    """Decide every image of one set (no files written)."""
    ps = cfg["person_sets"]
    seed = int(cfg["split"]["seed"])
    group = [index[i] for i in sorted(ids)]
    decisions = [classify_person_image(im, cfg.limits) for im in group if im.persons]
    positives = [d for d in decisions if d.kept]
    exclusions: Counter[str] = Counter()
    for d in decisions:
        exclusions.update(d.reasons)

    candidates = [im for im in group if not im.persons]
    screened = screen_negatives(
        candidates,
        seed,
        set_name,
        caption_regex(ps["caption_person_words"]),
        detector,
        float(ps["detector"]["conf"]),
        images,
    )
    wanted = int(ps["negatives"][set_name])
    passing = [r for r in screened if not r["caption_word"] and not r["detector_person"]]
    negatives = passing[:wanted]
    exclusions["negative_caption"] = sum(1 for r in screened if r["caption_word"])
    exclusions["negative_detector"] = sum(1 for r in screened if r["detector_person"])
    exclusions["negative_over_quota"] = len(passing) - len(negatives)
    if len(negatives) < wanted:
        log.warning(
            f"{set_name}: only {len(negatives)} negatives pass the filters ({wanted} wanted)"
        )

    heights = [b["height_px_at_imgsz"] for d in positives for b in d.boxes]
    widths = [b["width_px_at_imgsz"] for d in positives for b in d.boxes]
    small_px = float(ps["small_box_report_px"])
    entries = [
        {
            "image_id": d.image_id,
            "file": index[d.image_id].file_name,
            "kind": "positive",
            "boxes": d.boxes,
        }
        for d in positives
    ] + [
        {
            "image_id": r["image_id"],
            "file": index[r["image_id"]].file_name,
            "kind": "negative",
            "boxes": [],
        }
        for r in negatives
    ]
    return {
        "set": set_name,
        "note": NOTE,
        "counts": {
            "group_images": len(group),
            "group_images_with_persons": len(decisions),
            "images": len(entries),
            "positives": len(positives),
            "negatives": len(negatives),
            "person_boxes": len(heights),
            "rle_person_boxes_kept": sum(1 for d in positives for b in d.boxes if b["rle"]),
            "person_images_excluded": sum(1 for d in decisions if not d.kept),
        },
        "exclusions_by_reason": dict(exclusions.most_common()),
        "negative_filters": {
            "candidates": len(screened),
            "removed_by_caption": exclusions["negative_caption"],
            "removed_by_detector": exclusions["negative_detector"],
            "removed_by_both": sum(
                1 for r in screened if r["caption_word"] and r["detector_person"]
            ),
            "removed_by_either": len(screened) - len(passing),
            "passed": len(passing),
            "selected": len(negatives),
        },
        "box_height_px_at_imgsz": percentiles(heights),
        "box_width_px_at_imgsz": percentiles(widths),
        f"share_boxes_under_{small_px:g}px_tall": (
            round(sum(1 for h in heights if h < small_px) / len(heights), 4) if heights else None
        ),
        "estimated_bytes": sum(images.size(e["file"]) for e in entries),
        "images": entries,
        "negative_screening": screened,
    }


def write_set(plan: dict[str, Any], out_root: Path, images: ImageZip) -> None:
    """Copy the original JPEG bytes and write YOLO labels (empty file for a negative)."""
    out = out_root / plan["set"]
    for entry in plan["images"]:
        data = images.read(entry["file"])
        atomic_write_bytes(out / "images" / entry["file"], data)
        lines = "".join(f"0 {' '.join(f'{v:.6f}' for v in b['yolo'])}\n" for b in entry["boxes"])
        atomic_write_bytes(out / "labels" / (Path(entry["file"]).stem + ".txt"), lines.encode())
        entry["sha256"] = sha256_bytes(data)


def data_yaml(set_name: str, set_dir: Path) -> str:
    """Ultralytics data file: names {0: person}, images of this set only."""
    images = set_dir.resolve().as_posix() + "/images"
    if set_name == "sanity":
        header = (
            "# COCO sanity set, not arena. EVALUATION ONLY: never train on this set.\n"
            "# Ultralytics requires a 'train' key; it points at the same images so that\n"
            "# nothing else is ever read. data_tools/leakage.py fails if these images\n"
            "# appear in a training directory.\n"
        )
    else:
        header = (
            "# COCO anchor set (real photos mixed into training). COCO sanity set, not arena.\n"
        )
    return f"{header}path: {set_dir.resolve().as_posix()}\ntrain: {images}\nval: {images}\nnames:\n  0: person\n"


def run(
    cfg: CocoConfig,
    detector: PersonDetector | None = None,
    plan_only: bool = False,
    yes: bool = False,
    overwrite: bool = False,
) -> dict[str, dict[str, Any]]:
    paths = cfg["paths"]
    out_root = Path(paths["person_sets"])
    split = load_split(paths["split"])
    index = load_index(paths["instances"], paths["captions"])
    if detector is None:
        d = cfg["person_sets"]["detector"]
        detector = yolo_person_detector(
            d["weights"], float(d["conf"]), int(d["imgsz"]), d["device"]
        )

    with ImageZip(paths["images_zip"]) as images:
        plans = {s: plan_set(s, split[s], index, cfg, detector, images) for s in SETS}
        total = sum(p["estimated_bytes"] for p in plans.values())
        for s, p in plans.items():
            c = p["counts"]
            print(
                f"{s}: {c['images']} images ({c['positives']} with persons, {c['negatives']} negatives), "
                f"{c['person_boxes']} person boxes, estimated {p['estimated_bytes'] / GB:.3f} GB"
            )
        print(f"estimated total {total / GB:.3f} GB; free {free_bytes(out_root) / GB:.2f} GB")
        for s, p in plans.items():
            atomic_write_json(out_root / f"{s}.plan.json", {k: v for k, v in p.items()})
        if plan_only:
            return plans
        check_disk(out_root, total, cfg, "copy anchor and sanity sets")
        if total > float(cfg["disk"]["confirm_above_gb"]) * GB and not yes:
            raise ConfirmationNeeded(
                f"estimated {total / GB:.3f} GB is above {cfg['disk']['confirm_above_gb']} GB; "
                "re-run with --yes after the owner agrees"
            )
        for s in SETS:
            set_dir = out_root / s
            if set_dir.exists() and any(set_dir.iterdir()):
                if not overwrite:
                    raise FileExistsError(f"{set_dir} is not empty; use --overwrite to rebuild it")
                shutil.rmtree(set_dir)
            write_set(plans[s], out_root, images)

    for s, p in plans.items():
        p["config_hash"] = cfg.hash()
        p["git_commit"] = git_commit()
        p["split_sha256"] = sha256_file(paths["split"])
        p["copy_mode"] = cfg["person_sets"]["copy_mode"]
        p["limits"] = cfg.limits.__dict__
        atomic_write_json(out_root / s / "manifest.json", p)
        (out_root / f"{s}.plan.json").unlink(missing_ok=True)
        atomic_write_bytes(out_root / f"{s}.yaml", data_yaml(s, out_root / s).encode())
        print_report(p)
    return plans


def print_report(p: dict[str, Any]) -> None:
    c, nf = p["counts"], p["negative_filters"]
    small_key = next(k for k in p if k.startswith("share_boxes_under_"))
    print(f"\n== {p['set']} ({NOTE}) ==")
    print(f"  images {c['images']}: with persons {c['positives']}, negatives {c['negatives']}")
    print(f"  person boxes {c['person_boxes']} (RLE-segmented, kept: {c['rle_person_boxes_kept']})")
    print(f"  box height at imgsz: {p['box_height_px_at_imgsz']}")
    print(f"  box width at imgsz:  {p['box_width_px_at_imgsz']}")
    print(f"  {small_key}: {p[small_key]}")
    print(
        f"  person images excluded {c['person_images_excluded']} of {c['group_images_with_persons']}"
    )
    print(
        f"  exclusions by reason (images; one image can have several): {p['exclusions_by_reason']}"
    )
    print(
        f"  negatives: {nf['candidates']} candidates, caption removed {nf['removed_by_caption']}, "
        f"detector removed {nf['removed_by_detector']} (both {nf['removed_by_both']}), "
        f"passed {nf['passed']}, selected {nf['selected']}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="configs/coco.yaml")
    ap.add_argument("--plan", action="store_true", help="plan and estimate only, copy nothing")
    ap.add_argument("--yes", action="store_true", help="copy even above disk.confirm_above_gb")
    ap.add_argument("--overwrite", action="store_true", help="rebuild existing set folders")
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
