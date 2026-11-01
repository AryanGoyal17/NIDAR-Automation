"""
data_tools/check_generated.py — Validate a generated run and print its statistics
=================================================================================

Reads only what ``sim.generator`` wrote (images, label files, meta.jsonl,
attempts.jsonl, scene summaries, manifest.json) and checks:

- every image has a label file and a meta line, and the other way round;
- label lines have 5 fields, class 0, finite values in [0, 1], a box inside
  the image and at least 1 px wide and tall (no degenerate boxes);
- label files match the meta labels; an empty label file only for a negative;
  no dropped label in a written frame;
- image size matches the meta; no duplicate frame ids, image bytes, or poses
  within a scene; each scene's summary matches its meta lines.

Prints counts per mode and per category; ``--stats`` adds the run statistics
(acceptance, retries, box sizes at imgsz, truncation / occlusion / contrast
histograms, yields per mode and posture, far-speck attempts, phase timing).

Usage::

    python -m data_tools.check_generated --run data/sim_raw/run1 --stats
    python -m data_tools.check_generated --run data/sim_raw/run1 --stats --json stats.json

Exit code 0 if the run is valid, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv.logger import get_logger

log = get_logger("check_generated", log_file=None)

# Histogram bins as in outputs/scratch/labelfix/report.md
FRACTION_BINS = [
    ("0", 0.0, 0.0),
    ("(0, 0.1)", 0.0, 0.1),
    ("[0.1, 0.25)", 0.1, 0.25),
    ("[0.25, 0.5)", 0.25, 0.5),
    ("[0.5, 0.75)", 0.5, 0.75),
    ("[0.75, 0.85)", 0.75, 0.85),
    ("[0.85, 0.90)", 0.85, 0.90),
    ("[0.90, 1]", 0.90, 1.0001),
]
CONTRAST_EDGES = [0, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 255.0001]
EPS = 1e-6


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_label_file(path: Path) -> list[list[str]]:
    """Label lines split into fields (no parsing, so malformed lines can be reported)."""
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run(run: Path) -> dict[str, Any]:
    manifest_path = run / "manifest.json"
    return {
        "meta": read_jsonl(run / "meta.jsonl"),
        "attempts": read_jsonl(run / "attempts.jsonl"),
        "summaries": [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((run / "scenes").glob("*.done.json"))
        ],
        "manifest": (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        ),
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def check_label_lines(
    lines: list[list[str]], width: int, height: int, where: str
) -> tuple[list[str], list[tuple[float, ...]]]:
    """Errors and parsed (cx, cy, w, h) boxes of one label file."""
    errors: list[str] = []
    boxes: list[tuple[float, ...]] = []
    for n, fields in enumerate(lines, start=1):
        at = f"{where}:{n}"
        if len(fields) != 5:
            errors.append(f"{at}: {len(fields)} fields, expected 5")
            continue
        if fields[0] != "0":
            errors.append(f"{at}: class {fields[0]}, expected 0")
        try:
            cx, cy, w, h = (float(v) for v in fields[1:])
        except ValueError:
            errors.append(f"{at}: non-numeric value")
            continue
        values = (cx, cy, w, h)
        if not all(math.isfinite(v) and -EPS <= v <= 1 + EPS for v in values):
            errors.append(f"{at}: value outside [0, 1]: {values}")
            continue
        if cx - w / 2 < -EPS or cx + w / 2 > 1 + EPS or cy - h / 2 < -EPS or cy + h / 2 > 1 + EPS:
            errors.append(f"{at}: box extends outside the image: {values}")
        if w * width < 1.0 or h * height < 1.0:
            errors.append(f"{at}: degenerate box ({w * width:.2f} x {h * height:.2f} px)")
        boxes.append(values)
    return errors, boxes


def validate(run: Path, data: dict[str, Any] | None = None) -> list[str]:
    """Every problem found in a run (empty list: valid)."""
    data = data if data is not None else load_run(run)
    errors: list[str] = []
    metas = data["meta"]
    if not (run / "meta.jsonl").exists():
        return [f"{run}: no meta.jsonl"]

    images = (
        {p.stem for p in (run / "images").glob("*.jpg")} if (run / "images").exists() else set()
    )
    labels = (
        {p.stem for p in (run / "labels").glob("*.txt")} if (run / "labels").exists() else set()
    )
    ids = [m["frame_id"] for m in metas]
    for fid, n in Counter(ids).items():
        if n > 1:
            errors.append(f"{fid}: {n} meta lines")
    meta_ids = set(ids)
    for fid in sorted(images - labels):
        errors.append(f"{fid}: image without label file")
    for fid in sorted(labels - images):
        errors.append(f"{fid}: label file without image")
    for fid in sorted((images | labels) - meta_ids):
        errors.append(f"{fid}: file without meta line")
    for fid in sorted(meta_ids - images):
        errors.append(f"{fid}: meta line without image")

    digests: dict[str, str] = {}
    poses: dict[tuple, str] = {}
    for m in metas:
        fid = m["frame_id"]
        img_path, lbl_path = run / m["image"], run / m["label"]
        if not img_path.exists() or not lbl_path.exists():
            continue
        raw = img_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in digests:
            errors.append(f"{fid}: same image bytes as {digests[digest]}")
        digests.setdefault(digest, fid)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        w, h = int(m["camera"]["width"]), int(m["camera"]["height"])
        if img is None:
            errors.append(f"{fid}: image does not decode")
        elif img.shape[:2] != (h, w):
            errors.append(f"{fid}: image {img.shape[1]}x{img.shape[0]}, meta {w}x{h}")

        pose = m["pose"]
        key = (m["scene_id"], round(pose["x"], 2), round(pose["y"], 2), round(pose["yaw"]) % 360)
        if key in poses:
            errors.append(f"{fid}: same pose as {poses[key]}")
        poses.setdefault(key, fid)

        line_errors, boxes = check_label_lines(read_label_file(lbl_path), w, h, m["label"])
        errors += line_errors
        expected = [lb["yolo_bbox"] for lb in m["labels"]]
        if len(boxes) != len(expected) or not all(
            np.allclose(b, e, atol=EPS) for b, e in zip(boxes, expected, strict=False)
        ):
            errors.append(f"{fid}: label file does not match meta labels")
        if (len(expected) == 0) != (m["outcome"] == "negative"):
            errors.append(f"{fid}: outcome {m['outcome']} with {len(expected)} labels")
        if m.get("dropped"):
            errors.append(f"{fid}: written frame has dropped labels (ambiguous)")

    per_scene = Counter(m["scene_id"] for m in metas)
    for s in data["summaries"]:
        if s["frames_written"] != per_scene.get(s["scene_id"], 0):
            errors.append(
                f"{s['scene_id']}: summary says {s['frames_written']} frames, "
                f"meta has {per_scene.get(s['scene_id'], 0)}"
            )
    return errors


# --------------------------------------------------------------------------- #
# Counts and statistics
# --------------------------------------------------------------------------- #
def category(meta: dict[str, Any]) -> str:
    if meta["outcome"] == "positive":
        return "positive"
    return "negative_scene_with_survivors" if meta["survivors"] else "negative_empty_scene"


def counts(data: dict[str, Any]) -> dict[str, Any]:
    metas = data["meta"]
    labels = [lb for m in metas for lb in m["labels"]]
    return {
        "frames": len(metas),
        "by_mode": dict(sorted(Counter(m["mode"] for m in metas).items())),
        "by_category": dict(sorted(Counter(category(m) for m in metas).items())),
        "by_mode_and_outcome": dict(
            sorted(Counter(f"{m['mode']}/{m['outcome']}" for m in metas).items())
        ),
        "labels": len(labels),
        "hard_labels": sum(lb["hard"] for lb in labels),
        "low_light_frames": sum(m["low_light"] for m in metas),
    }


def percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "min": round(float(a.min()), 2),
        "p10": round(float(np.percentile(a, 10)), 2),
        "median": round(float(np.median(a)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
        "max": round(float(a.max()), 2),
    }


def fraction_histogram(values: list[float]) -> dict[str, int]:
    out = {}
    for name, lo, hi in FRACTION_BINS:
        if name == "0":
            out[name] = sum(v == 0 for v in values)
        elif name == "(0, 0.1)":
            out[name] = sum(0 < v < hi for v in values)
        else:
            out[name] = sum(lo <= v < hi for v in values)
    return out


def contrast_histogram(values: list[float]) -> dict[str, int]:
    out = {}
    for lo, hi in zip(CONTRAST_EDGES[:-1], CONTRAST_EDGES[1:], strict=True):
        name = f"[{lo:g},{min(hi, 255):g}{']' if hi > 255 else ')'}"
        out[name] = sum(lo <= v < hi for v in values)
    return out


def _share(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def stats(data: dict[str, Any]) -> dict[str, Any]:
    """Run statistics from the written meta and attempt files."""
    metas, attempts, manifest = data["meta"], data["attempts"], data["manifest"]
    gcfg = manifest.get("generator_config", {})
    imgsz = gcfg.get("imgsz", 416)
    rendered = [a for a in attempts if a.get("outcome") is not None]
    slots = {(a["scene_id"], a["frame_index"]): a for a in attempts}
    slot_mode = Counter(a["mode"] for a in slots.values())
    n_slots = len(slots)
    labels = [lb for m in metas for lb in m["labels"]]

    # Mode shares and acceptance per mode
    weights = dict(zip(("aimed", "free", "negative"), gcfg.get("mode_weights", []), strict=False))
    per_mode = {}
    for mode in ("aimed", "free", "negative"):
        written = [m for m in metas if m["mode"] == mode]
        r = [a for a in rendered if a["mode"] == mode]
        per_mode[mode] = {
            "configured_weight": weights.get(mode),
            "slots": slot_mode.get(mode, 0),
            "slot_share": _share(slot_mode.get(mode, 0), n_slots),
            "attempts": sum(a["mode"] == mode for a in attempts),
            "render_attempts": len(r),
            "written": len(written),
            "positives": sum(m["outcome"] == "positive" for m in written),
            "negatives": sum(m["outcome"] == "negative" for m in written),
            "written_per_render_attempt": _share(len(written), len(r)),
            "written_per_slot": _share(len(written), slot_mode.get(mode, 0)),
            "positives_per_slot": _share(
                sum(m["outcome"] == "positive" for m in written), slot_mode.get(mode, 0)
            ),
        }

    # Attempt outcomes by reason
    reasons = Counter(a["reason"] or "accepted" for a in attempts)
    reasons_by_mode: dict[str, Counter] = defaultdict(Counter)
    for a in attempts:
        reasons_by_mode[a["mode"]][a["reason"] or "accepted"] += 1
    drop_reasons = Counter(r for a in rendered for d in a.get("dropped", []) for r in d["reasons"])

    # Far-speck case: a rejected attempt with one survivor kept and another dropped
    speck = [a for a in rendered if a["reason"] is not None and a["kept"] and a["dropped"]]
    speck_dropped = [d for a in speck for d in a["dropped"]]

    # Postures and survivor counts
    pos_by_posture: Counter[str] = Counter()
    placed_by_posture: Counter[str] = Counter()
    seen_scenes = set()
    for m in metas:
        if m["scene_id"] not in seen_scenes:
            seen_scenes.add(m["scene_id"])
            placed_by_posture.update(s["posture"] for s in m["survivors"])
        for lb in m["labels"]:
            pos_by_posture[m["survivors"][lb["survivor_id"]]["posture"]] += 1
    by_n: dict[int, dict[str, int]] = defaultdict(lambda: {"frames": 0, "positives": 0})
    for m in metas:
        by_n[len(m["survivors"])]["frames"] += 1
        by_n[len(m["survivors"])]["positives"] += m["outcome"] == "positive"

    # Phase timing per render attempt
    phases = defaultdict(list)
    for a in rendered:
        for k, v in a.get("timing", {}).items():
            phases[k].append(v)

    retries = [m["retries"] for m in metas]
    return {
        "frames_written": len(metas),
        "frame_slots": n_slots,
        "attempts": len(attempts),
        "render_attempts": len(rendered),
        "acceptance_written_per_render_attempt": _share(len(metas), len(rendered)),
        "acceptance_written_per_attempt": _share(len(metas), len(attempts)),
        "written_per_slot": _share(len(metas), n_slots),
        "retries_per_written_frame": {
            "mean": round(float(np.mean(retries)), 3) if retries else None,
            "distribution": dict(sorted(Counter(retries).items())),
        },
        "positive_share": _share(sum(m["outcome"] == "positive" for m in metas), len(metas)),
        "kept_labels": len(labels),
        "box_w_at_imgsz": percentiles([lb["box_w_at_imgsz"] for lb in labels]),
        "box_h_at_imgsz": percentiles([lb["box_h_at_imgsz"] for lb in labels]),
        "imgsz": imgsz,
        "truncation_hist": fraction_histogram([lb["truncation_fraction"] for lb in labels]),
        "occlusion_hist": fraction_histogram([lb["occlusion_fraction"] for lb in labels]),
        "contrast_hist_kept": contrast_histogram(
            [lb["contrast"] for lb in labels if lb["contrast"] is not None]
        ),
        "contrast_missing": sum(lb["contrast"] is None for lb in labels),
        "hard_labels": sum(lb["hard"] for lb in labels),
        "hard_share": _share(sum(lb["hard"] for lb in labels), len(labels)),
        "low_light_frames": sum(m["low_light"] for m in metas),
        "low_light_share": _share(sum(m["low_light"] for m in metas), len(metas)),
        "low_light_positives": sum(m["low_light"] and m["outcome"] == "positive" for m in metas),
        "per_mode": per_mode,
        "attempt_reasons": dict(sorted(reasons.items())),
        "attempt_reasons_by_mode": {k: dict(sorted(v.items())) for k, v in reasons_by_mode.items()},
        "drop_reasons_in_rendered_attempts": dict(sorted(drop_reasons.items())),
        "skip_reasons": dict(
            sorted(Counter(a["skip_reason"] for a in slots.values() if a["skip_reason"]).items())
        ),
        "far_speck": {
            "attempts": len(speck),
            "dropped_reasons": dict(
                sorted(Counter(r for d in speck_dropped for r in d["reasons"]).items())
            ),
            "dropped_box_h_at_imgsz": percentiles([d["box_h_at_imgsz"] for d in speck_dropped]),
            "dropped_box_w_at_imgsz": percentiles([d["box_w_at_imgsz"] for d in speck_dropped]),
            "dropped": [
                {
                    "scene_id": a["scene_id"],
                    "frame_index": a["frame_index"],
                    "attempt": a["attempt"],
                    "reasons": d["reasons"],
                    "box_w_at_imgsz": d["box_w_at_imgsz"],
                    "box_h_at_imgsz": d["box_h_at_imgsz"],
                }
                for a in speck
                for d in a["dropped"]
            ],
        },
        "kept_labels_by_posture": dict(sorted(pos_by_posture.items())),
        "survivors_placed_by_posture": dict(sorted(placed_by_posture.items())),
        "frames_by_survivor_count": {str(k): v for k, v in sorted(by_n.items())},
        "phase_ms_per_render_attempt": {
            k: {
                "mean": round(float(np.mean(v)), 1),
                "median": round(float(np.median(v)), 1),
            }
            for k, v in sorted(phases.items())
        },
        "timing_power": {
            "start": manifest.get("timing", {}).get("power_start"),
            "end": manifest.get("timing", {}).get("power_end"),
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a sim.generator run")
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--stats", action="store_true", help="Also print run statistics")
    parser.add_argument("--json", help="Write counts (and stats) to this JSON file")
    args = parser.parse_args(argv)

    run = Path(args.run)
    data = load_run(run)
    errors = validate(run, data)
    result: dict[str, Any] = {"run": str(run), "errors": errors, "counts": counts(data)}
    if args.stats:
        result["stats"] = stats(data)

    for e in errors[:200]:
        log.error(e)
    if len(errors) > 200:
        log.error(f"... {len(errors) - 200} more errors")
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, indent=2))
    status = "VALID" if not errors else f"INVALID ({len(errors)} errors)"
    print(f"{run}: {status}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
