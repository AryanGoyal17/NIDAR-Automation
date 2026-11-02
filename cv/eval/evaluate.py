"""
cv/eval/evaluate.py — Detector evaluation: own matching, sweep, height bins, speed
==================================================================================

For each model (fine-tuned weights or a pretrained COCO name) and each named
dataset (an Ultralytics list file or an image folder; labels at the
``/images/`` -> ``/labels/`` path):

1. predicts at a low confidence (``--pred-conf``, 0.001) at imgsz 416, keeping
   only the ``person`` class of the model (class 0 for COCO-pretrained models;
   every other predicted class is ignored);
2. matches with our greedy rule at IoU 0.5 (README section 6.8,
   ``cv/eval/metrics.py``) and computes TP, FP, FN, precision, recall, F1 and
   our AP50; cross-checks mAP50 with Ultralytics ``val`` on the same images;
3. sweeps the confidence 0.10-0.80 (step 0.05) and finds the threshold with
   at most 0.2 false positives per image (ASSUMPTION) on the **operating set**
   (``--operating-set``, default ``anchor_val``). That threshold is the
   model's operating point and is applied unchanged to every other dataset.
   The sanity set can never be the operating set;
4. reports recall per ground-truth box height at 416 (<20, 20-40, 40-80,
   >80 px), and per-image results.

Every table carries its data label: "simulation validation, not arena" or
"COCO sanity set, not arena" (COCO photos are never an arena score).

``--speed`` adds detector-only time per frame at 416 on CPU and GPU, batch 1,
with the power plan. These numbers come from the development laptop and do
not represent the Raspberry Pi.

Usage::

    python -m cv.eval.evaluate --model yolov8n.pt --model yolo11n.pt \\
        --data sim_val=data/yolo_ds/eval/sim_val.txt \\
        --data anchor_val=data/yolo_ds/eval/anchor_val.txt \\
        --data sanity=data/yolo_ds/eval/sanity.txt --out outputs/eval/baseline --speed
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv.eval.metrics import (
    HEIGHT_BINS,
    ImageMatch,
    average_precision,
    counts_at,
    match_image,
    prf,
    recall_by_height,
    sweep,
    threshold_at_fp_budget,
)
from cv.logger import get_logger
from cv.utils.offline import events_enabled, ultralytics_offline
from cv.utils.run_info import git_state, library_versions, power_state, utc_now
from data_tools.build_dataset import IMAGE_SUFFIXES, label_path
from data_tools.leakage import read_list

log = get_logger("evaluate", log_file=None)
ultralytics_offline()  # before any Ultralytics import: no usage events, no downloads

IMGSZ = 416
IOU_MATCH = 0.5
NMS_IOU = 0.7  # Ultralytics predict/val default
SWEEP = tuple(round(0.10 + 0.05 * k, 2) for k in range(15))  # 0.10 ... 0.80
FP_BUDGET = 0.2  # ASSUMPTION: false positives per image allowed at the operating point
SIM_LABEL = "simulation validation, not arena"
COCO_LABEL = "COCO sanity set, not arena"
MODEL_DIRS = (Path("models"), Path("."))


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def data_label(name: str, source: str) -> str:
    """The label every table of a dataset must carry."""
    text = f"{name} {source}".lower()
    if "sim" in text:
        return SIM_LABEL
    if "coco" in text or "anchor" in text or "sanity" in text:
        return COCO_LABEL
    if "test_real" in text:
        return "real test set (data/test_real)"
    if "test_proxy" in text:
        return "proxy test set (data/test_proxy), not arena"
    return "unlabelled source, not arena"


def is_sanity(name: str, source: str) -> bool:
    return "sanity" in f"{name} {source}".lower()


def image_paths(source: str | Path) -> list[Path]:
    """Images of a list file or a folder (searched recursively), sorted."""
    source = Path(source)
    if source.is_dir():
        paths = [p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    elif source.suffix.lower() == ".txt":
        paths = read_list(source)
    else:
        raise ValueError(f"{source}: not a folder or a .txt list file")
    if not paths:
        raise ValueError(f"{source}: no images")
    return sorted(paths)


def read_gt(image: Path, width: int, height: int) -> np.ndarray:
    """Ground-truth xyxy pixel boxes of an image from its YOLO label file."""
    lp = label_path(image)
    if not lp.exists():
        raise FileNotFoundError(f"label file missing: {lp}")
    boxes = []
    for line in lp.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if int(parts[0]) != 0:
            raise ValueError(f"{lp}: class {parts[0]}, expected 0 (person)")
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ]
        )
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def person_class(names: dict[int, str] | list[str]) -> int:
    """Index of ``person`` in a model's class names (0 for COCO-pretrained and our models)."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    found = [int(k) for k, v in items if v == "person"]
    if len(found) != 1:
        raise ValueError(f"model names have {len(found)} 'person' classes: {names}")
    return found[0]


def keep_person(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, person: int
) -> tuple[np.ndarray, np.ndarray]:
    """Drop every prediction whose class is not ``person``."""
    keep = np.asarray(classes).astype(int) == person
    return np.asarray(boxes).reshape(-1, 4)[keep], np.asarray(scores).reshape(-1)[keep]


def resolve_weights(model: str) -> str:
    p = Path(model)
    if p.exists():
        return str(p)
    for d in MODEL_DIRS:
        if (d / p.name).exists():
            return str(d / p.name)
    raise FileNotFoundError(f"weights {model} not found in {[str(d) for d in MODEL_DIRS]}")


# --------------------------------------------------------------------------- #
# Prediction and metrics
# --------------------------------------------------------------------------- #
@dataclass
class ImageResult:
    path: Path
    width: int
    height: int
    gt: np.ndarray
    match: ImageMatch
    gt_h416: np.ndarray = field(default_factory=lambda: np.zeros(0))


def predict_dataset(
    model: Any, paths: list[Path], person: int, device: str, pred_conf: float, batch: int = 16
) -> list[ImageResult]:
    results = []
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        preds = model.predict(
            [str(p) for p in chunk],
            imgsz=IMGSZ,
            conf=pred_conf,
            iou=NMS_IOU,
            classes=[person],
            device=device,
            max_det=300,
            verbose=False,
        )
        for path, r in zip(chunk, preds, strict=True):
            h, w = r.orig_shape
            b = r.boxes
            boxes, scores = keep_person(
                b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy(), person
            )
            gt = read_gt(path, w, h)
            results.append(
                ImageResult(
                    path,
                    w,
                    h,
                    gt,
                    match_image(boxes, scores, gt, IOU_MATCH),
                    (gt[:, 3] - gt[:, 1]) * IMGSZ / max(w, h),
                )
            )
    return results


def dataset_metrics(results: list[ImageResult], conf: float | None) -> dict[str, Any]:
    matches = [r.match for r in results]
    out: dict[str, Any] = {
        "images": len(results),
        "gt_boxes": int(sum(len(r.gt) for r in results)),
        "negative_images": sum(1 for r in results if not len(r.gt)),
        "ap50_ours": average_precision(matches),
        "sweep": sweep(matches, SWEEP),
        "fp_budget_on_this_set": threshold_at_fp_budget(matches, FP_BUDGET, min(SWEEP)),
    }
    if conf is not None:
        c = counts_at(matches, conf)
        out["at_operating_conf"] = {
            "conf": conf,
            **c,
            **prf(**c),
            "fp_per_image": c["fp"] / max(len(results), 1),
        }
        out["recall_by_height_416"] = recall_by_height(matches, [r.gt_h416 for r in results], conf)
    return out


def ultralytics_map50(
    weights: str, names: dict, paths: list[Path], person: int, device: str, work: Path
) -> float | None:
    """mAP50 from Ultralytics val on the same images (person class only)."""
    from ultralytics import YOLO

    work.mkdir(parents=True, exist_ok=True)
    lst = work / "images.txt"
    lst.write_text("".join(f"{p.as_posix()}\n" for p in paths), encoding="utf-8")
    yaml_lines = [f"path: {work.as_posix()}", "train: images.txt", "val: images.txt", "names:"]
    yaml_lines += [f"  {k}: {v}" for k, v in sorted(names.items())]
    (work / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    metrics = YOLO(weights).val(
        data=str(work / "data.yaml"),
        imgsz=IMGSZ,
        batch=16,
        conf=0.001,
        iou=NMS_IOU,
        classes=[person],
        device=device,
        plots=False,
        verbose=False,
        project=str(work),
        name="val",
        exist_ok=True,
        workers=0,
    )
    return float(metrics.box.map50)


def speed(weights: str, paths: list[Path], device: str, n: int = 100, warmup: int = 10) -> dict:
    """Detector-only ms per frame (pre + inference + post), batch 1, imgsz 416."""
    from ultralytics import YOLO

    model = YOLO(weights)
    frames = [cv2.imread(str(p)) for p in paths[: max(n, 1)]]
    for f in frames[:warmup]:
        model.predict(f, imgsz=IMGSZ, device=device, verbose=False)
    per, wall = [], []
    power_start = power_state()
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(f, imgsz=IMGSZ, device=device, verbose=False)[0]
        wall.append((time.perf_counter() - t0) * 1000)
        per.append(sum(r.speed.values()))
    import torch

    q = statistics.quantiles(per, n=10) if len(per) > 1 else [per[0]] * 9
    return {
        "device": device,
        "frames": len(per),
        "warmup": warmup,
        "ms_median": statistics.median(per),
        "ms_p90": q[8],
        "wall_ms_median": statistics.median(wall),
        "torch_threads": torch.get_num_threads(),
        "power_start": power_start,
        "power_end": power_state(),
        "note": "development laptop; does not represent the Raspberry Pi",
    }


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def evaluate(
    models: list[str],
    datasets: dict[str, str],
    out: Path,
    operating_set: str = "anchor_val",
    device: str = "0",
    pred_conf: float = 0.001,
    cross_check: bool = True,
    do_speed: bool = False,
) -> dict[str, Any]:
    from ultralytics import YOLO

    if operating_set not in datasets:
        raise ValueError(f"operating set {operating_set!r} not among {list(datasets)}")
    if is_sanity(operating_set, datasets[operating_set]):
        raise ValueError("the sanity set can never choose the operating threshold")
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: image_paths(src) for name, src in datasets.items()}
    record: dict[str, Any] = {
        "started_utc": utc_now(),
        "power_start": power_state(),
        "git": git_state(),
        "libraries": library_versions(),
        "args": {
            "models": models,
            "datasets": datasets,
            "operating_set": operating_set,
            "device": device,
            "pred_conf": pred_conf,
            "imgsz": IMGSZ,
            "iou_match": IOU_MATCH,
            "nms_iou": NMS_IOU,
            "sweep": list(SWEEP),
            "fp_budget_per_image": FP_BUDGET,
        },
        "operating_point_rule": (
            f"lowest confidence with at most {FP_BUDGET} FP per image on {operating_set} "
            "(never the sanity set), applied unchanged to every dataset"
        ),
        "models": {},
        "ultralytics_events_enabled": events_enabled(),
    }
    if record["ultralytics_events_enabled"]:
        raise RuntimeError("Ultralytics usage events are enabled; refusing to run")
    per_image_rows = []
    for m in models:
        weights = resolve_weights(m)
        model = YOLO(weights)
        names = {int(k): v for k, v in model.names.items()}
        person = person_class(names)
        res = {n: predict_dataset(model, p, person, device, pred_conf) for n, p in paths.items()}
        op = threshold_at_fp_budget([r.match for r in res[operating_set]], FP_BUDGET, min(SWEEP))
        conf = op["conf"]
        if conf is None:
            log.warning(f"{m}: no threshold >= {min(SWEEP)} meets the FP budget on {operating_set}")
        entry: dict[str, Any] = {
            "weights": weights,
            "pretrained_coco": len(names) == 80,
            "person_class": person,
            "operating_conf": conf,
            "datasets": {},
        }
        for name, results in res.items():
            d = dataset_metrics(results, conf)
            d["label"] = data_label(name, datasets[name])
            if cross_check:
                d["map50_ultralytics"] = ultralytics_map50(
                    weights, names, paths[name], person, device, out / "ultra" / Path(m).stem / name
                )
            entry["datasets"][name] = d
            for r in results:
                c = counts_at([r.match], conf if conf is not None else min(SWEEP))
                per_image_rows.append(
                    {
                        "model": Path(m).stem,
                        "dataset": name,
                        "label": d["label"],
                        "image": r.path.as_posix(),
                        "gt": len(r.gt),
                        **c,
                    }
                )
        if do_speed:
            sample = next(iter(paths.values()))
            entry["speed"] = [speed(weights, sample, "cpu")]
            if device != "cpu":
                entry["speed"].append(speed(weights, sample, device))
        record["models"][Path(m).stem] = entry
    record["finished_utc"] = utc_now()
    record["power_end"] = power_state()
    write_outputs(record, per_image_rows, out)
    return record


def _f(v: Any, nd: int = 3) -> str:
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_outputs(record: dict[str, Any], per_image: list[dict], out: Path) -> None:
    (out / "results.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    summary, sweeps, bins = [], [], []
    for mname, e in record["models"].items():
        for dname, d in e["datasets"].items():
            op = d.get("at_operating_conf", {})
            summary.append(
                {
                    "label": d["label"],
                    "model": mname,
                    "dataset": dname,
                    "operating_conf": e["operating_conf"],
                    "chosen_on": record["args"]["operating_set"],
                    "images": d["images"],
                    "gt_boxes": d["gt_boxes"],
                    **{k: op.get(k) for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
                    "fp_per_image": op.get("fp_per_image"),
                    "ap50_ours": d["ap50_ours"],
                    "map50_ultralytics": d.get("map50_ultralytics"),
                }
            )
            sweeps += [
                {"label": d["label"], "model": mname, "dataset": dname, **r} for r in d["sweep"]
            ]
            for b, v in d.get("recall_by_height_416", {}).items():
                bins.append(
                    {"label": d["label"], "model": mname, "dataset": dname, "bin_416": b, **v}
                )
    for fname, rows in (
        ("summary.csv", summary),
        ("sweep.csv", sweeps),
        ("recall_by_height.csv", bins),
        ("per_image.csv", per_image),
    ):
        if rows:
            with open(out / fname, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    (out / "report.md").write_text(report_md(record), encoding="utf-8")


def report_md(record: dict[str, Any]) -> str:
    a = record["args"]
    lines = [
        "# Detector evaluation",
        "",
        f"- Started {record['started_utc']}, finished {record.get('finished_utc')}; "
        f"git {record['git']['commit'][:7]} (dirty: {record['git']['dirty']}).",
        f"- Power: start {record['power_start']}; end {record.get('power_end')}.",
        f"- imgsz {a['imgsz']}, prediction conf {a['pred_conf']}, NMS IoU {a['nms_iou']}, "
        f"match IoU {a['iou_match']} (greedy by score, README 6.8).",
        f"- **Operating point:** {record['operating_point_rule']}. "
        "The COCO sanity set never chooses a threshold.",
        "",
    ]
    by_label: dict[str, list[tuple[str, str, dict, dict]]] = {}
    for mname, e in record["models"].items():
        for dname, d in e["datasets"].items():
            by_label.setdefault(dname, []).append((mname, d["label"], d, e))
    for dname, rows in by_label.items():
        label = rows[0][1]
        lines += [
            f"## {dname} — {label}",
            "",
            f"*{label}.* Threshold chosen on {a['operating_set']} and applied here unchanged.",
            "",
            "| model | conf (chosen on "
            f"{a['operating_set']}) | images | GT | TP | FP | FN | P | R | F1 | FP/img | "
            "AP50 ours | mAP50 Ultralytics |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for mname, _, d, e in rows:
            op = d.get("at_operating_conf", {})
            lines.append(
                f"| {mname} | {_f(e['operating_conf'])} | {d['images']} | {d['gt_boxes']} | "
                f"{_f(op.get('tp'))} | {_f(op.get('fp'))} | {_f(op.get('fn'))} | "
                f"{_f(op.get('precision'))} | {_f(op.get('recall'))} | {_f(op.get('f1'))} | "
                f"{_f(op.get('fp_per_image'))} | {_f(d['ap50_ours'])} | {_f(d.get('map50_ultralytics'))} |"
            )
        lines += [
            "",
            f"Recall by ground-truth box height at 416 ({label}):",
            "",
            "| model | " + " | ".join(n for n, _, _ in HEIGHT_BINS) + " |",
            "|---|" + "---|" * len(HEIGHT_BINS),
        ]
        for mname, _, d, _ in rows:
            hb = d.get("recall_by_height_416", {})
            cells = [
                f"{_f(hb[n]['recall'])} ({hb[n]['tp']}/{hb[n]['gt']})" if n in hb else "—"
                for n, _, _ in HEIGHT_BINS
            ]
            lines.append(f"| {mname} | " + " | ".join(cells) + " |")
        lines += [""]
    speeds = [(m, s) for m, e in record["models"].items() for s in e.get("speed", [])]
    if speeds:
        lines += [
            "## Detector-only speed at 416 (development laptop; does NOT represent the Raspberry Pi)",
            "",
            "| model | device | frames | ms median | ms p90 | torch threads | power (start / end) |",
            "|---|---|---|---|---|---|---|",
        ]
        for m, s in speeds:
            lines.append(
                f"| {m} | {s['device']} | {s['frames']} | {s['ms_median']:.1f} | {s['ms_p90']:.1f} | "
                f"{s['torch_threads']} | {s['power_start']} / {s['power_end']} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_data(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        name, sep, src = item.partition("=")
        if not sep or not name or not src:
            raise ValueError(f"--data {item!r}: expected name=path")
        out[name] = src
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Detector evaluation (README 6.8 matching).")
    ap.add_argument("--model", action="append", required=True, help="weights or pretrained name")
    ap.add_argument("--data", action="append", required=True, help="name=list.txt or name=folder")
    ap.add_argument("--operating-set", default="anchor_val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--pred-conf", type=float, default=0.001)
    ap.add_argument("--no-cross-check", action="store_true")
    ap.add_argument("--speed", action="store_true")
    args = ap.parse_args(argv)
    work = Path(args.out)
    evaluate(
        args.model,
        parse_data(args.data),
        work,
        args.operating_set,
        args.device,
        args.pred_conf,
        not args.no_cross_check,
        args.speed,
    )
    print((work / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
