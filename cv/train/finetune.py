"""
cv/train/finetune.py — Fine-tuning wrapper over Ultralytics with run records
============================================================================

Trains each model in ``configs/train.yaml`` on each dataset variant built by
``data_tools/build_dataset.py`` (E1 sim only, E2 sim + anchor_train; ADR 008).
The COCO sanity set is never in these lists and never chooses an epoch: the
best epoch is picked by Ultralytics on the variant's own ``val.txt``.

Before training, the variant's ``manifest.json`` must exist, report 0 leakage
errors, and match the list files byte for byte (sha256), so a list edited
after the build is refused.

Each run folder (``<project>/<name>``) gets ``run_record.json``: arguments,
the effective Ultralytics arguments, the augmentation values actually in
effect, library versions, git commit and dirty flag, dataset manifest sha256,
power plan at start and end, UTC start and end, per-epoch wall time, peak GPU
memory (torch reserved and nvidia-smi), images and iterations per epoch.

Ultralytics runs offline (``cv/utils/offline.py``): no usage events, and the
AMP check uses local weights instead of downloading ``yolo26n.pt``.

Usage::

    python -m cv.train.finetune                                  # all models x variants, full runs
    python -m cv.train.finetune --model yolo11n.pt --variant e2
    python -m cv.train.finetune --subset 300 --epochs 1 --tag dry   # dry run
    python -m cv.train.finetune --probe-batch --variant e2       # largest batch under 80 % GPU memory
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from cv.config import load_config
from cv.logger import get_logger
from cv.utils.offline import events_enabled, patch_amp_check, ultralytics_offline
from cv.utils.run_info import git_state, library_versions, power_state, sha256_file, utc_now
from data_tools.leakage import read_list

log = get_logger("finetune", log_file=None)
ultralytics_offline()  # before any Ultralytics import

DEFAULT_CONFIG = Path("configs/train.yaml")
MODEL_DIR = Path("models")
AUGMENTATION_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "copy_paste_mode",
    "close_mosaic",
    "multi_scale",
    "rect",
)


class DatasetRefused(ValueError):
    """The dataset variant has no valid manifest or its lists changed after the build."""


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def check_variant(variant_dir: str | Path) -> dict[str, Any]:
    """Manifest of a built variant; raise DatasetRefused if it is missing or stale."""
    d = Path(variant_dir)
    mpath = d / "manifest.json"
    if not mpath.exists() or not (d / "data.yaml").exists():
        raise DatasetRefused(f"{d}: no manifest.json or data.yaml; run data_tools.build_dataset")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("leakage", {}).get("errors") != 0:
        raise DatasetRefused(f"{d}: manifest does not report 0 leakage errors")
    for part, digest in manifest.get("lists_sha256", {}).items():
        if sha256_file(d / f"{part}.txt") != digest:
            raise DatasetRefused(f"{d}/{part}.txt changed after the build (sha256 mismatch)")
    if set(manifest.get("lists_sha256", {})) != {"train", "val"}:
        raise DatasetRefused(f"{d}: manifest must list train and val")
    return {"manifest": manifest, "manifest_sha256": sha256_file(mpath)}


def subset_list(train_list: Path, n: int, seed: int, out: Path) -> Path:
    """A deterministic n-image subset of a list file (sorted), written to ``out``."""
    paths = sorted(p.as_posix() for p in read_list(train_list))
    if n > len(paths):
        raise ValueError(f"subset {n} > {len(paths)} images in {train_list}")
    chosen = sorted(random.Random(seed).sample(paths, n))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{p}\n" for p in chosen), encoding="utf-8", newline="\n")
    return out


def run_data_yaml(variant_dir: Path, train_list: Path, run_dir: Path) -> Path:
    """data.yaml for a run: the variant's val list and the given train list."""
    text = (
        f"# written by cv/train/finetune.py for {variant_dir.as_posix()}\n"
        f"train: {train_list.resolve().as_posix()}\n"
        f"val: {(variant_dir / 'val.txt').resolve().as_posix()}\n"
        "names:\n  0: person\n"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "data.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
class GpuMemoryPeak:
    """Peak ``nvidia-smi`` memory.used (MiB) on GPU 0 while active (polled)."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self.total_mib: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _query(self) -> tuple[int, int] | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            used, total = (int(v) for v in out.strip().split(","))
            return used, total
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            q = self._query()
            if q:
                self.peak_mib = max(self.peak_mib or 0, q[0])
                self.total_mib = q[1]
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuMemoryPeak:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=15)


def epoch_callbacks(store: dict[str, Any]) -> dict[str, Any]:
    """Ultralytics callbacks recording per-epoch train and train+val wall time and memory."""
    import torch

    def on_train_epoch_start(trainer) -> None:
        store["_t0"] = time.perf_counter()
        store["_in_epoch"] = True

    def on_train_epoch_end(trainer) -> None:
        store["_t_train"] = time.perf_counter() - store["_t0"]

    def on_fit_epoch_end(trainer) -> None:
        now = time.perf_counter()
        if not store.get("_in_epoch"):
            # Ultralytics fires this again after validating best.pt at the end
            store["final_validation_s"] = now - store.get("_t_end", now)
            return
        store["_in_epoch"] = False
        store["_t_end"] = now
        store["epochs"].append(
            {
                "epoch": int(trainer.epoch) + 1,
                "train_s": store.get("_t_train"),
                "train_val_s": now - store["_t0"],
                "batch": int(trainer.batch_size),
                "iterations": len(trainer.train_loader),
                "torch_max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
                if torch.cuda.is_available()
                else None,
            }
        )

    return {
        "on_train_epoch_start": on_train_epoch_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
    }


def augmentation_in_effect(args: Any) -> dict[str, Any]:
    """Augmentation values from the trainer's resolved arguments."""
    return {k: getattr(args, k) for k in AUGMENTATION_KEYS if hasattr(args, k)}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_one(
    model_name: str,
    variant: str,
    cfg: dict[str, Any],
    epochs: int | None = None,
    subset: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
    tag: str = "",
    val: bool = True,
) -> dict[str, Any]:
    """Train one model on one variant; returns the run record (also written to the run folder)."""
    import torch
    from ultralytics import YOLO

    if events_enabled():
        raise RuntimeError("Ultralytics usage events are enabled; refusing to run")
    patch_amp_check()
    variant_dir = Path(cfg["dataset_dir"]) / variant
    ds = check_variant(variant_dir)
    stamp = utc_now().replace(":", "").replace("-", "").replace(".", "")
    name = f"{variant}_{Path(model_name).stem}{'_' + tag if tag else ''}_{stamp}"
    run_dir = Path(cfg["project"]) / name
    train_list = variant_dir / "train.txt"
    if subset:
        train_list = subset_list(train_list, subset, int(cfg["seed"]), run_dir / "train_subset.txt")
    n_train = len(read_list(train_list))
    data_yaml = run_data_yaml(variant_dir, train_list, run_dir)
    weights = MODEL_DIR / Path(model_name).name
    if not weights.exists():
        raise FileNotFoundError(f"{weights} missing; run scripts/download_weights.py once")
    train_args = {
        "data": str(data_yaml),
        "imgsz": int(cfg["imgsz"]),
        "epochs": int(epochs or cfg["epochs"]),
        "patience": int(cfg["patience"]),
        "batch": int(batch or cfg["batch"]),
        "seed": int(cfg["seed"]),
        "single_cls": bool(cfg["single_cls"]),
        "device": cfg["device"],
        "workers": int(cfg["workers"] if workers is None else workers),
        "cache": cfg["cache"],
        "project": str(Path(cfg["project"]).resolve()),
        "name": name,
        "exist_ok": True,
        "val": val,
        "verbose": False,
    }
    record: dict[str, Any] = {
        "started_utc": utc_now(),
        "power_start": power_state(),
        "git": git_state(),
        "libraries": library_versions(),
        "model": str(weights),
        "weights_sha256": sha256_file(weights),
        "variant": variant,
        "dataset_manifest": (variant_dir / "manifest.json").as_posix(),
        "dataset_manifest_sha256": ds["manifest_sha256"],
        "train_list": train_list.as_posix(),
        "train_list_sha256": sha256_file(train_list),
        "subset": subset,
        "train_images": n_train,
        "iterations_per_epoch": math.ceil(n_train / train_args["batch"]),
        "args": train_args,
        "ultralytics_events_enabled": events_enabled(),
        "amp_check": "local comparison on models/yolov8n.pt (cv/utils/offline.py)",
        "epochs": [],
        "status": "running",
    }
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    model = YOLO(str(weights))
    for event, fn in epoch_callbacks(record).items():
        model.add_callback(event, fn)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        with GpuMemoryPeak() as gpu:
            model.train(**train_args)
        record["status"] = "finished"
    except Exception as exc:  # recorded, then re-raised
        record["status"] = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        for key in ("_t0", "_t_train", "_t_end", "_in_epoch"):
            record.pop(key, None)
        record["wall_s"] = time.perf_counter() - t0
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            record["effective_args"] = {k: str(v) for k, v in vars(trainer.args).items()}
            record["augmentation"] = augmentation_in_effect(trainer.args)
            record["batch_used"] = int(trainer.batch_size)
            record["oom_retries"] = int(getattr(trainer, "_oom_retries", 0))
            record["accumulate"] = int(getattr(trainer, "accumulate", 1))
            record["save_dir"] = Path(trainer.save_dir).as_posix()
        record["gpu_peak_nvidia_smi_mib"] = gpu.peak_mib if "gpu" in locals() else None
        record["gpu_total_mib"] = gpu.total_mib if "gpu" in locals() else None
        record["torch_max_reserved_mib"] = (
            torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else None
        )
        record["finished_utc"] = utc_now()
        record["power_end"] = power_state()
        (run_dir / "run_record.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
    return record


def probe_batches(
    cfg: dict[str, Any],
    models: list[str],
    variant: str,
    workers: int | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    """1 epoch on the probe subset per batch size; the largest batch under the memory limit."""
    p = cfg["probe"]
    limit = float(p["gpu_memory_fraction_max"])
    rows = []
    for m in models:
        for b in p["batches"]:
            rec = train_one(
                m,
                variant,
                cfg,
                epochs=1,
                subset=int(p["subset"]),
                batch=int(b),
                workers=workers,
                tag=f"probe_b{b}",
            )
            oom = rec.get("batch_used") != b or rec.get("oom_retries", 0) > 0
            frac = (
                rec["gpu_peak_nvidia_smi_mib"] / rec["gpu_total_mib"]
                if rec["gpu_total_mib"]
                else None
            )
            rows.append(
                {
                    "model": m,
                    "batch": b,
                    "batch_used": rec.get("batch_used"),
                    "oom": oom,
                    "gpu_peak_nvidia_smi_mib": rec["gpu_peak_nvidia_smi_mib"],
                    "gpu_total_mib": rec["gpu_total_mib"],
                    "fraction": frac,
                    "torch_max_reserved_mib": rec["torch_max_reserved_mib"],
                    "wall_s": rec["wall_s"],
                    "power": f"{rec['power_start']} / {rec['power_end']}",
                }
            )
            log.info(json.dumps(rows[-1]))
            if out is not None:  # after every row: a crash (seen after an OOM retry) keeps the rows
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
            if oom:
                break
    return {"rows": rows, "limit_fraction": limit, **choose_batch(rows, models, limit)}


def choose_batch(rows: list[dict], models: list[str], limit: float) -> dict[str, int | None]:
    """Largest batch that ran without OOM for every model, and largest also under ``limit``."""

    def mine(b: int) -> list[dict]:
        return [r for r in rows if r["batch"] == b]

    batches = sorted({r["batch"] for r in rows})
    fits = [
        b for b in batches if len(mine(b)) == len(models) and not any(r["oom"] for r in mine(b))
    ]
    under = [
        b
        for b in fits
        if all(r["fraction"] is not None and r["fraction"] <= limit for r in mine(b))
    ]
    return {
        "largest_fitting": max(fits, default=None),
        "largest_under_limit": max(under, default=None),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO models on the E1/E2 lists.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--model", action="append", help="default: config models")
    ap.add_argument("--variant", action="append", help="default: config variants")
    ap.add_argument("--dataset-dir", help="override config dataset_dir (e.g. data/yolo_ds_dry)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--subset", type=int, help="train on a deterministic subset of N images")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--probe-batch", action="store_true", help="batch probe (config probe section)")
    ap.add_argument("--out", help="write the summary JSON here")
    args = ap.parse_args(argv)
    cfg = load_config(args.config).to_dict()
    if args.dataset_dir:
        cfg["dataset_dir"] = args.dataset_dir
    models = args.model or cfg["models"]
    variants = args.variant or cfg["variants"]
    if args.probe_batch:
        result = probe_batches(
            cfg, models, variants[0], args.workers, Path(args.out) if args.out else None
        )
    else:
        result = {
            "runs": [
                train_one(m, v, cfg, args.epochs, args.subset, args.batch, args.workers, args.tag)
                for v in variants
                for m in models
            ]
        }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
