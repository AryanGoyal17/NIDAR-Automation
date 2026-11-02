"""
tests/test_cv/test_finetune.py — Training wrapper without training
==================================================================

A stub replaces ``ultralytics.YOLO`` (no GPU, no weights downloaded): dataset
refusals, deterministic subsets, the run record's fields, and the batch choice.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("yaml")
pytest.importorskip("ultralytics")

import cv.train.finetune as ft  # noqa: E402
from cv.utils.run_info import sha256_file  # noqa: E402


def variant(tmp_path: Path, n: int = 20, leakage_errors: int = 0) -> Path:
    d = tmp_path / "yolo_ds" / "e1"
    d.mkdir(parents=True)
    imgs = [f"{(tmp_path / 'run' / 'images' / f'f{k:03d}.jpg').as_posix()}\n" for k in range(n)]
    (d / "train.txt").write_text("".join(imgs), encoding="utf-8")
    (d / "val.txt").write_text(imgs[0], encoding="utf-8")
    (d / "data.yaml").write_text("names:\n  0: person\n", encoding="utf-8")
    manifest = {
        "leakage": {"errors": leakage_errors},
        "lists_sha256": {p: sha256_file(d / f"{p}.txt") for p in ("train", "val")},
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


class TestDataset:
    def test_valid(self, tmp_path) -> None:
        d = variant(tmp_path)
        assert len(ft.check_variant(d)["manifest_sha256"]) == 64

    def test_missing_manifest(self, tmp_path) -> None:
        d = variant(tmp_path)
        (d / "manifest.json").unlink()
        with pytest.raises(ft.DatasetRefused):
            ft.check_variant(d)

    def test_leakage_errors(self, tmp_path) -> None:
        with pytest.raises(ft.DatasetRefused, match="leakage"):
            ft.check_variant(variant(tmp_path, leakage_errors=2))

    def test_list_changed_after_build(self, tmp_path) -> None:
        d = variant(tmp_path)
        with open(d / "train.txt", "a", encoding="utf-8") as f:
            f.write("D:/data/coco_person/sanity/images/000000000101.jpg\n")
        with pytest.raises(ft.DatasetRefused, match="changed"):
            ft.check_variant(d)

    def test_subset_deterministic_sorted(self, tmp_path) -> None:
        d = variant(tmp_path, n=50)
        a = ft.subset_list(d / "train.txt", 10, 0, tmp_path / "a.txt").read_text()
        b = ft.subset_list(d / "train.txt", 10, 0, tmp_path / "b.txt").read_text()
        assert a == b and a.splitlines() == sorted(a.splitlines()) and len(a.splitlines()) == 10
        assert ft.subset_list(d / "train.txt", 10, 1, tmp_path / "c.txt").read_text() != a
        with pytest.raises(ValueError):
            ft.subset_list(d / "train.txt", 51, 0, tmp_path / "d.txt")


class FakeYOLO:
    """Records train() arguments and fires the epoch callbacks once."""

    calls: list[dict] = []

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.callbacks: dict = {}
        self.trainer = None

    def add_callback(self, event, fn) -> None:
        self.callbacks[event] = fn

    def train(self, **kwargs) -> None:
        FakeYOLO.calls.append(kwargs)
        args = SimpleNamespace(**kwargs, mosaic=1.0, fliplr=0.5, flipud=0.0, hsv_h=0.015)
        self.trainer = SimpleNamespace(
            args=args,
            batch_size=kwargs["batch"],
            epoch=0,
            train_loader=[0] * 3,
            accumulate=4,
            save_dir=kwargs["project"] + "/" + kwargs["name"],
        )
        for event in ("on_train_epoch_start", "on_train_epoch_end", "on_fit_epoch_end"):
            self.callbacks[event](self.trainer)
        self.trainer.epoch = 1  # Ultralytics fires on_fit_epoch_end again after the final val
        self.callbacks["on_fit_epoch_end"](self.trainer)


@pytest.fixture
def stubbed(tmp_path, monkeypatch) -> dict:
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(ft, "power_state", lambda: "power plan Test, plugged in (AC)")
    monkeypatch.setattr(ft.GpuMemoryPeak, "_query", lambda self: (1000, 6144))
    weights = tmp_path / "models"
    weights.mkdir()
    (weights / "yolo11n.pt").write_bytes(b"fake")
    monkeypatch.setattr(ft, "MODEL_DIR", weights)
    variant(tmp_path, n=30)
    cfg = {
        "dataset_dir": str(tmp_path / "yolo_ds"),
        "project": str(tmp_path / "runs"),
        "imgsz": 416,
        "epochs": 40,
        "patience": 10,
        "batch": 16,
        "seed": 0,
        "single_cls": True,
        "device": 0,
        "workers": 2,
        "cache": False,
    }
    FakeYOLO.calls = []
    return {"cfg": cfg, "tmp": tmp_path}


class TestRunRecord:
    def test_fields(self, stubbed) -> None:
        rec = ft.train_one("yolo11n.pt", "e1", stubbed["cfg"], epochs=1, subset=12, tag="dry")
        saved = json.loads(Path(rec["save_dir"], "run_record.json").read_text(encoding="utf-8"))
        for key in (
            "args",
            "libraries",
            "git",
            "dataset_manifest_sha256",
            "power_start",
            "power_end",
            "started_utc",
            "finished_utc",
            "augmentation",
            "epochs",
            "train_images",
            "iterations_per_epoch",
            "gpu_peak_nvidia_smi_mib",
        ):
            assert key in saved, key
        assert set(saved["git"]) == {"commit", "dirty"}
        assert saved["status"] == "finished" and saved["train_images"] == 12
        assert saved["iterations_per_epoch"] == 1  # ceil(12 / 16)
        assert saved["augmentation"]["flipud"] == 0.0 and saved["augmentation"]["mosaic"] == 1.0
        assert saved["ultralytics_events_enabled"] is False
        assert len(saved["epochs"]) == 1 and "final_validation_s" in saved
        assert not any(k.startswith("_") for k in saved)
        call = FakeYOLO.calls[0]
        assert call["imgsz"] == 416 and call["single_cls"] and call["seed"] == 0
        assert call["epochs"] == 1 and call["workers"] == 2 and call["cache"] is False
        assert Path(call["data"]).read_text(encoding="utf-8").count("train_subset.txt") == 1

    def test_refuses_bad_dataset(self, stubbed) -> None:
        (Path(stubbed["cfg"]["dataset_dir"]) / "e1" / "manifest.json").unlink()
        with pytest.raises(ft.DatasetRefused):
            ft.train_one("yolo11n.pt", "e1", stubbed["cfg"])
        assert FakeYOLO.calls == []


class TestBatchChoice:
    def row(self, model: str, batch: int, frac: float, oom: bool = False) -> dict:
        return {"model": model, "batch": batch, "fraction": frac, "oom": oom}

    def test_largest_under_limit_and_fitting(self) -> None:
        rows = [
            self.row(m, b, f)
            for m in ("a", "b")
            for b, f in ((8, 0.3), (16, 0.5), (32, 0.79), (64, 0.9 if m == "a" else 0.7))
        ]
        rows.append(self.row("a", 128, 0.99, oom=True))
        out = ft.choose_batch(rows, ["a", "b"], 0.80)
        assert out == {"largest_fitting": 64, "largest_under_limit": 32}

    def test_missing_model_row_not_chosen(self) -> None:
        rows = [self.row("a", 8, 0.3), self.row("b", 8, 0.3), self.row("a", 16, 0.4)]
        assert ft.choose_batch(rows, ["a", "b"], 0.8)["largest_under_limit"] == 8
