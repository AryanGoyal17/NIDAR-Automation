"""
tests/test_cv/test_eval_metrics.py — Matching, metrics, sweep, height bins, inputs
==================================================================================

Hand-made boxes only (no model, no downloads).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cv.eval.metrics import (
    average_precision,
    counts_at,
    height_bin,
    iou_matrix,
    match_image,
    prf,
    recall_by_height,
    sweep,
    threshold_at_fp_budget,
)

GT = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)


class TestMatching:
    def test_iou(self) -> None:
        iou = iou_matrix(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10], [0, 0, 10, 10]]))
        assert iou[0, 0] == pytest.approx(50 / 150) and iou[0, 1] == 1.0
        assert iou_matrix(np.zeros((0, 4)), GT).shape == (0, 2)

    def test_tp_fp_fn(self) -> None:
        preds = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
        m = match_image(preds, [0.9, 0.8], GT)
        assert m.tp.tolist() == [True, False]
        assert counts_at([m], 0.0) == {"tp": 1, "fp": 1, "fn": 1}

    def test_duplicate_is_fp_and_score_order_wins(self) -> None:
        # two predictions on one GT: the higher score is the TP, even if listed second
        preds = np.array([[0, 0, 10, 10], [0, 0, 10, 9]], float)
        m = match_image(preds, [0.3, 0.7], GT[:1])
        assert m.tp.tolist() == [False, True]
        assert m.gt_matched_score.tolist() == [0.7]

    def test_iou_threshold_edges(self) -> None:
        # IoU exactly 0.5 is a TP; just under is not
        at = np.array([[0, 0, 10, 20]], float)  # GT 0..10 x 0..10 inside: IoU 100/200
        assert match_image(at, [0.9], GT[:1]).tp.tolist() == [True]
        under = np.array([[0, 0, 10, 20.5]], float)
        assert match_image(under, [0.9], GT[:1]).tp.tolist() == [False]

    def test_takes_best_unmatched_gt(self) -> None:
        # one prediction overlapping both GTs, better with the second
        gts = np.array([[0, 0, 10, 10], [2, 0, 12, 10]], float)
        m = match_image(np.array([[2, 0, 12, 10]], float), [0.9], gts)
        assert m.gt_matched_score[1] == 0.9 and m.gt_matched_score[0] == -math.inf

    def test_empty_cases(self) -> None:
        m = match_image(np.zeros((0, 4)), [], GT)
        assert counts_at([m], 0.5) == {"tp": 0, "fp": 0, "fn": 2}
        m = match_image(np.array([[0, 0, 1, 1]], float), [0.6], np.zeros((0, 4)))
        assert counts_at([m], 0.5) == {"tp": 0, "fp": 1, "fn": 0}
        assert counts_at([m], 0.7) == {"tp": 0, "fp": 0, "fn": 0}

    def test_prf(self) -> None:
        assert prf(3, 1, 1) == {"precision": 0.75, "recall": 0.75, "f1": 0.75}
        assert prf(0, 0, 0) == {"precision": None, "recall": None, "f1": None}
        assert prf(0, 2, 0)["recall"] is None and prf(0, 2, 1)["f1"] is None


class TestAPAndSweep:
    def test_ap_perfect_and_none(self) -> None:
        m = match_image(GT, [0.9, 0.8], GT)
        assert average_precision([m]) == pytest.approx(0.995)  # Ultralytics integration
        assert average_precision([match_image(np.zeros((0, 4)), [], np.zeros((0, 4)))]) is None

    def test_ap_half(self) -> None:
        # one of two GT found at precision 1: about 0.5 (0.495 on the 101-point grid)
        m = match_image(GT[:1], [0.9], GT)
        assert average_precision([m]) == pytest.approx(0.495)

    def test_ap_equals_ultralytics_compute_ap(self) -> None:
        metrics = pytest.importorskip("ultralytics.utils.metrics")
        m = match_image(GT[:1], [0.9], GT)
        ap, _, _ = metrics.compute_ap([0.5], [1.0])
        assert average_precision([m]) == pytest.approx(ap)
        # two images, mixed TP/FP: recall/precision curve in score order
        a = match_image(np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float), [0.9, 0.6], GT)
        b = match_image(np.array([[20, 0, 30, 10]], float), [0.4], GT)
        ap, _, _ = metrics.compute_ap([0.25, 0.25, 0.5], [1.0, 0.5, 2 / 3])
        assert average_precision([a, b]) == pytest.approx(ap)

    def test_sweep_monotone(self) -> None:
        rng = np.random.default_rng(0)
        matches = []
        for _ in range(20):
            gt = rng.uniform(0, 100, (3, 2))
            gt = np.hstack([gt, gt + 10])
            pr = np.vstack([gt + rng.normal(0, 2, gt.shape), rng.uniform(0, 100, (2, 4))])
            pr[:, 2:] = np.maximum(pr[:, 2:], pr[:, :2] + 1)
            matches.append(match_image(pr, rng.uniform(0, 1, len(pr)), gt))
        rows = sweep(matches, [0.1, 0.3, 0.5, 0.8])
        rec = [r["recall"] for r in rows]
        fpi = [r["fp_per_image"] for r in rows]
        assert rec == sorted(rec, reverse=True) and fpi == sorted(fpi, reverse=True)

    def test_fp_budget_picks_lowest_feasible(self) -> None:
        # 2 images; FPs at 0.6 and 0.3, TPs at 0.9 and 0.2
        a = match_image(np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float), [0.9, 0.6], GT[:1])
        b = match_image(np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float), [0.2, 0.3], GT[:1])
        # budget 0.5 FP/image -> at most 1 FP over 2 images -> threshold 0.6 (FPs: 0.6 only)
        r = threshold_at_fp_budget([a, b], 0.5)
        assert r["conf"] == 0.6 and r["fp"] == 1 and r["tp"] == 1
        # budget 1.0 -> 2 FP allowed -> lowest score 0.2, both TPs
        r = threshold_at_fp_budget([a, b], 1.0)
        assert r["conf"] == 0.2 and r["tp"] == 2 and r["fp"] == 2
        # min_conf respected
        assert threshold_at_fp_budget([a, b], 1.0, min_conf=0.25)["conf"] == 0.3
        # infeasible
        assert threshold_at_fp_budget([a, b], 0.0, min_conf=0.95)["conf"] is None


class TestHeightBins:
    def test_edges(self) -> None:
        assert height_bin(0) == "<20" and height_bin(19.99) == "<20"
        assert height_bin(20) == "20-40" and height_bin(39.9) == "20-40"
        assert height_bin(40) == "40-80" and height_bin(80) == ">80"

    def test_recall_by_height(self) -> None:
        gt = np.array([[0, 0, 10, 10], [20, 0, 30, 100]], float)
        m = match_image(gt[1:], [0.9], gt)
        out = recall_by_height([m], [np.array([15.0, 90.0])], 0.5)
        assert out["<20"] == {"gt": 1, "tp": 0, "recall": 0.0}
        assert out[">80"] == {"gt": 1, "tp": 1, "recall": 1.0}
        assert out["20-40"]["recall"] is None


class TestEvaluateInputs:
    @pytest.fixture(autouse=True)
    def _need_cv2(self) -> None:
        pytest.importorskip("cv2")
        pytest.importorskip("yaml")

    def test_person_class_and_other_classes_ignored(self) -> None:
        from cv.eval.evaluate import keep_person, person_class

        coco = {0: "person", 1: "bicycle", 2: "car"}
        assert person_class(coco) == 0 and person_class({0: "person"}) == 0
        boxes = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]], float)
        kept, scores = keep_person(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 2, 0]), 0)
        assert scores.tolist() == [0.9, 0.7] and kept.tolist() == [[0, 0, 1, 1], [2, 2, 3, 3]]
        with pytest.raises(ValueError):
            person_class({0: "car"})

    def test_labels_on_every_table(self) -> None:
        from cv.eval.evaluate import COCO_LABEL, SIM_LABEL, data_label

        assert data_label("sim_val", "x.txt") == SIM_LABEL
        assert data_label("anchor_val", "data/yolo_ds/eval/anchor_val.txt") == COCO_LABEL
        assert data_label("sanity", "data/yolo_ds/eval/sanity.txt") == COCO_LABEL
        assert "not arena" in data_label("other", "folder")

    def test_list_file_and_folder_inputs(self, tmp_path: Path) -> None:
        from cv.eval.evaluate import image_paths, read_gt

        for k in range(3):
            img = tmp_path / "set" / "images" / f"{k}.jpg"
            img.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(b"")
            lab = tmp_path / "set" / "labels" / f"{k}.txt"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text("0 0.5 0.5 0.5 0.25\n" if k else "", encoding="utf-8")
        lst = tmp_path / "l.txt"
        lst.write_text("".join(f"{p}\n" for p in (tmp_path / "set" / "images").glob("*.jpg")))
        assert image_paths(lst) == image_paths(tmp_path / "set" / "images")
        gt = read_gt(tmp_path / "set" / "images" / "1.jpg", 200, 100)
        assert gt.tolist() == [[50.0, 37.5, 150.0, 62.5]]
        assert read_gt(tmp_path / "set" / "images" / "0.jpg", 200, 100).shape == (0, 4)
        (tmp_path / "set" / "labels" / "2.txt").unlink()
        with pytest.raises(FileNotFoundError):
            read_gt(tmp_path / "set" / "images" / "2.jpg", 200, 100)

    def test_sanity_cannot_choose_threshold(self, tmp_path: Path) -> None:
        pytest.importorskip("ultralytics")
        from cv.eval.evaluate import evaluate

        with pytest.raises(ValueError, match="sanity"):
            evaluate(["yolov8n.pt"], {"sanity": "x.txt"}, tmp_path, operating_set="sanity")
