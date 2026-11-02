"""
cv/eval/metrics.py — Box matching and detection metrics (README section 6.8)
============================================================================

Pure NumPy, no Ultralytics. Boxes are ``xyxy`` float pixels.

- Matching: predictions in order of decreasing score; each takes the unmatched
  ground-truth box with the highest IoU, and is a true positive if that IoU is
  at least ``iou_thr`` (0.5). Unmatched predictions are false positives,
  unmatched ground-truth boxes false negatives.
- Because matching is greedy in score order, dropping every prediction below a
  threshold never changes the matches of the ones above it: one match at the
  lowest confidence gives the counts at every higher threshold.
- AP: area under the interpolated precision-recall envelope, sampled at 101
  recall points, with (last recall, precision 0) and (recall 1, precision 0)
  appended: the same integration as Ultralytics 8.4 ``compute_ap`` (one GT of
  two found at precision 1 gives 0.495, a perfect result 0.995; tested against
  the installed function). Ultralytics' matching differs (it assigns
  by IoU, not by score order), so the two mAP50 values are a cross-check, not
  an identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

HEIGHT_BINS = (
    ("<20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-80", 40.0, 80.0),
    (">80", 80.0, np.inf),
)
"""Box height bins at imgsz 416, half-open [lo, hi)."""


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` (N, 4) with every box in ``b`` (M, 4)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


@dataclass
class ImageMatch:
    """Matching result of one image (predictions keep their input order)."""

    scores: np.ndarray
    tp: np.ndarray
    """True where the prediction is a true positive."""
    gt_matched_score: np.ndarray
    """Per ground-truth box: score of the prediction matched to it, or -inf."""


def match_image(
    pred_boxes: np.ndarray, scores: np.ndarray, gt_boxes: np.ndarray, iou_thr: float = 0.5
) -> ImageMatch:
    """Greedy matching in order of decreasing score (README section 6.8)."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    iou = iou_matrix(pred_boxes, gt_boxes)
    tp = np.zeros(len(scores), dtype=bool)
    gt_score = np.full(len(gt_boxes), -np.inf)
    free = np.ones(len(gt_boxes), dtype=bool)
    for i in np.argsort(-scores, kind="stable"):
        if not free.any():
            break
        cand = np.where(free, iou[i], -1.0)
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            tp[i], free[j], gt_score[j] = True, False, scores[i]
    return ImageMatch(scores, tp, gt_score)


def counts_at(matches: Sequence[ImageMatch], thr: float) -> dict[str, int]:
    """TP, FP, FN over images, keeping predictions with score >= thr."""
    tp = fp = fn = 0
    for m in matches:
        keep = m.scores >= thr
        tp += int(np.sum(m.tp & keep))
        fp += int(np.sum(~m.tp & keep))
        fn += int(np.sum(m.gt_matched_score < thr))
    return {"tp": tp, "fp": fp, "fn": fn}


def prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """Precision, recall, F1; None where the denominator is 0."""
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else None
    return {"precision": p, "recall": r, "f1": f1}


def average_precision(matches: Sequence[ImageMatch]) -> float | None:
    """AP at the matching IoU: 101-point interpolated area under the PR envelope."""
    n_gt = sum(len(m.gt_matched_score) for m in matches)
    if n_gt == 0:
        return None
    scores = np.concatenate([m.scores for m in matches]) if matches else np.zeros(0)
    tp = np.concatenate([m.tp for m in matches]) if matches else np.zeros(0, bool)
    order = np.argsort(-scores, kind="stable")
    tpc = np.cumsum(tp[order])
    fpc = np.cumsum(~tp[order])
    recall = tpc / n_gt
    precision = tpc / np.maximum(tpc + fpc, 1)
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def height_bin(h416: float) -> str:
    for name, lo, hi in HEIGHT_BINS:
        if lo <= h416 < hi:
            return name
    raise ValueError(f"height {h416}")


def recall_by_height(
    matches: Sequence[ImageMatch], gt_heights_416: Sequence[np.ndarray], thr: float
) -> dict[str, dict[str, float | int | None]]:
    """Recall per box-height bin at 416 (ground-truth height), at threshold ``thr``."""
    out = {name: {"gt": 0, "tp": 0} for name, _, _ in HEIGHT_BINS}
    for m, heights in zip(matches, gt_heights_416, strict=True):
        for s, h in zip(m.gt_matched_score, heights, strict=True):
            b = out[height_bin(float(h))]
            b["gt"] += 1
            b["tp"] += int(s >= thr)
    for b in out.values():
        b["recall"] = b["tp"] / b["gt"] if b["gt"] else None
    return out


def sweep(matches: Sequence[ImageMatch], thresholds: Sequence[float]) -> list[dict]:
    """Counts, P/R/F1 and FP per image at each threshold."""
    n = max(len(matches), 1)
    rows = []
    for t in thresholds:
        c = counts_at(matches, t)
        rows.append({"conf": float(t), **c, **prf(**c), "fp_per_image": c["fp"] / n})
    return rows


def threshold_at_fp_budget(
    matches: Sequence[ImageMatch], fp_per_image: float, min_conf: float = 0.0
) -> dict[str, float | int | None]:
    """
    Lowest threshold (highest recall) with at most ``fp_per_image`` false positives per image.

    Candidates are the prediction scores themselves (>= ``min_conf``), so the
    choice is exact, not limited to a grid. Returns None for ``conf`` if no
    threshold meets the budget.
    """
    n = len(matches)
    none = {"conf": None, "tp": None, "fp": None, "fn": None, "recall": None}
    if not n:
        return none
    scores = np.concatenate([m.scores for m in matches])
    tp = np.concatenate([m.tp for m in matches])
    order = np.argsort(-scores, kind="stable")
    s, fp_cum = scores[order], np.cumsum(~tp[order])
    # last index of each run of equal scores: FP count with score >= that value
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1] if len(s) else np.zeros(0, int)
    ok = (fp_cum[last] / n <= fp_per_image) & (s[last] >= min_conf)
    if not ok.any():
        return none
    t = float(s[last][ok].min())  # FP falls as the threshold rises: lowest feasible
    c = counts_at(matches, t)
    return {"conf": t, **c, **prf(**c), "fp_per_image": c["fp"] / n}
