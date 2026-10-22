"""
sim/labels.py — Ground-truth YOLO label generator from PyBullet segmentation masks
===================================================================================

Computes 2D bounding boxes and normalised YOLO-format training annotations
(class 0: "person", normalised cx cy w h) from PyBullet segmentation masks.

PyBullet Segmentation Mask Encoding
-----------------------------------
When ``flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX`` is used in
``p.getCameraImage``, each pixel encodes::

    pixel_value = (link_index << 24) | object_unique_id

The lower 24 bits (bits 0..23) hold the PyBullet body unique ID:
    object_id = mask & 0xFFFFFF (0x00FFFFFF)

Background pixels that hit no object are represented by negative values
(typically -1).

Filtering & Quality Rules
-------------------------
1. **Minimum visible pixels** (``min_visible_pixels``):
   If the number of visible pixels for a survivor is below this threshold,
   the detection is considered unrecognisable and omitted.
2. **Minimum bounding box size** (``min_box_size``):
   If the clipped box width or height in pixels is below this threshold,
   the box is filtered out.
3. **Clipping & Truncation**:
   Bounding box coordinates are clipped to image boundaries [0, W] and [0, H].
   Truncation fraction represents the proportion of the object's extent cut off
   by the frame border.
4. **Occlusion tracking**:
   When total expected pixels or unoccluded references are available,
   the occlusion fraction (1 - visible / expected) is recorded.
   Optional ``max_occlusion`` and ``max_truncation`` filters can reject
   excessively obscured targets.

Usage (CLI smoke test)::

    python -m sim.labels

Programmatic::

    from sim.labels import compute_survivor_labels, save_yolo_labels

    labels = compute_survivor_labels(seg_mask, survivors, img_w=1280, img_h=720)
    save_yolo_labels(labels, "data/labels/frame_000001.txt")
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# PyBullet mask bitmask for object unique ID (lower 24 bits)
OBJECT_ID_MASK: int = 0xFFFFFF

# Default detection quality thresholds
DEFAULT_MIN_VISIBLE_PIXELS: int = 15
DEFAULT_MIN_BOX_SIZE: float = 6.0  # pixels
DEFAULT_MAX_OCCLUSION: float = 0.90
DEFAULT_MAX_TRUNCATION: float = 0.85
DEFAULT_CLASS_ID: int = 0
DEFAULT_CLASS_NAME: str = "person"


# --------------------------------------------------------------------------- #
# Mask decoding
# --------------------------------------------------------------------------- #

def decode_object_id(pixel_val: int) -> int:
    """
    Decode a single segmentation mask pixel into a PyBullet body unique ID.

    Parameters
    ----------
    pixel_val : int
        Raw integer pixel from PyBullet segmentation buffer.

    Returns
    -------
    int
        Decoded body unique ID (>= 0), or -1 if the pixel is background.
    """
    if pixel_val < 0:
        return -1
    return int(pixel_val & OBJECT_ID_MASK)


def decode_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    """
    Decode a full PyBullet segmentation mask array into body unique IDs.

    When ``p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX`` is enabled, the
    lower 24 bits contain the body unique ID, and the upper 8 bits contain
    the link index. Negative values (background) are mapped to -1.

    Parameters
    ----------
    mask : np.ndarray
        2D array of shape (H, W) with integer pixel values.

    Returns
    -------
    np.ndarray
        2D int32 array of shape (H, W) where each pixel is the body ID or -1.
    """
    decoded = np.where(mask >= 0, mask & OBJECT_ID_MASK, -1)
    return decoded.astype(np.int32)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class SurvivorLabel:
    """
    Ground-truth bounding box and metadata for a survivor in a single frame.
    """

    survivor_id: int
    """Unique identifier or index of the survivor."""

    class_id: int = DEFAULT_CLASS_ID
    """Target class integer (0 for 'person')."""

    class_name: str = DEFAULT_CLASS_NAME
    """Human-readable class label."""

    bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Clipped pixel coordinates (x_min, y_min, x_max, y_max)."""

    yolo_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Normalised YOLO coordinates (center_x, center_y, width, height) in [0, 1]."""

    visible_pixels: int = 0
    """Number of visible segmentation pixels belonging to this survivor."""

    total_expected_pixels: int | None = None
    """Estimated unoccluded visible pixel count, if available."""

    occlusion_fraction: float = 0.0
    """Fraction in [0.0, 1.0] representing how much of the target is occluded."""

    truncation_fraction: float = 0.0
    """Fraction in [0.0, 1.0] representing how much of the target is cut off by borders."""

    image_width: int = 1280
    """Image width in pixels."""

    image_height: int = 720
    """Image height in pixels."""

    def to_yolo_line(self, precision: int = 6) -> str:
        """
        Format as a standard YOLO text line: ``<class_id> <cx> <cy> <w> <h>``.

        Parameters
        ----------
        precision : int
            Number of decimal places for normalised floats (default: 6).

        Returns
        -------
        str
            Single space-delimited annotation line without newline.
        """
        cx, cy, w, h = self.yolo_bbox
        return f"{self.class_id} {cx:.{precision}f} {cy:.{precision}f} {w:.{precision}f} {h:.{precision}f}"

    def to_dict(self) -> dict[str, Any]:
        """Convert label to serialisable dictionary matching project conventions."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return {
            "survivor_id": self.survivor_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "bbox_center": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
            "yolo_bbox": [round(v, 6) for v in self.yolo_bbox],
            "visible_pixels": self.visible_pixels,
            "occlusion_fraction": round(self.occlusion_fraction, 4),
            "truncation_fraction": round(self.truncation_fraction, 4),
        }


# --------------------------------------------------------------------------- #
# Bounding box computation from segmentation mask
# --------------------------------------------------------------------------- #

def compute_box_for_body_ids(
    decoded_mask: np.ndarray,
    body_ids: int | Sequence[int] | set[int],
    image_width: int | None = None,
    image_height: int | None = None,
    min_visible_pixels: int = DEFAULT_MIN_VISIBLE_PIXELS,
    min_box_size: float = DEFAULT_MIN_BOX_SIZE,
    total_expected_pixels: int | None = None,
    unclipped_bbox: tuple[float, float, float, float] | None = None,
    max_occlusion: float = DEFAULT_MAX_OCCLUSION,
    max_truncation: float = DEFAULT_MAX_TRUNCATION,
    survivor_id: int = 0,
    class_id: int = DEFAULT_CLASS_ID,
    class_name: str = DEFAULT_CLASS_NAME,
) -> SurvivorLabel | None:
    """
    Compute clipped bounding box and normalised YOLO coordinates for given body ID(s).

    Parameters
    ----------
    decoded_mask : np.ndarray
        2D array of decoded body unique IDs (from ``decode_segmentation_mask``).
    body_ids : int or sequence of ints
        PyBullet body unique ID(s) belonging to the target survivor.
    image_width : int, optional
        Image width. If None, inferred from ``decoded_mask.shape[1]``.
    image_height : int, optional
        Image height. If None, inferred from ``decoded_mask.shape[0]``.
    min_visible_pixels : int
        Minimum number of visible pixels required to generate a label.
    min_box_size : float
        Minimum box width and height in pixels.
    total_expected_pixels : int, optional
        Expected pixel count if unoccluded. Used to calculate occlusion fraction.
    unclipped_bbox : tuple of 4 floats, optional
        Ground-truth unclipped bounding box (e.g. from 3D projection) to calculate
        exact truncation fraction.
    max_occlusion : float
        Maximum allowable occlusion fraction before discarding box.
    max_truncation : float
        Maximum allowable truncation fraction before discarding box.
    survivor_id : int
        ID for tracking this survivor.
    class_id : int
        YOLO class ID (default: 0).
    class_name : str
        Class name (default: "person").

    Returns
    -------
    SurvivorLabel or None
        Generated label if survivor passes visibility and size criteria; else None.
    """
    h_img, w_img = decoded_mask.shape[:2]
    w = image_width if image_width is not None else w_img
    h = image_height if image_height is not None else h_img

    if isinstance(body_ids, (int, np.integer)):
        target_ids = {int(body_ids)}
    else:
        target_ids = {int(bid) for bid in body_ids}

    # Locate pixels matching target body IDs
    if len(target_ids) == 1:
        bid = next(iter(target_ids))
        ys, xs = np.where(decoded_mask == bid)
    else:
        ys, xs = np.where(np.isin(decoded_mask, list(target_ids)))

    visible_pixels = int(len(xs))

    # Rule 1: Minimum visible pixels
    if visible_pixels < min_visible_pixels:
        return None

    # Compute bounding box from visible pixels
    # In discrete pixel indexing: col indices span [xs.min(), xs.max()],
    # continuous box extent spans [xs.min(), xs.max() + 1.0].
    raw_x1 = float(xs.min())
    raw_y1 = float(ys.min())
    raw_x2 = float(xs.max() + 1.0)
    raw_y2 = float(ys.max() + 1.0)

    # Use unclipped bbox if provided, else use raw visible bbox
    if unclipped_bbox is not None:
        u_x1, u_y1, u_x2, u_y2 = unclipped_bbox
    else:
        u_x1, u_y1, u_x2, u_y2 = raw_x1, raw_y1, raw_x2, raw_y2

    # Rule 2: Clip to image boundaries [0, w] and [0, h]
    clip_x1 = max(0.0, min(float(w), u_x1))
    clip_y1 = max(0.0, min(float(h), u_y1))
    clip_x2 = max(0.0, min(float(w), u_x2))
    clip_y2 = max(0.0, min(float(h), u_y2))

    box_w = max(0.0, clip_x2 - clip_x1)
    box_h = max(0.0, clip_y2 - clip_y1)

    # Rule 3: Minimum box size
    if box_w < min_box_size or box_h < min_box_size:
        return None

    # Compute truncation fraction
    # Truncation = proportion of the unclipped box extending beyond frame boundaries.
    unclipped_area = max(0.0, u_x2 - u_x1) * max(0.0, u_y2 - u_y1)
    clipped_area = box_w * box_h
    if unclipped_area > 1e-6:
        truncation_fraction = max(0.0, min(1.0, 1.0 - (clipped_area / unclipped_area)))
    else:
        truncation_fraction = 0.0

    # If unclipped_bbox wasn't provided, detect if visible pixels touch border
    if unclipped_bbox is None:
        touches_border = (
            xs.min() == 0 or xs.max() >= w - 1 or ys.min() == 0 or ys.max() >= h - 1
        )
        if touches_border and total_expected_pixels is not None and total_expected_pixels > 0:
            missing_frac = max(0.0, 1.0 - visible_pixels / total_expected_pixels)
            # Portion of missing pixels attributed to boundary cutoff
            truncation_fraction = min(1.0, missing_frac)

    # Rule 4: Compute occlusion fraction
    # Occlusion = proportion of expected pixels hidden behind other objects in-frame.
    if total_expected_pixels is not None and total_expected_pixels > 0:
        total_missing_fraction = max(0.0, min(1.0, 1.0 - (visible_pixels / total_expected_pixels)))
        # Occlusion is the portion not explained by border truncation
        occlusion_fraction = max(0.0, min(1.0, total_missing_fraction - truncation_fraction))
    else:
        occlusion_fraction = 0.0

    # Rule 5: Reject if exceeded max occlusion or max truncation
    if occlusion_fraction > max_occlusion:
        return None
    if truncation_fraction > max_truncation:
        return None

    # Normalised YOLO format: (center_x, center_y, width, height) in [0, 1]
    norm_cx = (clip_x1 + clip_x2) / (2.0 * w)
    norm_cy = (clip_y1 + clip_y2) / (2.0 * h)
    norm_w = box_w / float(w)
    norm_h = box_h / float(h)

    # Clamp normalised values to [0.0, 1.0] for safety
    norm_cx = max(0.0, min(1.0, norm_cx))
    norm_cy = max(0.0, min(1.0, norm_cy))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return SurvivorLabel(
        survivor_id=survivor_id,
        class_id=class_id,
        class_name=class_name,
        bbox_xyxy=(clip_x1, clip_y1, clip_x2, clip_y2),
        yolo_bbox=(norm_cx, norm_cy, norm_w, norm_h),
        visible_pixels=visible_pixels,
        total_expected_pixels=total_expected_pixels,
        occlusion_fraction=occlusion_fraction,
        truncation_fraction=truncation_fraction,
        image_width=w,
        image_height=h,
    )


def compute_survivor_labels(
    seg_mask: np.ndarray,
    survivors: Sequence[Any],
    image_width: int | None = None,
    image_height: int | None = None,
    min_visible_pixels: int = DEFAULT_MIN_VISIBLE_PIXELS,
    min_box_size: float = DEFAULT_MIN_BOX_SIZE,
    max_occlusion: float = DEFAULT_MAX_OCCLUSION,
    max_truncation: float = DEFAULT_MAX_TRUNCATION,
) -> list[SurvivorLabel]:
    """
    Extract YOLO survivor labels from a raw PyBullet segmentation mask.

    Handles either raw segmentation buffers (auto-decodes) or already-decoded
    masks. Accepts a sequence of :class:`sim.survivor.SurvivorInfo` objects,
    integer body IDs, or sequences of body IDs.

    Parameters
    ----------
    seg_mask : np.ndarray
        2D segmentation mask array from PyBullet camera rendering.
    survivors : sequence of SurvivorInfo or int or sequence of ints
        Survivors present in the simulation scene.
    image_width : int, optional
        Image width. Defaults to mask width.
    image_height : int, optional
        Image height. Defaults to mask height.
    min_visible_pixels : int
        Minimum visible pixel count threshold.
    min_box_size : float
        Minimum box dimension in pixels.
    max_occlusion : float
        Maximum allowed occlusion fraction before filtering out.
    max_truncation : float
        Maximum allowed truncation fraction before filtering out.

    Returns
    -------
    list[SurvivorLabel]
        Filtered list of visible survivors in YOLO-ready format.
    """
    # Auto-decode mask if raw PyBullet flags were used
    decoded = decode_segmentation_mask(seg_mask)

    labels: list[SurvivorLabel] = []
    for idx, survivor in enumerate(survivors):
        # Extract body IDs from SurvivorInfo or raw sequence/int
        if hasattr(survivor, "body_ids"):
            body_ids = survivor.body_ids
        elif isinstance(survivor, (int, np.integer)):
            body_ids = [int(survivor)]
        elif isinstance(survivor, (list, tuple, set)):
            body_ids = list(survivor)
        else:
            raise TypeError(f"Unsupported survivor type: {type(survivor)}")

        label = compute_box_for_body_ids(
            decoded_mask=decoded,
            body_ids=body_ids,
            image_width=image_width,
            image_height=image_height,
            min_visible_pixels=min_visible_pixels,
            min_box_size=min_box_size,
            max_occlusion=max_occlusion,
            max_truncation=max_truncation,
            survivor_id=idx,
        )
        if label is not None:
            labels.append(label)

    return labels


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #

def save_yolo_labels(
    labels: Sequence[SurvivorLabel],
    output_path: str | Path,
    precision: int = 6,
) -> Path:
    """
    Write survivor labels to a YOLO-formatted text file.

    Parameters
    ----------
    labels : sequence of SurvivorLabel
        Labels to write.
    output_path : str or Path
        Destination text file (e.g. ``data/labels/frame_000001.txt``).
    precision : int
        Number of decimal digits for normalised coordinates.

    Returns
    -------
    Path
        Resolved path to the written text file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [label.to_yolo_line(precision=precision) for label in labels]
    content = "\n".join(lines)
    if content:
        content += "\n"

    out.write_text(content, encoding="utf-8")
    return out


def load_yolo_labels(label_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    """
    Load YOLO format labels from a text file.

    Returns
    -------
    list of (class_id, cx, cy, w, h) tuples.
    """
    p_path = Path(label_path)
    if not p_path.is_file():
        raise FileNotFoundError(f"Label file not found: {p_path}")

    results: list[tuple[int, float, float, float, float]] = []
    text = p_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
        results.append((cid, cx, cy, w, h))

    return results


# --------------------------------------------------------------------------- #
# CLI Smoke Test
# --------------------------------------------------------------------------- #

def _main() -> None:
    """Standalone smoke test demonstrating decoding and YOLO label calculation."""
    print("=== sim.labels Smoke Test ===\n")

    # 1. Test decoding logic
    print("1. Testing segmentation mask decoding:")
    body_id = 7
    link_idx = 3
    encoded_pixel = (link_idx << 24) | body_id
    decoded = decode_object_id(encoded_pixel)
    print(f"   Encoded: 0x{encoded_pixel:08X} -> Decoded body ID: {decoded}")
    assert decoded == body_id, f"Decoding failed: got {decoded}, expected {body_id}"
    print("   [OK] Bitmask decoding verified.")

    # 2. Synthetic mask test
    print("\n2. Testing YOLO bounding box computation from synthetic mask:")
    w, h = 640, 480
    synthetic_mask = np.full((h, w), -1, dtype=np.int32)

    # Place survivor body #5 in a 40x80 box at [100..140, 200..280]
    # Encode with linkIndex 2
    encoded_val = (2 << 24) | 5
    synthetic_mask[200:280, 100:140] = encoded_val

    labels = compute_survivor_labels(
        seg_mask=synthetic_mask,
        survivors=[5],
        image_width=w,
        image_height=h,
    )
    assert len(labels) == 1, f"Expected 1 label, got {len(labels)}"
    lbl = labels[0]
    print(f"   Survivor label created:")
    print(f"   - Bounding box (pixels): {lbl.bbox_xyxy}")
    print(f"   - YOLO normalised: {lbl.yolo_bbox}")
    print(f"   - YOLO text line: '{lbl.to_yolo_line()}'")
    print(f"   - Visible pixels: {lbl.visible_pixels}")

    # Verify expected coordinates
    expected_cx = 120.0 / w
    expected_cy = 240.0 / h
    expected_w = 40.0 / w
    expected_h = 80.0 / h
    assert abs(lbl.yolo_bbox[0] - expected_cx) < 1e-5
    assert abs(lbl.yolo_bbox[1] - expected_cy) < 1e-5
    assert abs(lbl.yolo_bbox[2] - expected_w) < 1e-5
    assert abs(lbl.yolo_bbox[3] - expected_h) < 1e-5
    print("   [OK] Synthetic mask bounding box matches expected dimensions exactly.")

    # 3. File I/O test
    test_out = Path("outputs/scratch/test_yolo_label.txt")
    save_yolo_labels(labels, test_out)
    loaded = load_yolo_labels(test_out)
    assert len(loaded) == 1
    assert loaded[0][0] == 0  # class 0
    print(f"\n3. Round-trip file save and load verified -> {test_out}")
    print("\n=== sim.labels Smoke Test Complete ===")


if __name__ == "__main__":
    _main()
