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

Measured truncation and occlusion
---------------------------------
:class:`SilhouetteRenderer` copies the survivor parts into a second PyBullet
client (nothing else in it) and renders each survivor alone with the same
camera, on a canvas extended past the frame where the survivor overhangs it
(:meth:`sim.camera.PinholeCamera.render_segmentation`). From that
:class:`Silhouette`:

- truncation = survivor-alone pixels outside the frame / all survivor-alone pixels;
- occlusion = share of the survivor-alone pixels inside the frame that are
  not visible in the real render (hidden by walls, clutter or other survivors).

Usage (CLI smoke test)::

    python -m sim.labels

Programmatic::

    from sim.labels import SilhouetteRenderer, compute_survivor_labels, save_yolo_labels

    with SilhouetteRenderer(camera, survivors, physics_client=cid) as silhouettes:
        out = camera.render(pose, physics_client=cid)
        labels = compute_survivor_labels(
            out.seg_mask, survivors, silhouettes=silhouettes.render(pose)
        )
    save_yolo_labels(labels, "data/labels/frame_000001.txt")
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

# PyBullet mask bitmask for object unique ID (lower 24 bits)
OBJECT_ID_MASK: int = 0xFFFFFF

# Default detection quality thresholds
# Render pixels, for compute_box_for_body_ids / compute_survivor_labels.
# label_frame uses LabelConfig thresholds at training resolution instead.
DEFAULT_MIN_VISIBLE_PIXELS: int = 15
DEFAULT_MIN_BOX_SIZE: float = 6.0  # pixels
DEFAULT_MAX_OCCLUSION: float = 0.90
DEFAULT_MAX_TRUNCATION: float = 0.85
DEFAULT_CLASS_ID: int = 0
DEFAULT_CLASS_NAME: str = "person"
DEFAULT_ARENA_CONFIG = Path("configs/arena.yaml")

# Extra canvas pixels past a projected overhang (silhouette edge rounding).
CANVAS_PAD_PX: int = 2
# Hidden survivor copies are moved this far up, well beyond the far clip plane.
PARK_OFFSET_M: float = 1000.0


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

    truncation_exact: bool = True
    """False when truncation_fraction is only a lower bound (see :class:`Silhouette`)."""

    contrast: float | None = None
    """Visibility contrast against the surrounding ring, grey levels 0-255
    (:func:`survivor_contrast`); None if not measured or no ring pixels."""

    hard: bool = False
    """Kept, but with contrast below ``labels.hard_contrast``: a hard positive."""

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
            "truncation_exact": self.truncation_exact,
            "contrast": None if self.contrast is None else round(self.contrast, 2),
            "hard": self.hard,
        }


@dataclass
class Silhouette:
    """
    A survivor rendered alone (no walls, clutter or other survivors) from one pose.

    Pixel counts use the frame's focal length and pixel grid; ``outside_pixels``
    come from the canvas margins around the frame.
    """

    in_frame: np.ndarray
    """Boolean (H, W) mask of the survivor-alone pixels inside the frame."""

    in_frame_pixels: int
    """Number of True pixels in ``in_frame``."""

    outside_pixels: int
    """Survivor-alone pixels on the canvas outside the frame."""

    exact: bool
    """False when part of the survivor may lie beyond the canvas (canvas capped by
    ``max_canvas_area_factor``) or behind the near plane; truncation is then a
    lower bound."""

    canvas_area_factor: float
    """Area of the rendered canvas divided by the frame area (1.0 = no margin)."""

    @property
    def total_pixels(self) -> int:
        return self.in_frame_pixels + self.outside_pixels

    @property
    def truncation_fraction(self) -> float:
        """Share of the survivor-alone pixels outside the frame."""
        total = self.total_pixels
        return self.outside_pixels / total if total > 0 else 0.0

    def occlusion_fraction(self, visible_mask: np.ndarray) -> float:
        """Share of the in-frame survivor-alone pixels not visible in the real render."""
        if self.in_frame_pixels == 0:
            return 0.0
        seen = int(np.count_nonzero(visible_mask & self.in_frame))
        return 1.0 - seen / self.in_frame_pixels


@dataclass(frozen=True)
class LabelConfig:
    """
    Label filter settings, from the ``labels`` section of ``configs/arena.yaml``.

    Every value there is an ASSUMPTION; see the comments in that file. Pixel
    thresholds are given at training resolution: the frame letterboxed to
    ``imgsz`` with scale ``imgsz / max(width, height)``. :meth:`thresholds`
    converts them to the render size.
    """

    imgsz: int = 416
    """Training image size (letterboxed, longest side)."""

    min_box_height_px: float = 10.0
    """Minimum visible box height at training resolution."""

    min_box_width_px: float = 6.0
    """Minimum visible box width at training resolution."""

    presence_min_area_px: float = 1.6
    """Visible survivor area (px^2 at training resolution) below which the survivor
    counts as absent from the frame: neither labelled nor dropped."""

    max_truncation: float = DEFAULT_MAX_TRUNCATION
    """Drop a label whose truncation fraction is above this."""

    max_occlusion: float = DEFAULT_MAX_OCCLUSION
    """Drop a label whose occlusion fraction is above this."""

    max_canvas_area_factor: float = 4.0
    """Largest silhouette canvas, as a multiple of the frame area."""

    contrast_ring_px: float = 3.0
    """Width of the ring around the survivor for :func:`survivor_contrast`
    (px at training resolution)."""

    min_contrast: float = 3.0
    """Drop a label whose contrast is below this (grey levels): essentially invisible."""

    hard_contrast: float = 8.0
    """Tag a kept label as hard when its contrast is below this (grey levels)."""

    def __post_init__(self) -> None:
        for name in ("max_truncation", "max_occlusion"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"labels.{name} must be in [0, 1], got {getattr(self, name)}")
        if self.max_canvas_area_factor < 1.0:
            raise ValueError(
                f"labels.max_canvas_area_factor must be >= 1, got {self.max_canvas_area_factor}"
            )
        if self.imgsz <= 0:
            raise ValueError(f"labels.imgsz must be > 0, got {self.imgsz}")
        for name in ("min_box_height_px", "min_box_width_px", "presence_min_area_px"):
            if getattr(self, name) < 0:
                raise ValueError(f"labels.{name} must be >= 0, got {getattr(self, name)}")
        if self.contrast_ring_px <= 0:
            raise ValueError(f"labels.contrast_ring_px must be > 0, got {self.contrast_ring_px}")
        if not 0.0 <= self.min_contrast <= self.hard_contrast <= 255.0:
            raise ValueError(
                "labels need 0 <= min_contrast <= hard_contrast <= 255, got "
                f"{self.min_contrast} and {self.hard_contrast}"
            )

    def scale(self, width: int, height: int) -> float:
        """Letterbox scale from a ``width x height`` render to training resolution."""
        return self.imgsz / float(max(width, height))

    def thresholds(self, width: int, height: int) -> RenderThresholds:
        """The training-resolution thresholds converted to a ``width x height`` render."""
        s = self.scale(width, height)
        return RenderThresholds(
            min_box_height=self.min_box_height_px / s,
            min_box_width=self.min_box_width_px / s,
            presence_min_pixels=self.presence_min_area_px / (s * s),
            ring_px=max(1, round(self.contrast_ring_px / s)),
        )

    @classmethod
    def from_arena_config(cls, path: str | Path = DEFAULT_ARENA_CONFIG) -> LabelConfig:
        """Read the ``labels`` section; missing keys keep the defaults, unknown keys raise."""
        from cv.config import load_config  # needs yaml, so imported only here

        section = load_config(path).to_dict().get("labels") or {}
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(section) - known)
        if unknown:
            raise ValueError(f"unknown keys in the labels section of {path}: {unknown}")
        return cls(**{k: type(getattr(cls, k))(v) for k, v in section.items()})


@dataclass(frozen=True)
class RenderThresholds:
    """:class:`LabelConfig` pixel thresholds in render pixels (see :meth:`LabelConfig.thresholds`)."""

    min_box_height: float
    min_box_width: float
    presence_min_pixels: float
    ring_px: int


# Reasons a present survivor's label is dropped (see label_frame).
DROP_BOX_TOO_SHORT = "box_too_short"
DROP_BOX_TOO_NARROW = "box_too_narrow"
DROP_TRUNCATED = "truncated"
DROP_TRUNCATION_UNBOUNDED = "truncation_unbounded"
DROP_OCCLUDED = "occluded"
DROP_LOW_CONTRAST = "low_contrast"
DROP_REASONS = (
    DROP_BOX_TOO_SHORT,
    DROP_BOX_TOO_NARROW,
    DROP_TRUNCATED,
    DROP_TRUNCATION_UNBOUNDED,
    DROP_OCCLUDED,
    DROP_LOW_CONTRAST,
)


@dataclass
class DroppedLabel:
    """A survivor present in the frame whose label was dropped."""

    label: SurvivorLabel
    """The label it would have had, with its measured values."""

    reasons: tuple[str, ...]
    """Every rule it failed, in :data:`DROP_REASONS` order."""


@dataclass
class FrameLabels:
    """Labels of one frame (:func:`label_frame`)."""

    labels: list[SurvivorLabel]
    """Kept labels."""

    dropped: list[DroppedLabel]
    """Survivors present in the frame whose label was dropped."""

    @property
    def ambiguous(self) -> bool:
        """A survivor is in the frame but unlabelled: the generator should skip the frame."""
        return bool(self.dropped)

    @property
    def ambiguous_reasons(self) -> tuple[str, ...]:
        """Every drop reason in the frame, in :data:`DROP_REASONS` order."""
        found = {r for d in self.dropped for r in d.reasons}
        return tuple(r for r in DROP_REASONS if r in found)


# --------------------------------------------------------------------------- #
# Visibility
# --------------------------------------------------------------------------- #
def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a (2 * radius + 1) square, separable, numpy only."""
    out = mask
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad)
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1, axis=axis)
        out = windows.any(axis=-1)
    return out


def survivor_contrast(
    image: np.ndarray,
    visible: np.ndarray,
    survivor_pixels: np.ndarray,
    ring_px: int,
) -> float | None:
    """
    How much a survivor stands out from its surroundings, in grey levels (0-255).

    The largest per-channel absolute difference between the mean of the
    survivor's visible pixels and the mean of a ring around them: the visible
    mask dilated by ``ring_px`` (square), minus every survivor pixel. Means
    make it robust to pixel noise; the per-channel maximum keeps colour-only
    contrast (for example red on grey of equal brightness).

    Parameters
    ----------
    image : np.ndarray
        The frame after augmentation, (H, W, C) or (H, W).
    visible : np.ndarray
        Boolean (H, W) mask of this survivor's visible pixels.
    survivor_pixels : np.ndarray
        Boolean (H, W) mask of all survivors' visible pixels (excluded from the ring).
    ring_px : int
        Ring width in render pixels.

    Returns
    -------
    float or None
        None if the survivor has no visible pixels or the ring is empty.
    """
    ys, xs = np.nonzero(visible)
    if len(ys) == 0:
        return None
    h, w = visible.shape
    y1, y2 = max(0, ys.min() - ring_px), min(h, ys.max() + 1 + ring_px)
    x1, x2 = max(0, xs.min() - ring_px), min(w, xs.max() + 1 + ring_px)
    inside = visible[y1:y2, x1:x2]
    ring = _dilate(inside, ring_px) & ~survivor_pixels[y1:y2, x1:x2]
    if not ring.any():
        return None
    crop = image[y1:y2, x1:x2].astype(np.float64)
    if crop.ndim == 2:
        crop = crop[:, :, None]
    diff = crop[inside].mean(axis=0) - crop[ring].mean(axis=0)
    return float(np.abs(diff).max())


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
        touches_border = xs.min() == 0 or xs.max() >= w - 1 or ys.min() == 0 or ys.max() >= h - 1
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
    silhouettes: Sequence[Silhouette | None] | None = None,
) -> list[SurvivorLabel]:
    """
    Extract YOLO survivor labels from a raw PyBullet segmentation mask.

    Handles either raw segmentation buffers (auto-decodes) or already-decoded
    masks. Accepts a sequence of :class:`sim.survivor.SurvivorInfo` objects,
    integer body IDs, or sequences of body IDs.

    Truncation and occlusion are measured only when ``silhouettes`` (from
    :meth:`SilhouetteRenderer.render` for the same pose, one per survivor) are
    given; without them both stay 0 and their filters cannot fire. A label
    whose truncation is only a lower bound (``Silhouette.exact`` False) is
    dropped, because the truncation filter cannot be checked.

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
    silhouettes : sequence of Silhouette or None, optional
        Survivor-alone renders, indexed like ``survivors``.

    Returns
    -------
    list[SurvivorLabel]
        Filtered list of visible survivors in YOLO-ready format.
    """
    # Auto-decode mask if raw PyBullet flags were used
    decoded = decode_segmentation_mask(seg_mask)
    if silhouettes is not None and len(silhouettes) != len(survivors):
        raise ValueError(f"{len(silhouettes)} silhouettes for {len(survivors)} survivors")

    labels: list[SurvivorLabel] = []
    for idx, survivor in enumerate(survivors):
        body_ids = _survivor_body_ids(survivor)
        measured = silhouettes is not None

        label = compute_box_for_body_ids(
            decoded_mask=decoded,
            body_ids=body_ids,
            image_width=image_width,
            image_height=image_height,
            min_visible_pixels=min_visible_pixels,
            min_box_size=min_box_size,
            # With silhouettes the filters are applied below, to measured values.
            max_occlusion=1.0 if measured else max_occlusion,
            max_truncation=1.0 if measured else max_truncation,
            survivor_id=idx,
        )
        if label is not None and measured:
            label = _apply_silhouette(
                label, decoded, body_ids, silhouettes[idx], max_occlusion, max_truncation
            )
        if label is not None:
            labels.append(label)

    return labels


def _survivor_body_ids(survivor: Any) -> list[int]:
    """Body IDs of a SurvivorInfo, a single body ID or a sequence of body IDs."""
    if hasattr(survivor, "body_ids"):
        return [int(b) for b in survivor.body_ids]
    if isinstance(survivor, (int, np.integer)):
        return [int(survivor)]
    if isinstance(survivor, (list, tuple, set)):
        return [int(b) for b in survivor]
    raise TypeError(f"Unsupported survivor type: {type(survivor)}")


def _apply_silhouette(
    label: SurvivorLabel,
    decoded: np.ndarray,
    body_ids: list[int],
    silhouette: Silhouette | None,
    max_occlusion: float,
    max_truncation: float,
) -> SurvivorLabel | None:
    """Set measured truncation and occlusion on a label; None if a filter rejects it."""
    _measure(label, np.isin(decoded, body_ids), silhouette)
    if label.occlusion_fraction > max_occlusion or label.truncation_fraction > max_truncation:
        return None
    if not label.truncation_exact:
        return None
    return label


def _measure(label: SurvivorLabel, visible: np.ndarray, silhouette: Silhouette | None) -> None:
    """Set a label's truncation and occlusion from its survivor-alone silhouette."""
    if silhouette is None:
        raise ValueError(
            f"survivor {label.survivor_id} is visible but has no silhouette "
            "(silhouettes must come from the same pose and survivors)"
        )
    label.total_expected_pixels = silhouette.in_frame_pixels
    label.truncation_fraction = silhouette.truncation_fraction
    label.occlusion_fraction = silhouette.occlusion_fraction(visible)
    label.truncation_exact = silhouette.exact


def label_frame(
    seg_mask: np.ndarray,
    survivors: Sequence[Any],
    silhouettes: Sequence[Silhouette | None],
    image: np.ndarray,
    config: LabelConfig | None = None,
) -> FrameLabels:
    """
    Labels of one rendered frame, with the dropped survivors and their reasons.

    A survivor is present when it has at least ``presence_min_area_px``
    visible pixels (training resolution, converted to the render size). Absent
    survivors are ignored. A present survivor's label is dropped, with every
    reason that applies, when its visible box is shorter than
    ``min_box_height_px`` or narrower than ``min_box_width_px`` (training
    resolution), its truncation exceeds ``max_truncation`` (``truncated``) or
    is only a lower bound not above it (``truncation_unbounded``), its
    occlusion exceeds ``max_occlusion``, or its :func:`survivor_contrast` in
    ``image`` is below ``min_contrast`` (``low_contrast``: essentially
    invisible). A kept label with contrast below ``hard_contrast`` is tagged
    ``hard``. A frame with any dropped label is ambiguous
    (:attr:`FrameLabels.ambiguous`).

    Parameters
    ----------
    seg_mask : np.ndarray
        Segmentation mask of the real render (raw or decoded).
    survivors : sequence of SurvivorInfo, int or sequence of ints
        Survivors in the scene.
    silhouettes : sequence of Silhouette or None
        :meth:`SilhouetteRenderer.render` for the same pose, one per survivor.
    image : np.ndarray
        The frame as it will be trained on, i.e. after augmentation (same
        size as ``seg_mask``).
    config : LabelConfig, optional
        Thresholds; defaults to :class:`LabelConfig` defaults.
    """
    cfg = config if config is not None else LabelConfig()
    if len(silhouettes) != len(survivors):
        raise ValueError(f"{len(silhouettes)} silhouettes for {len(survivors)} survivors")
    decoded = decode_segmentation_mask(seg_mask)
    h, w = decoded.shape[:2]
    if image.shape[:2] != (h, w):
        raise ValueError(f"image is {image.shape[:2]}, segmentation mask is {(h, w)}")
    th = cfg.thresholds(w, h)
    all_ids = [b for s in survivors for b in _survivor_body_ids(s)]
    survivor_pixels = np.isin(decoded, all_ids)

    result = FrameLabels(labels=[], dropped=[])
    for idx, survivor in enumerate(survivors):
        body_ids = _survivor_body_ids(survivor)
        visible = np.isin(decoded, body_ids)
        n_visible = int(np.count_nonzero(visible))
        if n_visible == 0 or n_visible < th.presence_min_pixels:
            continue
        label = compute_box_for_body_ids(
            decoded,
            body_ids,
            min_visible_pixels=1,
            min_box_size=0.0,
            max_occlusion=1.0,
            max_truncation=1.0,
            survivor_id=idx,
        )
        assert label is not None  # no filter can reject it with these arguments
        _measure(label, visible, silhouettes[idx])
        label.contrast = survivor_contrast(image, visible, survivor_pixels, th.ring_px)
        low_contrast = label.contrast is not None and label.contrast < cfg.min_contrast

        x1, y1, x2, y2 = label.bbox_xyxy
        failed = {
            DROP_BOX_TOO_SHORT: (y2 - y1) < th.min_box_height,
            DROP_BOX_TOO_NARROW: (x2 - x1) < th.min_box_width,
            DROP_TRUNCATED: label.truncation_fraction > cfg.max_truncation,
            DROP_TRUNCATION_UNBOUNDED: (
                not label.truncation_exact and label.truncation_fraction <= cfg.max_truncation
            ),
            DROP_OCCLUDED: label.occlusion_fraction > cfg.max_occlusion,
            DROP_LOW_CONTRAST: low_contrast,
        }
        reasons = tuple(r for r in DROP_REASONS if failed[r])
        if reasons:
            result.dropped.append(DroppedLabel(label=label, reasons=reasons))
        else:
            label.hard = label.contrast is not None and label.contrast < cfg.hard_contrast
            result.labels.append(label)
    return result


# --------------------------------------------------------------------------- #
# Survivor-alone silhouettes
# --------------------------------------------------------------------------- #
# Corner c of a box has local signs (x, y, z) from bits (c & 4, c & 2, c & 1),
# the corner order sim.camera.PinholeCamera.project_box expects.
_CORNER_SIGNS = np.array(
    [[1.0 if (c >> k) & 1 else -1.0 for k in (2, 1, 0)] for c in range(8)], dtype=np.float64
)
# Shape types SilhouetteRenderer copies (PyBullet GEOM_* names).
_COPIED_SHAPES = ("GEOM_SPHERE", "GEOM_BOX", "GEOM_CYLINDER", "GEOM_CAPSULE")
_SHAPE_NAMES = (*_COPIED_SHAPES, "GEOM_MESH", "GEOM_PLANE", "GEOM_HEIGHTFIELD")


class UnsupportedShapeError(ValueError):
    """A body has a shape that :class:`SilhouetteRenderer` cannot copy."""


def _rects_overlap(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class SilhouetteRenderer:
    """
    Renders each survivor alone, for measured truncation and occlusion.

    On creation the survivor parts are copied into a second PyBullet DIRECT
    client that holds nothing else. The copy is generic, from PyBullet shape
    data: visual and collision shape type, dimensions and local frame, the
    inertial frame and the base pose. The main scene is never modified.
    Supported: single-link bodies with at most one visual and one collision
    shape, each a sphere, box, cylinder or capsule. Anything else raises
    :class:`UnsupportedShapeError` naming the body and shape type.

    For a pose, survivors whose parts project into the frame are rendered with
    the camera's :meth:`~sim.camera.PinholeCamera.render_segmentation`, on a
    canvas extended on each side where a part's box overhangs the frame (the
    part boxes bound the silhouette, so it cannot reach past the canvas unless
    the canvas is capped). Survivors whose projected part boxes overlap are
    rendered in separate passes, so each one is seen alone.

    Create one per scene after the survivors are placed (they are assumed
    static) and close it when done, or use it as a context manager.

    Parameters
    ----------
    camera : sim.camera.PinholeCamera
        The camera that renders the real frames (same size and FOV).
    survivors : sequence of SurvivorInfo, int or sequence of ints
        Survivors in the main scene, as for :func:`compute_survivor_labels`.
    physics_client : int
        The main scene's PyBullet client.
    max_canvas_area_factor : float, optional
        Canvas area cap as a multiple of the frame area; defaults to
        ``labels.max_canvas_area_factor`` in ``arena_config_path``.
    """

    def __init__(
        self,
        camera: Any,
        survivors: Sequence[Any],
        physics_client: int,
        max_canvas_area_factor: float | None = None,
        arena_config_path: str | Path = DEFAULT_ARENA_CONFIG,
    ) -> None:
        import pybullet as p  # optional dependency: only this class needs it

        self._p = p
        self.camera = camera
        if max_canvas_area_factor is None:
            max_canvas_area_factor = LabelConfig.from_arena_config(
                arena_config_path
            ).max_canvas_area_factor
        if max_canvas_area_factor < 1.0:
            raise ValueError(f"max_canvas_area_factor must be >= 1, got {max_canvas_area_factor}")
        self.max_canvas_area_factor = float(max_canvas_area_factor)
        self.copy_client: int | None = p.connect(p.DIRECT)
        self.copy_ids: list[list[int]] = []
        """Copy body IDs per survivor, in the copy client."""
        self.last_canvas_area_factors: list[float] = []
        """Canvas area / frame area of each pass of the last :meth:`render`."""
        self._corners: list[list[np.ndarray]] = []
        self._poses: dict[int, tuple[Any, Any]] = {}
        self._parked: set[int] = set()
        try:
            for survivor in survivors:
                body_ids = _survivor_body_ids(survivor)
                if not body_ids:
                    raise ValueError("survivor has no body parts")
                ids: list[int] = []
                corners: list[np.ndarray] = []
                for bid in body_ids:
                    copy, part_corners = self._copy_body(bid, physics_client)
                    ids.append(copy)
                    corners.extend(part_corners)
                self.copy_ids.append(ids)
                self._corners.append(corners)
        except BaseException:
            self.close()
            raise

    # -- Lifecycle --

    def close(self) -> None:
        """Disconnect the copy client (safe to call twice)."""
        if self.copy_client is not None:
            self._p.disconnect(self.copy_client)
            self.copy_client = None

    def __enter__(self) -> SilhouetteRenderer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- Copying --

    def _shape_name(self, shape_type: int) -> str:
        for name in _SHAPE_NAMES:
            if getattr(self._p, name) == shape_type:
                return name
        return "unknown"

    def _shape_kwargs(
        self, body_id: int, kind: str, shape_type: int, dims: Sequence[float]
    ) -> dict[str, Any]:
        """createVisualShape / createCollisionShape size arguments from shape data dimensions."""
        p = self._p
        if shape_type == p.GEOM_SPHERE:
            return {"radius": dims[0]}
        if shape_type == p.GEOM_BOX:
            return {"halfExtents": [d / 2.0 for d in dims[:3]]}
        if shape_type in (p.GEOM_CAPSULE, p.GEOM_CYLINDER):
            # Shape data gives (length, radius); the length argument is named
            # "length" for visual shapes and "height" for collision shapes.
            return {"radius": dims[1], ("length" if kind == "visual" else "height"): dims[0]}
        raise UnsupportedShapeError(
            f"body {body_id} has {kind} shape type {self._shape_name(shape_type)} "
            f"({shape_type}); SilhouetteRenderer copies only {', '.join(_COPIED_SHAPES)}"
        )

    def _half_extents(self, shape_type: int, dims: Sequence[float]) -> np.ndarray:
        """Half extents of the box enclosing a visual shape, in its local frame."""
        p = self._p
        if shape_type == p.GEOM_SPHERE:
            r = dims[0]
            return np.array([r, r, r])
        if shape_type == p.GEOM_BOX:
            return np.asarray(dims[:3], dtype=np.float64) / 2.0
        length, r = dims[0], dims[1]  # capsule or cylinder, axis along local z
        cap = r if shape_type == p.GEOM_CAPSULE else 0.0
        return np.array([r, r, length / 2.0 + cap])

    def _copy_body(self, body_id: int, src: int) -> tuple[int, list[np.ndarray]]:
        """Copy one body into the copy client; return its ID and its visual box corners."""
        p = self._p
        dst = self.copy_client
        if p.getNumJoints(body_id, physicsClientId=src) > 0:
            raise UnsupportedShapeError(
                f"body {body_id} has links; SilhouetteRenderer copies single-link bodies only"
            )
        com_pos, com_orn = p.getBasePositionAndOrientation(body_id, physicsClientId=src)
        inertial_pos, inertial_orn = p.getDynamicsInfo(body_id, -1, physicsClientId=src)[3:5]
        # Base link frame = centre-of-mass frame composed with the inverse inertial frame.
        link_pos, link_orn = p.multiplyTransforms(
            com_pos, com_orn, *p.invertTransform(inertial_pos, inertial_orn)
        )
        visuals = [v for v in p.getVisualShapeData(body_id, physicsClientId=src) if v[1] == -1]
        collisions = p.getCollisionShapeData(body_id, -1, physicsClientId=src)
        for kind, shapes in (("visual", visuals), ("collision", collisions)):
            if len(shapes) > 1:
                raise UnsupportedShapeError(
                    f"body {body_id} has {len(shapes)} {kind} shapes; "
                    "SilhouetteRenderer copies one per body"
                )

        visual_index, corners = -1, []
        if visuals:
            shape_type, dims, _, local_pos, local_orn, rgba = visuals[0][2:8]
            visual_index = p.createVisualShape(
                shape_type,
                rgbaColor=rgba,
                visualFramePosition=local_pos,
                visualFrameOrientation=local_orn,
                physicsClientId=dst,
                **self._shape_kwargs(body_id, "visual", shape_type, dims),
            )
            # Visual frames are given relative to the link frame.
            pos, orn = p.multiplyTransforms(link_pos, link_orn, local_pos, local_orn)
            rot = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float64).reshape(3, 3)
            local = _CORNER_SIGNS * self._half_extents(shape_type, dims)
            corners.append(np.asarray(pos) + local @ rot.T)

        collision_index = -1
        if collisions:
            shape_type, dims, _, local_pos, local_orn = collisions[0][2:7]
            # Collision frames are given relative to the inertial frame.
            frame_pos, frame_orn = p.multiplyTransforms(
                inertial_pos, inertial_orn, local_pos, local_orn
            )
            collision_index = p.createCollisionShape(
                shape_type,
                collisionFramePosition=frame_pos,
                collisionFrameOrientation=frame_orn,
                physicsClientId=dst,
                **self._shape_kwargs(body_id, "collision", shape_type, dims),
            )

        copy = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_index,
            baseVisualShapeIndex=visual_index,
            basePosition=link_pos,
            baseOrientation=link_orn,
            baseInertialFramePosition=inertial_pos,
            baseInertialFrameOrientation=inertial_orn,
            physicsClientId=dst,
        )
        p.resetBasePositionAndOrientation(copy, com_pos, com_orn, physicsClientId=dst)
        self._poses[copy] = (com_pos, com_orn)
        return copy, corners

    # -- Rendering --

    def _needed_margins(self, rect: tuple[float, float, float, float]) -> tuple[int, ...]:
        """(left, top, right, bottom) canvas margins that contain ``rect``."""
        w, h = self.camera.width, self.camera.height
        x1, y1, x2, y2 = rect
        return (
            math.ceil(-x1) + CANVAS_PAD_PX if x1 < 0 else 0,
            math.ceil(-y1) + CANVAS_PAD_PX if y1 < 0 else 0,
            math.ceil(x2 - w) + CANVAS_PAD_PX if x2 > w else 0,
            math.ceil(y2 - h) + CANVAS_PAD_PX if y2 > h else 0,
        )

    def _capped_margins(self, margins: tuple[int, ...]) -> tuple[int, ...]:
        """Scale margins down so the canvas area is at most the cap."""
        w, h = self.camera.width, self.camera.height
        left, top, right, bottom = margins
        cap = self.max_canvas_area_factor * w * h
        if (w + left + right) * (h + top + bottom) <= cap:
            return margins
        # Solve (w + s * mx) * (h + s * my) = cap for the scale s in [0, 1].
        mx, my = left + right, top + bottom
        a, b, c = mx * my, w * my + h * mx, w * h - cap
        s = -c / b if a == 0 else (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        return tuple(int(math.floor(m * s)) for m in margins)

    def _show_only(self, group: list[int]) -> None:
        """Park every survivor copy outside ``group`` beyond the far clip plane."""
        p = self._p
        keep = {copy for i in group for copy in self.copy_ids[i]}
        for copy, (pos, orn) in self._poses.items():
            park = copy not in keep
            if park == (copy in self._parked):
                continue
            target = (pos[0], pos[1], pos[2] + PARK_OFFSET_M) if park else pos
            p.resetBasePositionAndOrientation(copy, target, orn, physicsClientId=self.copy_client)
            if park:
                self._parked.add(copy)
            else:
                self._parked.discard(copy)

    def render(self, pose: Any) -> list[Silhouette | None]:
        """
        Survivor-alone silhouettes from ``pose``, one per survivor.

        None for a survivor none of whose parts project into the frame (it
        cannot have visible pixels there).
        """
        from sim.camera import camera_axes

        if self.copy_client is None:
            raise RuntimeError("SilhouetteRenderer is closed")
        cam = self.camera
        w, h = cam.width, cam.height
        eye, fwd, _ = camera_axes(pose)
        eye = np.asarray(eye, dtype=np.float64)

        rects: list[tuple[float, float, float, float] | None] = []
        behind: list[bool] = []
        for corners in self._corners:
            rect = None
            crosses = False
            for box in corners:
                crosses |= bool((((box - eye) @ fwd) < cam.near_clip).any())
                r = cam.project_box(pose, box, clip=False)
                if r is not None:
                    rect = (
                        r
                        if rect is None
                        else (
                            min(rect[0], r[0]),
                            min(rect[1], r[1]),
                            max(rect[2], r[2]),
                            max(rect[3], r[3]),
                        )
                    )
            if rect is not None and (rect[2] <= 0 or rect[0] >= w or rect[3] <= 0 or rect[1] >= h):
                rect = None
            rects.append(rect)
            behind.append(crosses)

        groups: list[list[int]] = []
        for i, rect in enumerate(rects):
            if rect is None:
                continue
            for group in groups:
                if not any(_rects_overlap(rect, rects[j]) for j in group):
                    group.append(i)
                    break
            else:
                groups.append([i])

        out: list[Silhouette | None] = [None] * len(rects)
        self.last_canvas_area_factors = []
        for group in groups:
            needed = {i: self._needed_margins(rects[i]) for i in group}
            margins = self._capped_margins(
                tuple(max(needed[i][k] for i in group) for k in range(4))
            )
            self._show_only(group)
            seg = cam.render_segmentation(pose, physics_client=self.copy_client, margins=margins)
            ids = np.where(seg >= 0, seg & OBJECT_ID_MASK, -1)
            factor = seg.size / float(w * h)
            self.last_canvas_area_factors.append(factor)
            left, top = margins[0], margins[1]
            for i in group:
                mask = np.isin(ids, self.copy_ids[i])
                in_frame = mask[top : top + h, left : left + w]
                n_in = int(np.count_nonzero(in_frame))
                fits = all(n <= m for n, m in zip(needed[i], margins, strict=True))
                out[i] = Silhouette(
                    in_frame=in_frame,
                    in_frame_pixels=n_in,
                    outside_pixels=int(np.count_nonzero(mask)) - n_in,
                    exact=fits and not behind[i],
                    canvas_area_factor=factor,
                )
        return out


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
    print("   Survivor label created:")
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
