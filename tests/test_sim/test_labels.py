"""
tests/test_sim/test_labels.py — Unit tests for sim.labels
===========================================================

Validates:
1. PyBullet segmentation mask bitmask decoding (object_id = mask & 0xFFFFFF).
2. Minimum visible pixels filtering rule.
3. Minimum box size filtering rule.
4. Bounding box clipping to image frame boundaries.
5. Normalised YOLO format output (class 0, cx cy w h in [0, 1]).
6. Occlusion and truncation fraction computation.
7. Multi-body survivor handling (grouping body IDs into a single bounding box).
8. Round-trip label file save and load.
9. Training-resolution thresholds converted to the render size (1280x720, 640x360).
10. label_frame: kept and dropped labels with reasons, from synthetic silhouettes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sim.labels import (
    DROP_BOX_TOO_NARROW,
    DROP_BOX_TOO_SHORT,
    DROP_OCCLUDED,
    DROP_TRUNCATED,
    DROP_TRUNCATION_UNBOUNDED,
    LabelConfig,
    Silhouette,
    SurvivorLabel,
    compute_box_for_body_ids,
    compute_survivor_labels,
    decode_object_id,
    decode_segmentation_mask,
    label_frame,
    load_yolo_labels,
    save_yolo_labels,
)


# --------------------------------------------------------------------------- #
# 1. Mask Decoding Tests
# --------------------------------------------------------------------------- #
class TestMaskDecoding:
    """Verify PyBullet segmentation mask decoding into object unique IDs."""

    def test_single_pixel_decoding(self) -> None:
        # Lower 24 bits: body_id = 42
        # Upper 8 bits: link_index = 5
        raw_val = (5 << 24) | 42
        decoded = decode_object_id(raw_val)
        assert decoded == 42

    def test_zero_link_index(self) -> None:
        # Base link or linkIndex = 0
        raw_val = (0 << 24) | 17
        decoded = decode_object_id(raw_val)
        assert decoded == 17

    def test_high_link_index(self) -> None:
        # Max 8-bit linkIndex = 255
        raw_val = (255 << 24) | 999
        decoded = decode_object_id(raw_val)
        assert decoded == 999

    def test_negative_background_pixel(self) -> None:
        # PyBullet background / no-hit is typically -1
        assert decode_object_id(-1) == -1
        assert decode_object_id(-100) == -1

    def test_array_decoding_shape_and_values(self) -> None:
        # 3x3 test mask
        raw_mask = np.array(
            [
                [-1, (1 << 24) | 3, (2 << 24) | 3],
                [(3 << 24) | 5, (0 << 24) | 5, -1],
                [(12 << 24) | 8, -1, (1 << 24) | 8],
            ],
            dtype=np.int32,
        )

        decoded = decode_segmentation_mask(raw_mask)
        expected = np.array(
            [
                [-1, 3, 3],
                [5, 5, -1],
                [8, -1, 8],
            ],
            dtype=np.int32,
        )

        np.testing.assert_array_equal(decoded, expected)


# --------------------------------------------------------------------------- #
# 2. YOLO Bounding Box & Format Tests
# --------------------------------------------------------------------------- #
class TestYOLOBoxComputation:
    """Verify bounding box calculation and YOLO coordinate normalisation."""

    @pytest.fixture
    def empty_mask(self) -> np.ndarray:
        return np.full((720, 1280), -1, dtype=np.int32)

    def test_known_box_coordinates(self, empty_mask: np.ndarray) -> None:
        # Place survivor body #1 in pixel region x: [200, 300), y: [100, 250)
        # width = 100, height = 150
        empty_mask[100:250, 200:300] = 1

        label = compute_box_for_body_ids(
            decoded_mask=empty_mask,
            body_ids=1,
            image_width=1280,
            image_height=720,
        )

        assert label is not None
        assert label.class_id == 0
        assert label.class_name == "person"
        assert label.visible_pixels == 100 * 150

        # Pixel coords: [x1, y1, x2, y2]
        x1, y1, x2, y2 = label.bbox_xyxy
        assert x1 == pytest.approx(200.0)
        assert y1 == pytest.approx(100.0)
        assert x2 == pytest.approx(300.0)
        assert y2 == pytest.approx(250.0)

        # Normalised YOLO coords: cx, cy, w, h
        cx, cy, w, h = label.yolo_bbox
        assert cx == pytest.approx(250.0 / 1280.0)
        assert cy == pytest.approx(175.0 / 720.0)
        assert w == pytest.approx(100.0 / 1280.0)
        assert h == pytest.approx(150.0 / 720.0)

    def test_to_yolo_line_formatting(self) -> None:
        label = SurvivorLabel(
            survivor_id=0,
            class_id=0,
            bbox_xyxy=(100.0, 100.0, 200.0, 200.0),
            yolo_bbox=(0.5, 0.4, 0.2, 0.3),
            visible_pixels=500,
        )

        line = label.to_yolo_line(precision=6)
        assert line == "0 0.500000 0.400000 0.200000 0.300000"

        # Verify parsing line back matches values
        parts = line.split()
        assert int(parts[0]) == 0
        assert [float(p) for p in parts[1:]] == [0.5, 0.4, 0.2, 0.3]


# --------------------------------------------------------------------------- #
# 3. Filtering Rules Tests
# --------------------------------------------------------------------------- #
class TestFilteringRules:
    """Verify minimum visible pixels and minimum box size rules."""

    def test_min_visible_pixels_rejected(self) -> None:
        mask = np.full((100, 100), -1, dtype=np.int32)
        # Place only 5 pixels
        mask[10:15, 10] = 1

        label = compute_box_for_body_ids(
            decoded_mask=mask,
            body_ids=1,
            min_visible_pixels=10,  # requires >= 10
        )
        assert label is None, "Should reject box with visible_pixels < min_visible_pixels"

    def test_min_visible_pixels_accepted(self) -> None:
        mask = np.full((100, 100), -1, dtype=np.int32)
        # Place 15 pixels in a 3x5 block
        mask[10:15, 10:13] = 1

        label = compute_box_for_body_ids(
            decoded_mask=mask,
            body_ids=1,
            min_visible_pixels=10,
            min_box_size=2.0,
        )
        assert label is not None
        assert label.visible_pixels == 15

    def test_min_box_size_rejected(self) -> None:
        mask = np.full((100, 100), -1, dtype=np.int32)
        # 1 pixel wide, 20 pixels tall (box_w = 1.0 < min_box_size=5.0)
        mask[10:30, 20] = 1

        label = compute_box_for_body_ids(
            decoded_mask=mask,
            body_ids=1,
            min_visible_pixels=15,
            min_box_size=5.0,
        )
        assert label is None, "Should reject box where width < min_box_size"


# --------------------------------------------------------------------------- #
# 4. Clipping & Truncation Tests
# --------------------------------------------------------------------------- #
class TestClippingAndTruncation:
    """Verify coordinate clipping and truncation fraction computation."""

    def test_box_clipping_to_image_boundaries(self) -> None:
        mask = np.full((100, 100), -1, dtype=np.int32)
        # Place pixels touching the top-left edge
        mask[0:20, 0:30] = 1

        # Simulate unclipped box that extended outside [-10, -10, 30, 20]
        label = compute_box_for_body_ids(
            decoded_mask=mask,
            body_ids=1,
            image_width=100,
            image_height=100,
            unclipped_bbox=(-10.0, -10.0, 30.0, 20.0),
        )

        assert label is not None
        x1, y1, x2, y2 = label.bbox_xyxy
        # Clipped coordinates must be >= 0
        assert x1 == 0.0
        assert y1 == 0.0
        assert x2 == 30.0
        assert y2 == 20.0

        # Unclipped area was 40 * 30 = 1200
        # Clipped area is 30 * 20 = 600
        # Truncation fraction should be 1 - (600 / 1200) = 0.5
        assert label.truncation_fraction == pytest.approx(0.5, abs=0.01)

    def test_border_touching_truncation_detection(self) -> None:
        mask = np.full((100, 100), -1, dtype=np.int32)
        # Visible pixels touch bottom border (y=99)
        mask[80:100, 40:60] = 1

        label = compute_box_for_body_ids(
            decoded_mask=mask,
            body_ids=1,
            image_width=100,
            image_height=100,
            total_expected_pixels=800,  # 800 expected, only 400 visible
        )

        assert label is not None
        assert label.truncation_fraction > 0.0

    def test_max_truncation_filter(self) -> None:
        mask = np.full((100, 100), -1, dtype=np.int32)
        mask[0:5, 0:20] = 1

        # Unclipped box was 100x100 = 10000, clipped is 5x20 = 100 -> truncation = 0.99
        label = compute_box_for_body_ids(
            decoded_mask=mask,
            body_ids=1,
            unclipped_bbox=(-40.0, -95.0, 60.0, 5.0),
            max_truncation=0.50,  # rejects if > 50% truncated
        )
        assert label is None, "Should filter out highly truncated box"


# --------------------------------------------------------------------------- #
# 5. Occlusion Fraction Tests
# --------------------------------------------------------------------------- #
class TestOcclusion:
    """Verify occlusion calculation and max occlusion threshold filtering."""

    def test_occlusion_fraction_calculated(self) -> None:
        mask = np.full((200, 200), -1, dtype=np.int32)
        # Place 200 visible pixels inside the image (not touching border)
        mask[50:60, 50:70] = 2

        # 1000 expected pixels, 200 visible -> 80% occluded
        label = compute_box_for_body_ids(
            decoded_mask=mask,
            body_ids=2,
            total_expected_pixels=1000,
            max_occlusion=0.95,
        )

        assert label is not None
        assert label.occlusion_fraction == pytest.approx(0.80, abs=0.01)
        assert label.truncation_fraction == pytest.approx(0.0)

    def test_max_occlusion_filter_rejects(self) -> None:
        mask = np.full((200, 200), -1, dtype=np.int32)
        mask[50:55, 50:60] = 2  # 50 visible pixels

        label = compute_box_for_body_ids(
            decoded_mask=mask,
            body_ids=2,
            total_expected_pixels=1000,  # 95% occluded
            max_occlusion=0.80,  # threshold is 80%
        )
        assert label is None, "Should reject box exceeding max occlusion"


# --------------------------------------------------------------------------- #
# 6. Multi-Body Survivor Grouping Tests
# --------------------------------------------------------------------------- #
class TestMultiBodySurvivor:
    """Verify that multiple PyBullet body IDs composing one survivor are grouped."""

    @dataclass
    class MockSurvivor:
        body_ids: list[int]

    def test_multi_body_merges_into_single_box(self) -> None:
        mask = np.full((300, 300), -1, dtype=np.int32)

        # Body 10 (torso): [100..150, 100..130]
        mask[100:150, 100:130] = 10
        # Body 11 (head): [80..100, 105..125]
        mask[80:100, 105:125] = 11
        # Body 12 (legs): [150..200, 100..130]
        mask[150:200, 100:130] = 12

        survivor = self.MockSurvivor(body_ids=[10, 11, 12])

        labels = compute_survivor_labels(
            seg_mask=mask,
            survivors=[survivor],
            image_width=300,
            image_height=300,
        )

        assert len(labels) == 1
        lbl = labels[0]

        # Overall merged box should span:
        # x: [100, 130], y: [80, 200]
        x1, y1, x2, y2 = lbl.bbox_xyxy
        assert x1 == pytest.approx(100.0)
        assert y1 == pytest.approx(80.0)
        assert x2 == pytest.approx(130.0)
        assert y2 == pytest.approx(200.0)

        # Total visible pixels = 50*30 + 20*20 + 50*30 = 1500 + 400 + 1500 = 3400
        assert lbl.visible_pixels == 3400

    def test_multiple_distinct_survivors(self) -> None:
        mask = np.full((500, 500), -1, dtype=np.int32)

        # Survivor 1 (body 1): top-left
        mask[50:100, 50:100] = 1

        # Survivor 2 (body 2): bottom-right
        mask[300:380, 350:420] = 2

        labels = compute_survivor_labels(
            seg_mask=mask,
            survivors=[1, 2],
            image_width=500,
            image_height=500,
        )

        assert len(labels) == 2
        assert labels[0].survivor_id == 0
        assert labels[1].survivor_id == 1


# --------------------------------------------------------------------------- #
# 7. File I/O Tests
# --------------------------------------------------------------------------- #
class TestFileIO:
    """Verify saving and loading YOLO annotation files."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        labels = [
            SurvivorLabel(
                survivor_id=0,
                class_id=0,
                bbox_xyxy=(10.0, 20.0, 50.0, 100.0),
                yolo_bbox=(0.25, 0.40, 0.15, 0.30),
                visible_pixels=500,
            ),
            SurvivorLabel(
                survivor_id=1,
                class_id=0,
                bbox_xyxy=(200.0, 300.0, 280.0, 420.0),
                yolo_bbox=(0.60, 0.70, 0.20, 0.25),
                visible_pixels=800,
            ),
        ]

        out_file = tmp_path / "labels" / "frame_0001.txt"
        saved_path = save_yolo_labels(labels, out_file)
        assert saved_path.is_file()

        # Load back
        loaded = load_yolo_labels(saved_path)
        assert len(loaded) == 2

        # Check first line
        cid1, cx1, cy1, w1, h1 = loaded[0]
        assert cid1 == 0
        assert cx1 == pytest.approx(0.25, abs=1e-5)
        assert cy1 == pytest.approx(0.40, abs=1e-5)
        assert w1 == pytest.approx(0.15, abs=1e-5)
        assert h1 == pytest.approx(0.30, abs=1e-5)

        # Check second line
        cid2, cx2, cy2, w2, h2 = loaded[1]
        assert cid2 == 0
        assert cx2 == pytest.approx(0.60, abs=1e-5)
        assert cy2 == pytest.approx(0.70, abs=1e-5)
        assert w2 == pytest.approx(0.20, abs=1e-5)
        assert h2 == pytest.approx(0.25, abs=1e-5)

    def test_save_empty_labels_creates_empty_file(self, tmp_path: Path) -> None:
        out_file = tmp_path / "empty.txt"
        save_yolo_labels([], out_file)
        assert out_file.is_file()
        assert out_file.read_text(encoding="utf-8") == ""
        loaded = load_yolo_labels(out_file)
        assert len(loaded) == 0


# --------------------------------------------------------------------------- #
# 9. Training-resolution thresholds
# --------------------------------------------------------------------------- #
class TestTrainingResolutionThresholds:
    @pytest.mark.parametrize(
        ("width", "height", "scale"), [(1280, 720, 416 / 1280), (640, 360, 416 / 640)]
    )
    def test_conversion_to_render_pixels(self, width: int, height: int, scale: float) -> None:
        cfg = LabelConfig(
            imgsz=416, min_box_height_px=10, min_box_width_px=6, presence_min_area_px=1.6
        )
        th = cfg.thresholds(width, height)
        assert cfg.scale(width, height) == pytest.approx(scale)
        assert th.min_box_height == pytest.approx(10 / scale)
        assert th.min_box_width == pytest.approx(6 / scale)
        assert th.presence_min_pixels == pytest.approx(1.6 / scale**2)

    def test_known_values(self) -> None:
        cfg = LabelConfig()
        hi, lo = cfg.thresholds(1280, 720), cfg.thresholds(640, 360)
        assert hi.min_box_height == pytest.approx(30.769, abs=1e-3)
        assert lo.min_box_height == pytest.approx(15.385, abs=1e-3)
        assert hi.min_box_width == pytest.approx(18.462, abs=1e-3)
        assert lo.min_box_width == pytest.approx(9.231, abs=1e-3)
        assert hi.presence_min_pixels == pytest.approx(15.15, abs=0.01)
        assert lo.presence_min_pixels == pytest.approx(3.79, abs=0.01)

    def test_portrait_uses_longest_side(self) -> None:
        assert LabelConfig().scale(360, 640) == pytest.approx(416 / 640)

    def test_values_from_arena_yaml(self) -> None:
        pytest.importorskip("yaml")
        cfg = LabelConfig.from_arena_config(Path("configs/arena.yaml"))
        assert cfg.imgsz == 416
        assert cfg.min_box_height_px == 10
        assert cfg.min_box_width_px == 6
        assert cfg.max_truncation == 0.85
        assert cfg.max_occlusion == 0.90

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        cfg_file = tmp_path / "arena.yaml"
        cfg_file.write_text("labels:\n  min_box_hieght_px: 10\n", encoding="utf-8")
        with pytest.raises(ValueError, match="min_box_hieght_px"):
            LabelConfig.from_arena_config(cfg_file)

    def test_invalid_values_rejected(self) -> None:
        with pytest.raises(ValueError):
            LabelConfig(max_truncation=1.5)
        with pytest.raises(ValueError):
            LabelConfig(min_box_width_px=-1)


# --------------------------------------------------------------------------- #
# 10. label_frame with synthetic silhouettes
# --------------------------------------------------------------------------- #
BODY = 5


def _scene(width: int, height: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """Mask with body BODY filling box (x1, y1, x2, y2)."""
    mask = np.full((height, width), -1, dtype=np.int32)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = BODY
    return mask


def _silhouette(in_frame: np.ndarray, outside: int = 0, exact: bool = True) -> Silhouette:
    return Silhouette(
        in_frame=in_frame,
        in_frame_pixels=int(in_frame.sum()),
        outside_pixels=outside,
        exact=exact,
        canvas_area_factor=1.0,
    )


def _frame(width, height, box, alone_box=None, outside=0, exact=True, config=None):
    seg = _scene(width, height, box)
    alone = _scene(width, height, alone_box or box) == BODY
    return label_frame(seg, [BODY], [_silhouette(alone, outside, exact)], config)


class TestLabelFrame:
    @pytest.mark.parametrize(
        ("width", "height", "box"),
        [(1280, 720, (100, 100, 120, 140)), (640, 360, (50, 50, 60, 70))],
    )
    def test_narrow_standing_figure_kept(self, width, height, box) -> None:
        # 20 x 40 px at 1280 = 6.5 x 13 px at 416; 10 x 20 px at 640 is the same.
        result = _frame(width, height, box)
        assert len(result.labels) == 1 and result.dropped == []
        assert result.labels[0].truncation_fraction == 0.0
        assert result.labels[0].occlusion_fraction == 0.0

    @pytest.mark.parametrize(
        ("width", "height", "box"),
        [(1280, 720, (100, 100, 115, 140)), (640, 360, (50, 50, 58, 70))],
    )
    def test_too_narrow_dropped(self, width, height, box) -> None:
        # 15 px at 1280 = 4.9 px at 416; 8 px at 640 = 5.2 px at 416.
        result = _frame(width, height, box)
        assert result.labels == []
        assert result.dropped[0].reasons == (DROP_BOX_TOO_NARROW,)

    @pytest.mark.parametrize(
        ("width", "height", "box"),
        [(1280, 720, (100, 100, 140, 125)), (640, 360, (50, 50, 70, 62))],
    )
    def test_too_short_dropped(self, width, height, box) -> None:
        # 25 px at 1280 = 8.1 px at 416; 12 px at 640 = 7.8 px at 416.
        result = _frame(width, height, box)
        assert result.dropped[0].reasons == (DROP_BOX_TOO_SHORT,)

    def test_presence_floor_scales_with_render_size(self) -> None:
        # 9 px at 1280x720 is under the floor (15.1 px): absent, not dropped.
        absent = _frame(1280, 720, (10, 10, 13, 13))
        assert absent.labels == [] and absent.dropped == []
        # 4 px at 640x360 is over the floor (3.8 px): present, dropped for size.
        present = _frame(640, 360, (10, 10, 12, 12))
        assert present.labels == []
        assert present.dropped[0].reasons == (DROP_BOX_TOO_SHORT, DROP_BOX_TOO_NARROW)

    def test_truncated(self) -> None:
        box = (0, 100, 40, 160)  # 2400 px in frame
        result = _frame(1280, 720, box, outside=21600)  # 90 % outside
        assert result.dropped[0].reasons == (DROP_TRUNCATED,)
        assert result.dropped[0].label.truncation_fraction == pytest.approx(0.9)

    def test_truncation_kept_below_limit(self) -> None:
        result = _frame(1280, 720, (0, 100, 40, 160), outside=2400)
        assert result.labels[0].truncation_fraction == pytest.approx(0.5)

    def test_truncation_unbounded(self) -> None:
        result = _frame(1280, 720, (0, 100, 40, 160), outside=2400, exact=False)
        assert result.dropped[0].reasons == (DROP_TRUNCATION_UNBOUNDED,)
        # A lower bound already over the limit is plainly truncated.
        result = _frame(1280, 720, (0, 100, 40, 160), outside=24000, exact=False)
        assert result.dropped[0].reasons == (DROP_TRUNCATED,)

    def test_occluded(self) -> None:
        # Visible 40 x 40 of a 40 x 440 alone silhouette: occlusion 0.909.
        result = _frame(1280, 720, (100, 100, 140, 140), alone_box=(100, 100, 140, 540))
        assert result.dropped[0].reasons == (DROP_OCCLUDED,)
        assert result.dropped[0].label.occlusion_fraction == pytest.approx(1 - 1 / 11)

    def test_occlusion_kept_below_limit(self) -> None:
        result = _frame(1280, 720, (100, 100, 140, 140), alone_box=(100, 100, 140, 180))
        assert result.labels[0].occlusion_fraction == pytest.approx(0.5)

    def test_config_thresholds_used(self) -> None:
        strict = LabelConfig(min_box_width_px=8.0)
        result = _frame(1280, 720, (100, 100, 120, 140), config=strict)  # 6.5 px at 416
        assert result.dropped[0].reasons == (DROP_BOX_TOO_NARROW,)

    def test_silhouette_count_must_match(self) -> None:
        seg = _scene(320, 180, (10, 10, 60, 90))
        with pytest.raises(ValueError, match="silhouettes"):
            label_frame(seg, [BODY], [])

    def test_visible_without_silhouette_raises(self) -> None:
        seg = _scene(320, 180, (10, 10, 60, 90))
        with pytest.raises(ValueError, match="no silhouette"):
            label_frame(seg, [BODY], [None])


class TestComputeSurvivorLabelsWithSilhouettes:
    def test_measured_values_set(self) -> None:
        seg = _scene(320, 180, (0, 10, 40, 90))
        alone = _scene(320, 180, (0, 10, 40, 170)) == BODY
        labels = compute_survivor_labels(seg, [BODY], silhouettes=[_silhouette(alone, 800)])
        assert labels[0].truncation_fraction == pytest.approx(800 / (6400 + 800))
        assert labels[0].occlusion_fraction == pytest.approx(0.5)

    def test_unbounded_truncation_dropped(self) -> None:
        seg = _scene(320, 180, (0, 10, 40, 90))
        sil = _silhouette(seg == BODY, 100, exact=False)
        assert compute_survivor_labels(seg, [BODY], silhouettes=[sil]) == []
