"""
tests/test_cv/test_detection_output.py
=======================================
Validates that a sample detection JSONL line conforms to the project's
output convention defined in docs/conventions.md (v0.1):

- bbox_xyxy: list of 4 floats (pixels, xyxy format)
- bbox_center: list of 2 floats (pixels)
- timestamp_utc: ISO 8601 UTC with millisecond precision (YYYY-MM-DDTHH:MM:SS.mmmZ)
- class_id: int
- confidence: float in [0, 1]
- All required top-level and per-detection keys present
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Sample JSONL line matching what detect_survivors.py actually produces
# ---------------------------------------------------------------------------
SAMPLE_JSONL = (
    '{"frame_id": 0, "source": "survivor_hallway.jpg", "video_time_s": null,'
    ' "timestamp_utc": "2026-10-21T17:54:32.668Z",'
    ' "frame_width": 1280, "frame_height": 720, "inference_ms": 55.2,'
    ' "detections": [{"det_id": 0, "class_id": 0, "class_name": "person",'
    ' "confidence": 0.7127, "bbox_xyxy": [557.7, 464.3, 837.3, 704.9],'
    ' "bbox_center": [697.5, 584.6]}]}'
)

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)

REQUIRED_FRAME_KEYS = {
    "frame_id",
    "source",
    "video_time_s",
    "timestamp_utc",
    "frame_width",
    "frame_height",
    "inference_ms",
    "detections",
}

REQUIRED_DETECTION_KEYS = {
    "det_id",
    "class_id",
    "class_name",
    "confidence",
    "bbox_xyxy",
    "bbox_center",
}


@pytest.fixture
def record() -> dict:
    return json.loads(SAMPLE_JSONL)


# ---------- Top-level frame record ----------


class TestFrameRecord:
    def test_required_keys_present(self, record: dict) -> None:
        missing = REQUIRED_FRAME_KEYS - record.keys()
        assert not missing, f"Missing top-level keys: {missing}"

    def test_frame_id_is_int(self, record: dict) -> None:
        assert isinstance(record["frame_id"], int)

    def test_timestamp_format(self, record: dict) -> None:
        ts = record["timestamp_utc"]
        assert TIMESTAMP_RE.match(ts), (
            f"timestamp_utc '{ts}' does not match "
            "YYYY-MM-DDTHH:MM:SS.mmmZ (millisecond precision, UTC)"
        )

    def test_frame_dimensions_are_int(self, record: dict) -> None:
        assert isinstance(record["frame_width"], int)
        assert isinstance(record["frame_height"], int)

    def test_inference_ms_is_number(self, record: dict) -> None:
        assert isinstance(record["inference_ms"], (int, float))

    def test_detections_is_list(self, record: dict) -> None:
        assert isinstance(record["detections"], list)


# ---------- Per-detection fields ----------


class TestDetectionFields:
    @pytest.fixture
    def det(self, record: dict) -> dict:
        assert len(record["detections"]) > 0, "No detections in sample"
        return record["detections"][0]

    def test_required_keys_present(self, det: dict) -> None:
        missing = REQUIRED_DETECTION_KEYS - det.keys()
        assert not missing, f"Missing detection keys: {missing}"

    def test_class_id_is_int(self, det: dict) -> None:
        assert isinstance(det["class_id"], int)

    def test_confidence_is_float_in_range(self, det: dict) -> None:
        c = det["confidence"]
        assert isinstance(c, float)
        assert 0.0 <= c <= 1.0, f"confidence {c} outside [0, 1]"

    def test_bbox_xyxy_is_four_floats(self, det: dict) -> None:
        bbox = det["bbox_xyxy"]
        assert isinstance(bbox, list) and len(bbox) == 4, (
            f"bbox_xyxy must be a list of 4 elements, got {bbox}"
        )
        for v in bbox:
            assert isinstance(v, (int, float)), (
                f"bbox_xyxy values must be numeric, got {type(v)}"
            )

    def test_bbox_xyxy_ordering(self, det: dict) -> None:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        assert x1 <= x2, f"x1 ({x1}) > x2 ({x2})"
        assert y1 <= y2, f"y1 ({y1}) > y2 ({y2})"

    def test_bbox_center_is_two_floats(self, det: dict) -> None:
        center = det["bbox_center"]
        assert isinstance(center, list) and len(center) == 2, (
            f"bbox_center must be a list of 2 elements, got {center}"
        )
        for v in center:
            assert isinstance(v, (int, float)), (
                f"bbox_center values must be numeric, got {type(v)}"
            )

    def test_bbox_center_matches_xyxy(self, det: dict) -> None:
        """Center should be the midpoint of the xyxy box."""
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx, cy = det["bbox_center"]
        assert abs(cx - (x1 + x2) / 2) < 1.0, "cx not near midpoint of x1, x2"
        assert abs(cy - (y1 + y2) / 2) < 1.0, "cy not near midpoint of y1, y2"

    def test_class_name_is_person(self, det: dict) -> None:
        assert det["class_name"] == "person"


# ---------- Live output file (if it exists) ----------


class TestLiveOutput:
    """Validate the actual detections.jsonl file if it exists on disk."""

    JSONL_PATH = Path("outputs/runs_nidar/detections.jsonl")

    @pytest.fixture
    def live_records(self) -> list[dict]:
        if not self.JSONL_PATH.exists():
            pytest.skip("No live detections.jsonl found")
        records = []
        for line in self.JSONL_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        if not records:
            pytest.skip("detections.jsonl is empty")
        return records

    def test_all_frame_keys_present(self, live_records: list[dict]) -> None:
        for i, rec in enumerate(live_records):
            missing = REQUIRED_FRAME_KEYS - rec.keys()
            assert not missing, f"Record {i}: missing keys {missing}"

    def test_all_timestamps_ms_precision(self, live_records: list[dict]) -> None:
        for i, rec in enumerate(live_records):
            ts = rec["timestamp_utc"]
            assert TIMESTAMP_RE.match(ts), (
                f"Record {i}: timestamp '{ts}' not in ms-precision format"
            )

    def test_all_detections_have_class_id(self, live_records: list[dict]) -> None:
        for i, rec in enumerate(live_records):
            for j, det in enumerate(rec["detections"]):
                assert "class_id" in det, (
                    f"Record {i}, detection {j}: missing class_id"
                )

    def test_all_bboxes_are_float(self, live_records: list[dict]) -> None:
        for i, rec in enumerate(live_records):
            for j, det in enumerate(rec["detections"]):
                for v in det["bbox_xyxy"]:
                    assert isinstance(v, (int, float)), (
                        f"Record {i}, det {j}: bbox value {v} is not numeric"
                    )
