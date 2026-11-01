"""
tests/test_data_tools/test_coco_person_sets.py — COCO anchor and sanity person sets
===================================================================================

Box conversion, the size rule at 416, the all-or-nothing and crowd rules, negative
filters, unchanged JPEG bytes and labels. Synthetic fixtures only (``conftest.py``).
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from data_tools.coco_common import (  # noqa: E402
    CocoImage,
    ConfirmationNeeded,
    ImageZip,
    xywh_to_yolo,
)
from data_tools.coco_person_sets import (  # noqa: E402
    caption_mentions_person,
    caption_regex,
    classify_person_image,
    data_yaml,
    plan_set,
    run,
    write_set,
)
from data_tools.coco_sheets import yolo_sheet  # noqa: E402
from tests.test_data_tools.coco_synth import LIMITS, REPO, SPEC, person, stub_detector  # noqa: E402


class TestBoxes:
    def test_xywh_to_yolo_hand_made(self):
        assert xywh_to_yolo([10, 20, 30, 40], 100, 200) == pytest.approx((0.25, 0.2, 0.3, 0.2))
        assert xywh_to_yolo([0, 0, 640, 480], 640, 480) == pytest.approx((0.5, 0.5, 1.0, 1.0))

    def test_xywh_to_yolo_clips_to_image(self):
        assert xywh_to_yolo([-10, 90, 20, 20], 100, 100) == pytest.approx((0.05, 0.95, 0.1, 0.1))

    def test_limits_come_from_arena_yaml(self, coco):
        arena = yaml.safe_load((REPO / "configs" / "arena.yaml").read_text(encoding="utf-8"))[
            "labels"
        ]
        lim = coco["cfg"].limits
        assert (lim.imgsz, lim.min_height_px, lim.min_width_px) == (
            arena["imgsz"],
            arena["min_box_height_px"],
            arena["min_box_width_px"],
        )

    def test_size_rule_after_letterbox(self):
        # 832x416 -> scale 0.5: 20 px tall / 12 px wide in the source is exactly 10 / 6 at 416
        def im(bbox):
            return CocoImage(1, "x.jpg", 832, 416, persons=[person(1, bbox)])

        assert classify_person_image(im([0, 0, 12, 20]), LIMITS).kept
        assert classify_person_image(im([0, 0, 12, 19.9]), LIMITS).reasons == ["person_too_short"]
        assert classify_person_image(im([0, 0, 11.9, 20]), LIMITS).reasons == ["person_too_narrow"]
        # Portrait 416x832 uses the height for the scale too
        portrait = CocoImage(1, "x.jpg", 416, 832, persons=[person(1, [0, 0, 12, 19.9])])
        assert not classify_person_image(portrait, LIMITS).kept

    def test_one_small_person_excludes_whole_image(self, coco):
        d = classify_person_image(coco["index"][2], coco["cfg"].limits)
        assert not d.kept
        assert set(d.reasons) == {"person_too_short", "person_too_narrow"}

    def test_person_crowd_excludes_image(self, coco):
        d = classify_person_image(coco["index"][3], coco["cfg"].limits)
        assert not d.kept and d.reasons == ["person_crowd"]

    def test_other_category_crowd_ignored(self, coco):
        assert classify_person_image(coco["index"][1], coco["cfg"].limits).kept

    def test_rle_person_kept_for_boxes(self, coco):
        d = classify_person_image(coco["index"][4], coco["cfg"].limits)
        assert d.kept and d.boxes[0]["rle"]


class TestPersonSets:
    def test_plan_keeps_only_clean_images(self, coco):
        with ImageZip(coco["zip"]) as z:
            plan = plan_set("anchor", set(SPEC), coco["index"], coco["cfg"], stub_detector, z)
        kinds = {e["image_id"]: e["kind"] for e in plan["images"]}
        assert {i for i, k in kinds.items() if k == "positive"} == {1, 4}
        assert 2 not in kinds and 3 not in kinds
        assert plan["exclusions_by_reason"]["person_crowd"] == 1
        assert plan["exclusions_by_reason"]["person_too_short"] == 1
        assert plan["counts"]["rle_person_boxes_kept"] == 1

    def test_negative_filters(self, coco):
        with ImageZip(coco["zip"]) as z:
            plan = plan_set("anchor", set(SPEC), coco["index"], coco["cfg"], stub_detector, z)
        negatives = {e["image_id"] for e in plan["images"] if e["kind"] == "negative"}
        assert 6 not in negatives and 7 not in negatives
        assert len(negatives) == 2 and negatives <= {5, 8, 9}
        nf = plan["negative_filters"]
        assert (nf["candidates"], nf["removed_by_caption"], nf["removed_by_detector"]) == (5, 1, 1)
        assert (nf["passed"], nf["selected"]) == (3, 2)
        assert plan["exclusions_by_reason"]["negative_over_quota"] == 1

    def test_caption_words_whole_word(self):
        pattern = caption_regex(["man", "he"])
        assert (
            caption_mentions_person(CocoImage(1, "", 1, 1, captions=["A Man walks"]), pattern)
            == "man"
        )
        assert (
            caption_mentions_person(CocoImage(1, "", 1, 1, captions=["the mango tree"]), pattern)
            is None
        )

    def test_written_bytes_equal_zip_member(self, coco):
        out = coco["tmp"] / "sets"
        with ImageZip(coco["zip"]) as z:
            plan = plan_set("anchor", set(SPEC), coco["index"], coco["cfg"], stub_detector, z)
            write_set(plan, out, z)
        with zipfile.ZipFile(coco["zip"]) as zf:
            for e in plan["images"]:
                member = hashlib.sha256(zf.read(f"val2017/{e['file']}")).hexdigest()
                written = hashlib.sha256(
                    (out / "anchor" / "images" / e["file"]).read_bytes()
                ).hexdigest()
                assert member == written == e["sha256"]

    def test_written_labels(self, coco):
        out = coco["tmp"] / "sets"
        with ImageZip(coco["zip"]) as z:
            plan = plan_set("anchor", set(SPEC), coco["index"], coco["cfg"], stub_detector, z)
            write_set(plan, out, z)
        labels = out / "anchor" / "labels"
        line = (labels / "000000000001.txt").read_text().split()
        assert line[0] == "0"
        assert [float(v) for v in line[1:]] == pytest.approx(
            [125 / 640, 200 / 480, 50 / 640, 200 / 480], abs=1e-6
        )
        for e in plan["images"]:
            text = (labels / f"{Path(e['file']).stem}.txt").read_text()
            assert (text == "") == (e["kind"] == "negative")

    def test_data_yaml(self, tmp_path):
        data = yaml.safe_load(data_yaml("sanity", tmp_path / "sanity"))
        assert data["names"] == {0: "person"}
        assert (
            data["val"] == data["train"] == (tmp_path / "sanity").resolve().as_posix() + "/images"
        )


def write_split(cfg, groups: dict[str, list[int]]) -> None:
    Path(cfg["paths"]["split"]).write_text(json.dumps({"groups": groups}))


class TestRun:
    def test_end_to_end(self, coco):
        cfg = coco["cfg"]
        write_split(cfg, {"cutout_source": [], "anchor": [1, 2, 3, 5, 6, 7], "sanity": [4, 8, 9]})
        plans = run(cfg, detector=stub_detector)
        out = Path(cfg["paths"]["person_sets"])
        anchor = json.loads((out / "anchor" / "manifest.json").read_text())
        assert anchor["counts"]["positives"] == 1 and anchor["counts"]["negatives"] == 1
        assert sorted(p.name for p in (out / "anchor" / "images").iterdir()) == [
            "000000000001.jpg",
            "000000000005.jpg",
        ]
        sanity = yaml.safe_load((out / "sanity.yaml").read_text())
        assert sanity["names"] == {0: "person"}
        assert sanity["val"].endswith("/sanity/images")
        assert plans["sanity"]["counts"]["rle_person_boxes_kept"] == 1
        assert not list(out.glob("*.plan.json"))
        sheet = yolo_sheet(out / "anchor", 4, 0)
        assert sheet.ndim == 3 and sheet.shape[2] == 3

    def test_refuses_to_overwrite(self, coco):
        cfg = coco["cfg"]
        write_split(cfg, {"cutout_source": [], "anchor": [1, 5], "sanity": [9]})
        run(cfg, detector=stub_detector)
        with pytest.raises(FileExistsError):
            run(cfg, detector=stub_detector)
        run(cfg, detector=stub_detector, overwrite=True)

    def test_confirmation_above_size_limit(self, coco):
        cfg = coco["cfg"]
        cfg.raw["disk"]["confirm_above_gb"] = 1e-9
        write_split(cfg, {"cutout_source": [], "anchor": [1, 5], "sanity": [9]})
        with pytest.raises(ConfirmationNeeded):
            run(cfg, detector=stub_detector)
        assert not (Path(cfg["paths"]["person_sets"]) / "anchor").exists()
        run(cfg, detector=stub_detector, yes=True)
