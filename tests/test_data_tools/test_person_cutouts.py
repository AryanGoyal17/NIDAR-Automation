"""
tests/test_data_tools/test_person_cutouts.py — RGBA person cutouts
==================================================================

Alpha nonzero and tight, colour zeroed outside the mask, and the cutout rules
(height, border, RLE, overlap, mask fraction). Synthetic images only.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from data_tools.coco_common import CocoImage
from data_tools.coco_sheets import cutout_sheet
from data_tools.make_person_cutouts import (
    CutoutRules,
    cutout_from_mask,
    plan_cutouts,
    polygon_mask,
    run,
)
from tests.test_data_tools.coco_synth import person


class TestCutouts:
    def test_alpha_nonzero_and_tight(self):
        rng = np.random.default_rng(0)
        image = rng.integers(1, 255, (300, 400, 3), dtype=np.uint8)
        diamond = [[150, 40, 210, 140, 150, 240, 90, 140]]
        mask = polygon_mask(diamond, 400, 300)
        bgra, box = cutout_from_mask(image, mask)
        alpha = bgra[:, :, 3]
        assert alpha.any()
        assert box == [90, 40, 121, 201]
        # Tight: every edge row and column holds some alpha
        assert alpha[0].any() and alpha[-1].any() and alpha[:, 0].any() and alpha[:, -1].any()
        # Colour is zeroed where alpha is 0, untouched inside
        assert not bgra[alpha == 0][:, :3].any()
        crop = image[40:241, 90:211]
        assert np.array_equal(bgra[alpha > 0][:, :3], crop[alpha > 0])

    def test_rules(self):
        rules = CutoutRules(150, 2, 0.10, 0.30, 1.2)
        good = person(1, [100, 50, 80, 200])
        sparse = dict(
            person(6, [400, 50, 80, 200]), segmentation=[[400, 50, 410, 50, 410, 250, 400, 250]]
        )
        im = CocoImage(
            1,
            "x.jpg",
            640,
            480,
            persons=[
                good,
                person(2, [300, 50, 80, 100]),  # too short
                person(3, [0, 100, 80, 200]),  # touches border
                person(4, [500, 100, 60, 200], rle=True),  # RLE
                sparse,  # mask 10/80 of box width
            ],
        )
        accepted, first, every, candidates = plan_cutouts({1}, {1: im}, rules)
        assert candidates == 5
        assert [c["ann_id"] for c in accepted] == [1]
        assert first == {
            "too_short": 1,
            "touches_border": 1,
            "rle_segmentation": 1,
            "mask_fraction": 1,
        }
        assert accepted[0]["cutout_box_xywh"] == [100, 50, 81, 201]

    def test_overlap_rule(self):
        rules = CutoutRules(150, 2, 0.10, 0.30, 1.2)
        a = person(1, [100, 50, 100, 200])
        b = person(2, [180, 60, 100, 200])  # covers 20 % of a
        crowd = person(3, [400, 50, 100, 200], crowd=1)
        c = person(4, [395, 60, 80, 200])  # covered by the crowd region
        im = CocoImage(1, "x.jpg", 640, 480, persons=[a, b, crowd, c])
        accepted, first, _, candidates = plan_cutouts({1}, {1: im}, rules)
        assert candidates == 3 and not accepted
        assert first == {"overlap": 3}


class TestRun:
    def test_end_to_end(self, coco):
        cfg = coco["cfg"]
        groups = {"cutout_source": [1, 2, 3, 4], "anchor": [5, 6, 7, 8], "sanity": [9]}
        Path(cfg["paths"]["split"]).write_text(json.dumps({"groups": groups}))
        manifest = run(cfg)
        out = Path(cfg["paths"]["cutouts"])
        assert [c["ann_id"] for c in manifest["cutouts"]] == [11, 21, 31]
        assert manifest["counts"]["rejected_first_reason"] == {
            "too_short": 1,
            "rle_segmentation": 1,
        }
        for c in manifest["cutouts"]:
            bgra = cv2.imread(str(out / "images" / c["file"]), cv2.IMREAD_UNCHANGED)
            assert bgra.shape == (c["height"], c["width"], 4)
            assert bgra[:, :, 3].min() == 255  # rectangle polygon: fully opaque, tight crop
        on_disk = json.loads((out / "manifest.json").read_text())
        assert on_disk["counts"]["accepted"] == 3
        sheet = cutout_sheet(out, 3, 0)
        assert sheet.ndim == 3
