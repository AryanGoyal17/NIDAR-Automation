"""
tests/test_data_tools/test_check_generated.py — Run validation and contact sheets
=================================================================================

One tiny generated run (160x90, 3 scenes x 3 frames), copied per test before
it is corrupted.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("pybullet")

import sim.generator as gen  # noqa: E402
from data_tools.check_generated import counts, load_run, main, stats, validate  # noqa: E402
from data_tools.draw_boxes import draw_sheet  # noqa: E402


@pytest.fixture(scope="module")
def base_run(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("gen") / "run"
    cfg = replace(gen.GeneratorConfig.from_yaml(), width=160, height=90)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gen, "power_state", lambda: "not read (test)")
        gen.generate(out, 3, 3, 7, workers=1, cfg=cfg)
    return out


@pytest.fixture
def run(base_run, tmp_path) -> Path:
    out = tmp_path / "run"
    shutil.copytree(base_run, out)
    return out


def _positive(run: Path) -> dict:
    metas = load_run(run)["meta"]
    return next(m for m in metas if m["labels"])


class TestValidate:
    def test_valid_run(self, base_run) -> None:
        assert validate(base_run) == []
        assert main(["--run", str(base_run), "--stats"]) == 0

    def test_counts_and_stats(self, base_run) -> None:
        data = load_run(base_run)
        c = counts(data)
        assert c["frames"] == len(data["meta"]) == sum(c["by_mode"].values())
        assert sum(c["by_category"].values()) == c["frames"]
        s = stats(data)
        assert s["frames_written"] == c["frames"]
        assert s["kept_labels"] == c["labels"] == sum(s["truncation_hist"].values())
        assert sum(s["kept_labels_by_posture"].values()) == c["labels"]
        assert {m: v["written"] for m, v in s["per_mode"].items() if v["written"]} == c["by_mode"]

    def test_value_out_of_range(self, run) -> None:
        m = _positive(run)
        (run / m["label"]).write_text("0 1.5 0.5 0.1 0.1\n", encoding="utf-8")
        errors = validate(run)
        assert any("outside [0, 1]" in e for e in errors)
        assert main(["--run", str(run)]) == 1

    def test_degenerate_box(self, run) -> None:
        m = _positive(run)
        (run / m["label"]).write_text("0 0.5 0.5 0.0 0.1\n", encoding="utf-8")
        assert any("degenerate" in e for e in validate(run))

    def test_wrong_class_and_fields(self, run) -> None:
        m = _positive(run)
        (run / m["label"]).write_text("1 0.5 0.5 0.1 0.1\n0 0.5 0.5\n", encoding="utf-8")
        errors = validate(run)
        assert any("class 1" in e for e in errors)
        assert any("3 fields" in e for e in errors)

    def test_missing_label_file(self, run) -> None:
        m = load_run(run)["meta"][0]
        (run / m["label"]).unlink()
        assert any("image without label file" in e for e in validate(run))

    def test_label_does_not_match_meta(self, run) -> None:
        m = _positive(run)
        (run / m["label"]).write_text("", encoding="utf-8")
        errors = validate(run)
        assert any("does not match meta" in e for e in errors)

    def test_duplicate_image(self, run) -> None:
        metas = load_run(run)["meta"]
        shutil.copyfile(run / metas[0]["image"], run / metas[1]["image"])
        assert any("same image bytes" in e for e in validate(run))

    def test_orphan_image(self, run) -> None:
        m = load_run(run)["meta"][0]
        shutil.copyfile(run / m["image"], run / "images" / "s09999_f000.jpg")
        errors = validate(run)
        assert any("s09999_f000: image without label file" in e for e in errors)
        assert any("s09999_f000: file without meta line" in e for e in errors)

    def test_ambiguous_meta_flagged(self, run) -> None:
        lines = (run / "meta.jsonl").read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["dropped"] = [{"reasons": ["truncated"]}]
        lines[0] = json.dumps(first)
        (run / "meta.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert any("dropped labels" in e for e in validate(run))


class TestDrawBoxes:
    def test_sheet_from_written_files(self, run, tmp_path) -> None:
        sheet = tmp_path / "sheet.png"
        chosen = draw_sheet(run, sheet, tiles=4, seed=0, cols=2)
        assert len(chosen) == 4
        img = cv2.imread(str(sheet))
        assert img is not None and img.shape[1] == 2 * 320

    def test_boxes_come_from_label_files(self, run, tmp_path) -> None:
        m = _positive(run)
        fid = m["frame_id"]
        before = tmp_path / "before.png"
        after = tmp_path / "after.png"
        chosen = draw_sheet(run, before, tiles=99, seed=0)
        assert fid in chosen
        (run / m["label"]).write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")
        draw_sheet(run, after, tiles=99, seed=0)
        a, b = cv2.imread(str(before)), cv2.imread(str(after))
        assert a.shape == b.shape
        assert not np.array_equal(a, b)
