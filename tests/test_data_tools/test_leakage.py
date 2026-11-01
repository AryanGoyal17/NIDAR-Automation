"""
tests/test_data_tools/test_leakage.py — Train/protected leakage check
=====================================================================

Synthetic images only: planted byte duplicates, a planted COCO id with other
bytes, a resized copy (near match, warning only) and cutout_source violations.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

from data_tools.coco_common import ImageZip, dhash  # noqa: E402
from data_tools.leakage import (  # noqa: E402
    check,
    hamming_pairs,
    main,
    read_list,
    scan_cutout_sources,
    scan_dir,
    scan_paths,
)

REPO = Path(__file__).resolve().parents[2]


def photo(seed: int, w: int = 320, h: int = 240) -> np.ndarray:
    """Smooth random image (blurred noise) so that dHash is stable under resizing."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def write(path: Path, image: np.ndarray, quality: int = 95) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    train, protected = tmp_path / "train", tmp_path / "protected"
    for k in range(4):
        write(train / f"{k + 1:012d}.jpg", photo(k))
        write(protected / f"{k + 101:012d}.jpg", photo(k + 100))
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "groups": {
                    "cutout_source": [201, 202],
                    "anchor": [1, 2, 3, 4],
                    "sanity": [101, 102, 103, 104],
                }
            }
        )
    )
    return {"train": train, "protected": protected, "split": split, "tmp": tmp_path}


def run(dirs: dict[str, Path], *extra: str) -> int:
    return main(
        [
            "--train",
            str(dirs["train"]),
            "--protected",
            str(dirs["protected"]),
            "--split",
            str(dirs["split"]),
        ]
        + list(extra)
    )


class TestLeakage:
    def test_clean_passes(self, dirs, capsys):
        assert run(dirs) == 0
        assert "0 errors" in capsys.readouterr().out

    def test_planted_duplicate_fails(self, dirs, capsys):
        # Same bytes under another name: caught by sha256
        (dirs["train"] / "copy_of_sanity.jpg").write_bytes(
            (dirs["protected"] / "000000000101.jpg").read_bytes()
        )
        assert run(dirs) == 1
        assert "same sha256" in capsys.readouterr().out

    def test_same_coco_id_other_bytes_fails(self, dirs, capsys):
        write(dirs["train"] / "000000000102.jpg", photo(999))
        assert run(dirs) == 1
        assert "same COCO id 102" in capsys.readouterr().out

    def test_resized_copy_is_near_match_warning_only(self, dirs, capsys):
        small = cv2.resize(photo(100), (160, 120), interpolation=cv2.INTER_AREA)
        write(dirs["train"] / "resized.jpg", small, quality=80)
        assert run(dirs) == 0
        out = capsys.readouterr().out
        assert "WARNING near match" in out and "000000000101.jpg" in out
        assert "0 errors, 1 near-match warnings" in out

    def test_cutout_source_image_in_anchor_fails(self, dirs, capsys):
        write(dirs["train"] / "000000000201.jpg", photo(201))
        assert run(dirs) == 1
        assert "cutout_source image id 201" in capsys.readouterr().out

    def test_split_check_can_be_skipped(self, dirs):
        write(dirs["train"] / "000000000201.jpg", photo(201))
        assert (
            main(
                [
                    "--train",
                    str(dirs["train"]),
                    "--protected",
                    str(dirs["protected"]),
                    "--split",
                    "none",
                ]
            )
            == 0
        )

    def test_list_file_clean_and_planted(self, dirs, capsys):
        """A list file is scanned like a directory; a sanity image in it fails."""
        lst = dirs["tmp"] / "lists" / "train.txt"
        lst.parent.mkdir()
        good = sorted(dirs["train"].glob("*.jpg"))
        lst.write_text("".join(f"{p}\n" for p in good), encoding="utf-8")
        args = ["--train-list", str(lst), "--protected", str(dirs["protected"]), "--split", "none"]
        assert main(args) == 0
        assert f"training sources: {len(good)} images" in capsys.readouterr().out
        planted = dirs["protected"] / "000000000103.jpg"
        lst.write_text(lst.read_text(encoding="utf-8") + f"{planted}\n", encoding="utf-8")
        assert main(args) == 1
        assert "same sha256" in capsys.readouterr().out

    def test_read_list_relative_and_blank_lines(self, tmp_path):
        lst = tmp_path / "sub" / "l.txt"
        lst.parent.mkdir()
        lst.write_text(f"a/x.jpg\n\n{tmp_path / 'y.jpg'}\n", encoding="utf-8")
        assert read_list(lst) == [lst.parent / "a" / "x.jpg", tmp_path / "y.jpg"]

    def test_scan_paths_equals_scan_dir(self, dirs):
        assert scan_paths(sorted(dirs["train"].glob("*.jpg"))) == scan_dir(dirs["train"])

    def test_cutout_sources_from_manifest(self, dirs):
        zpath = dirs["tmp"] / "val2017.zip"
        # 101 is a sanity image reused as a cutout source: exact match and split violation
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("val2017/000000000201.jpg", cv2.imencode(".jpg", photo(201))[1].tobytes())
            zf.writestr(
                "val2017/000000000101.jpg", (dirs["protected"] / "000000000101.jpg").read_bytes()
            )
        cut = dirs["tmp"] / "cutouts"
        cut.mkdir()
        manifest = {
            "cutouts": [
                {
                    "file": "000000000201_5.png",
                    "source_image_id": 201,
                    "source_file": "000000000201.jpg",
                },
                {
                    "file": "000000000101_7.png",
                    "source_image_id": 101,
                    "source_file": "000000000101.jpg",
                },
            ]
        }
        (cut / "manifest.json").write_text(json.dumps(manifest))
        with ImageZip(zpath) as z:
            train = scan_cutout_sources(cut, z)
        protected = scan_dir(dirs["protected"])
        split = {"cutout_source": {201, 202}, "anchor": set(), "sanity": {101, 102, 103, 104}}
        result = check(train, protected, split, 4)
        assert any("same sha256" in e for e in result["errors"])
        assert any("cutout source id 101 is not in cutout_source" in e for e in result["errors"])
        assert not any("000000000201_5.png" in e for e in result["errors"])

    def test_hamming_pairs(self):
        assert hamming_pairs([0b1011, 2**63], [0b0011, 0], 1) == [(0, 0, 1), (1, 1, 1)]
        assert hamming_pairs([2**64 - 1], [0], 63) == []

    def test_dhash_stable_under_resize(self):
        a, b = dhash(photo(3)), dhash(cv2.resize(photo(3), (100, 75), interpolation=cv2.INTER_AREA))
        assert bin(a ^ b).count("1") <= 4
        assert bin(dhash(photo(3)) ^ dhash(photo(4))).count("1") > 4
