"""
tests/test_data_tools/test_build_dataset.py — List files, group split, leakage gate
===================================================================================

Tiny synthetic generator runs, anchor and sanity sets (no pybullet, no
downloads): label resolution, deterministic sorted lists, the anchor split,
E2 = E1 + anchor_train once, and the refusals (group overlap, shared run seed
or layout, a sanity image in a list by path or by byte copy, anchor share
outside the band).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

from data_tools.build_dataset import (  # noqa: E402
    EXIT_LEAK,
    EXIT_SHARE,
    GroupOverlapError,
    Item,
    ProtectedImageError,
    assert_groups_disjoint,
    assert_not_protected,
    build,
    label_path,
    main,
    split_anchor,
)

REPO = Path(__file__).resolve().parents[2]


def image(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.resize(small, (64, 48), interpolation=cv2.INTER_CUBIC))
    return path


def label(img: Path, boxes: int) -> None:
    lp = label_path(img)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text("".join("0 0.5 0.5 0.2 0.4\n" for _ in range(boxes)), encoding="utf-8")


def sim_run(root: Path, seed: int, scenes: int, layouts: list[str], frames: int = 3) -> Path:
    """Generator-like run: manifest.json + meta.jsonl + images/labels."""
    root.mkdir(parents=True)
    metas = []
    for s in range(scenes):
        name = layouts[s % len(layouts)]
        handmade = not name.startswith("random")
        for f in range(frames):
            fid = f"s{s:05d}_f{f:03d}"
            img = image(root / "images" / f"{fid}.jpg", seed * 1000 + s * 10 + f)
            label(img, f % 2)
            metas.append(
                {
                    "image": f"images/{fid}.jpg",
                    "label": f"labels/{fid}.txt",
                    "master_seed": seed,
                    "scene_index": s,
                    "layout": {
                        "name": name,
                        "source": "handmade" if handmade else "random",
                        "seed": None if handmade else seed * 100 + s,
                        "grid": [name, str(seed), str(s)],
                    },
                }
            )
    (root / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"failed_scenes": [], "config_hash": "x", "git_commit": "y"}), encoding="utf-8"
    )
    return root


def coco_set(root: Path, ids: range, boxes: int = 2) -> Path:
    for cid in ids:
        label(image(root / "images" / f"{cid:012d}.jpg", cid), boxes if cid % 3 else 0)
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    return root


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, Path]:
    train = sim_run(tmp_path / "sim_raw" / "train", 1000, 6, ["dead_end", "random_9x9"])
    val = sim_run(tmp_path / "sim_raw" / "val", 2000, 2, ["loop", "random_7x7"])
    anchor = coco_set(tmp_path / "coco_person" / "anchor", range(1, 41))
    sanity = coco_set(tmp_path / "coco_person" / "sanity", range(101, 111))
    coco_yaml = (REPO / "configs" / "coco.yaml").read_text(encoding="utf-8")
    coco_yaml = coco_yaml.replace("data/coco/split.json", (tmp_path / "no_split.json").as_posix())
    coco_yaml = coco_yaml.replace("min_free_gb: 1.0", "min_free_gb: 0.0")
    coco_yaml = coco_yaml.replace(
        "arena_config: configs/arena.yaml",
        f"arena_config: {(REPO / 'configs' / 'arena.yaml').as_posix()}",
    )
    (tmp_path / "coco.yaml").write_text(coco_yaml, encoding="utf-8")
    cfg = {
        "sim": {"train_runs": [train.as_posix()], "val_runs": [val.as_posix()]},
        "anchor": {"dir": anchor.as_posix(), "val_fraction": 0.15, "seed": 2026},
        "eval": {"sanity_dir": sanity.as_posix()},
        "protected": [(sanity / "images").as_posix(), (tmp_path / "test_real").as_posix()],
        "coco_config": (tmp_path / "coco.yaml").as_posix(),
        "out_dir": (tmp_path / "yolo_ds").as_posix(),
        "anchor_share_band": [0.10, 0.90],
    }
    (tmp_path / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return {
        "tmp": tmp_path,
        "train": train,
        "val": val,
        "anchor": anchor,
        "sanity": sanity,
        "config": tmp_path / "dataset.yaml",
        "out": tmp_path / "yolo_ds",
    }


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


class TestLabelPath:
    def test_ultralytics_rule(self) -> None:
        assert label_path("D:/a/run/images/s0_f1.jpg") == Path("D:/a/run/labels/s0_f1.txt")
        assert label_path(r"D:\a\images\x\images\b.c.png") == Path("D:/a/images/x/labels/b.c.txt")
        with pytest.raises(ValueError):
            label_path("D:/a/pics/b.jpg")

    def test_matches_ultralytics(self) -> None:
        utils = pytest.importorskip("ultralytics.data.utils")
        for p in ["D:/a/run/images/s0_f1.jpg", "D:/x/images/y/images/000000000139.jpg"]:
            native = p.replace("/", "\\") if "\\" in str(Path("a/b")) else p
            assert Path(utils.img2label_paths([native])[0]) == label_path(p)


class TestBuild:
    def test_builds_sorted_lists_and_manifest(self, setup) -> None:
        code, report = build(setup["config"])
        assert code == 0
        out = setup["out"]
        e1, e2 = lines(out / "e1" / "train.txt"), lines(out / "e2" / "train.txt")
        assert e1 == sorted(e1) and e2 == sorted(e2)
        assert len(e1) == 6 * 3 and all("/sim_raw/train/images/" in p for p in e1)
        anchor_train = [p for p in e2 if "/coco_person/anchor/" in p]
        # E2 = E1 + every anchor_train image exactly once
        assert sorted(set(e2) - set(anchor_train)) == e1
        assert len(anchor_train) == len(set(anchor_train))
        assert lines(out / "e1" / "val.txt") == lines(out / "eval" / "sim_val.txt")
        e2_val = set(lines(out / "e2" / "val.txt"))
        assert e2_val == set(lines(out / "eval" / "sim_val.txt")) | set(
            lines(out / "eval" / "anchor_val.txt")
        )
        assert len(lines(out / "eval" / "sanity.txt")) == 10
        data = yaml.safe_load((out / "e2" / "data.yaml").read_text(encoding="utf-8"))
        assert data["names"] == {0: "person"} and data["train"] == "train.txt"
        m = json.loads((out / "e2" / "manifest.json").read_text(encoding="utf-8"))
        assert m["train"]["by_source"]["anchor_train"]["images"] == len(anchor_train)
        assert m["train"]["groups_by_source"]["sim_train"] == 6
        assert m["leakage"]["errors"] == 0
        assert len(m["config_hash"]) == 64
        share = report["e2_anchor_share"]["images"]
        assert share == pytest.approx(len(anchor_train) / len(e2))

    def test_deterministic(self, setup) -> None:
        build(setup["config"])
        first = {p.name: p.read_bytes() for p in (setup["out"] / "e2").glob("*.txt")}
        build(setup["config"])
        again = {p.name: p.read_bytes() for p in (setup["out"] / "e2").glob("*.txt")}
        assert first == again

    def test_anchor_split_deterministic_and_disjoint(self, setup) -> None:
        a = split_anchor(setup["anchor"], 0.15, 2026)
        b = split_anchor(setup["anchor"], 0.15, 2026)
        assert a == b
        train, val = {i.group for i in a["anchor_train"]}, {i.group for i in a["anchor_val"]}
        assert train.isdisjoint(val) and len(train | val) == 40
        assert 0 < len(val) < 20
        assert split_anchor(setup["anchor"], 0.15, 7) != a

    def test_split_independent_of_coco_split(self, tmp_path) -> None:
        """Ids that coco_split put in anchor (u in [0.40, 0.80) with seed 2026) still split ~15 %."""
        from data_tools.coco_common import unit_hash

        ids = [i for i in range(1, 6000) if 0.40 <= unit_hash(2026, i) < 0.80][:400]
        root = tmp_path / "anchor"
        for cid in ids:
            p = root / "images" / f"{cid:012d}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
            label(p, 0)
        a = split_anchor(root, 0.15, 2026)
        assert abs(len(a["anchor_val"]) / len(ids) - 0.15) < 0.05

    def test_label_file_counts(self, setup) -> None:
        a = split_anchor(setup["anchor"], 0.15, 2026)
        for item in a["anchor_train"] + a["anchor_val"]:
            assert item.boxes == (2 if int(item.path.stem) % 3 else 0)

    def test_missing_label_refused(self, setup) -> None:
        label_path(setup["anchor"] / "images" / "000000000005.jpg").unlink()
        with pytest.raises(FileNotFoundError):
            build(setup["config"])


class TestRefusals:
    def test_group_overlap_assertion(self) -> None:
        a = Item(Path("x/images/a.jpg"), "sim_train", "sim:1:0", 1)
        b = Item(Path("y/images/b.jpg"), "sim_val", "sim:1:0", 0)
        with pytest.raises(GroupOverlapError):
            assert_groups_disjoint([a], [b])
        assert_groups_disjoint([a], [Item(b.path, b.source, "sim:2:0", 0)])

    def test_same_run_in_train_and_val_fails(self, setup, capsys) -> None:
        args = ["--config", str(setup["config"]), "--val-run", str(setup["train"])]
        assert main(args) == 1
        assert not (setup["out"] / "e1").exists()

    def test_shared_handmade_layout_fails(self, setup) -> None:
        other = sim_run(setup["tmp"] / "sim_raw" / "val2", 3000, 1, ["dead_end"])
        with pytest.raises(GroupOverlapError, match="handmade_layouts"):
            build(setup["config"], val_runs=[str(other)])

    def test_protected_path_assertion(self, setup) -> None:
        bad = Item(setup["sanity"] / "images" / "000000000101.jpg", "anchor_train", "coco:101", 2)
        with pytest.raises(ProtectedImageError):
            assert_not_protected([bad], [setup["sanity"] / "images"])

    def test_sanity_copy_in_anchor_refused_by_leakage(self, setup) -> None:
        """A sanity image copied into the anchor set under another id: sha256 match, exit 1."""
        src = setup["sanity"] / "images" / "000000000104.jpg"
        dst = setup["anchor"] / "images" / "000000000200.jpg"
        dst.write_bytes(src.read_bytes())
        label(dst, 1)
        code, report = build(setup["config"])
        assert code == EXIT_LEAK and report["leakage"]["errors"] >= 1
        assert not (setup["out"] / "e2").exists()

    def test_sanity_in_training_run_refused(self, setup) -> None:
        """A sanity image as a sim training frame (same bytes): refused, nothing written."""
        meta = setup["train"] / "meta.jsonl"
        first = json.loads(lines(meta)[0])
        img = setup["train"] / first["image"]
        img.write_bytes((setup["sanity"] / "images" / "000000000101.jpg").read_bytes())
        assert main(["--config", str(setup["config"])]) == EXIT_LEAK
        assert not (setup["out"] / "e1").exists()

    def test_anchor_share_band(self, setup) -> None:
        cfg = yaml.safe_load(setup["config"].read_text(encoding="utf-8"))
        cfg["anchor_share_band"] = [0.10, 0.50]  # synthetic set: anchor is about 2/3 of E2
        setup["config"].write_text(yaml.safe_dump(cfg), encoding="utf-8")
        code, report = build(setup["config"])
        assert code == EXIT_SHARE and report["e2_anchor_share"]["images"] > 0.5
        assert not (setup["out"] / "e2").exists()
        code, _ = build(setup["config"], allow_anchor_share=True)
        assert code == 0

    def test_plan_writes_nothing(self, setup) -> None:
        assert main(["--config", str(setup["config"]), "--plan"]) == 0
        assert not setup["out"].exists()
