"""
data_tools/build_dataset.py — Training list files without copying images
========================================================================

Builds the Ultralytics list files and data yaml files of the two training
variants (``configs/dataset.yaml``; protocol in ADR 008):

- ``E1``: train = sim_train; val = sim_val.
- ``E2``: train = sim_train + every anchor_train image once; val = sim_val + anchor_val.

Sources:

- sim: generator run dirs (``meta.jsonl``). Group = (master seed, scene index).
- anchor: the COCO anchor YOLO set, split by COCO id hash into anchor_train
  (85 %) and anchor_val (15 %). Each image is its own group. The hash key is
  salted (``anchor_val:<id>``): ``coco_split`` already chose the anchor group
  with ``unit_hash(seed, id)`` in [0.40, 0.80), so the unsalted hash with the
  same seed would never fall below 0.15.
- The COCO sanity set (COCO sanity set, not arena) only gets an evaluation
  list, ``eval/sanity.txt``; it is never in a training or validation list.

Checks (any failure: exit 1 and nothing is written):

- every listed image exists and has a label file at the Ultralytics path
  (``/images/`` replaced by ``/labels/``, suffix ``.txt``);
- train and val share no group; sim train and val runs share no master seed,
  maze seed, layout grid or hand-made layout;
- no listed path lies under a protected directory;
- ``data_tools.leakage.check`` of all training and validation images against
  the protected sets (sha256, COCO id, split rules) finds no error.

Output (``out_dir``, default ``data/yolo_ds``; lists sorted, absolute POSIX
paths, so the manifest hash is stable)::

    e1/ train.txt val.txt data.yaml manifest.json
    e2/ train.txt val.txt data.yaml manifest.json
    eval/ sim_val.txt anchor_val.txt sanity.txt

Usage::

    python -m data_tools.build_dataset                       # paths from configs/dataset.yaml
    python -m data_tools.build_dataset --plan                # counts and shares only, no files
    python -m data_tools.build_dataset --train-run data/sim_raw/dry_train \\
        --val-run data/sim_raw/dry_val --out data/yolo_ds_dry --allow-anchor-share
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cv.config import load_config
from cv.logger import get_logger
from cv.utils.run_info import git_state, sha256_file, utc_now
from data_tools.coco_common import CocoConfig, check_disk, load_split, unit_hash

log = get_logger("build_dataset", log_file=None)

DEFAULT_CONFIG = Path("configs/dataset.yaml")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VARIANTS = {
    "e1": {"train": ("sim_train",), "val": ("sim_val",)},
    "e2": {"train": ("sim_train", "anchor_train"), "val": ("sim_val", "anchor_val")},
}
EXIT_LEAK, EXIT_SHARE = 1, 3
ANCHOR_SALT = "anchor_val:"


class GroupOverlapError(ValueError):
    """Train and val share a group (scene, run seed, layout or image)."""


class ProtectedImageError(ValueError):
    """A protected image (sanity, test_proxy, test_real) is in a list."""


@dataclass(frozen=True)
class Item:
    """One listed image."""

    path: Path
    source: str
    group: str
    boxes: int


def label_path(image: str | Path) -> Path:
    """Label file Ultralytics reads for an image (``img2label_paths``)."""
    p = str(image).replace("\\", "/")
    head, sep, tail = p.rpartition("/images/")
    if not sep:
        raise ValueError(f"no /images/ directory in {image}")
    return Path(f"{head}/labels/{tail}".rsplit(".", 1)[0] + ".txt")


def count_boxes(label: Path) -> int:
    if not label.exists():
        raise FileNotFoundError(f"label file missing: {label}")
    return sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def load_sim_run(run: str | Path, source: str) -> tuple[list[Item], dict[str, Any]]:
    """Items and identity (seed, maze seeds, grids, hand-made layouts) of a generator run."""
    run = Path(run).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("failed_scenes"):
        raise ValueError(f"{run}: failed scenes {manifest['failed_scenes']}; re-run it first")
    items, maze_seeds, grids, handmade = [], set(), set(), set()
    seeds = set()
    for line in (run / "meta.jsonl").read_text(encoding="utf-8").splitlines():
        meta = json.loads(line)
        image = run / meta["image"]
        if not image.exists():
            raise FileNotFoundError(f"image missing: {image}")
        lp = label_path(image)
        if lp != run / meta["label"]:
            raise ValueError(f"label path {lp} differs from meta {meta['label']}")
        layout = meta["layout"]
        seeds.add(int(meta["master_seed"]))
        grids.add("\n".join(layout["grid"]))
        if layout["source"] == "random":
            maze_seeds.add(int(layout["seed"]))
        else:
            handmade.add(layout["name"])
        group = f"sim:{meta['master_seed']}:{meta['scene_index']}"
        items.append(Item(image, source, group, count_boxes(lp)))
    identity = {
        "run": run.as_posix(),
        "master_seeds": sorted(seeds),
        "maze_seeds": maze_seeds,
        "grids": grids,
        "handmade_layouts": sorted(handmade),
        "config_hash": manifest.get("config_hash"),
        "git_commit": manifest.get("git_commit"),
        "meta_sha256": sha256_file(run / "meta.jsonl"),
        "frames": len(items),
    }
    return items, identity


def coco_id(path: Path) -> int:
    return int(path.stem.split("_")[0])


def image_files(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def split_anchor(anchor_dir: str | Path, val_fraction: float, seed: int) -> dict[str, list[Item]]:
    """anchor_train / anchor_val by COCO id hash (stable when images are added or removed)."""
    out: dict[str, list[Item]] = {"anchor_train": [], "anchor_val": []}
    for image in image_files(Path(anchor_dir).resolve() / "images"):
        cid = coco_id(image)
        part = (
            "anchor_val"
            if unit_hash(seed, f"{ANCHOR_SALT}{cid}") < val_fraction
            else "anchor_train"
        )
        out[part].append(Item(image, part, f"coco:{cid}", count_boxes(label_path(image))))
    return out


def eval_items(images_dir: Path, source: str) -> list[Item]:
    return [
        Item(p, source, f"coco:{coco_id(p)}", count_boxes(label_path(p)))
        for p in image_files(images_dir)
    ]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def assert_groups_disjoint(train: list[Item], val: list[Item]) -> None:
    """Raise GroupOverlapError if any group is on both sides."""
    shared = {i.group for i in train} & {i.group for i in val}
    if shared:
        raise GroupOverlapError(f"{len(shared)} groups in train and val, e.g. {sorted(shared)[:5]}")


def assert_runs_disjoint(train_ids: list[dict], val_ids: list[dict]) -> None:
    """Raise GroupOverlapError if sim train and val runs share seeds, grids or hand-made layouts."""

    def union(ids: list[dict], key: str) -> set:
        return set().union(*(set(i[key]) for i in ids)) if ids else set()

    for key in ("master_seeds", "maze_seeds", "grids", "handmade_layouts"):
        shared = union(train_ids, key) & union(val_ids, key)
        if shared:
            shown = sorted(str(s).replace("\n", "/")[:40] for s in shared)[:3]
            raise GroupOverlapError(f"sim train and val runs share {key}: {shown}")


def assert_not_protected(items: list[Item], protected: list[Path]) -> None:
    """Raise ProtectedImageError if a listed path lies under a protected directory."""
    roots = [p.resolve() for p in protected]
    bad = [i.path for i in items for r in roots if i.path.resolve().is_relative_to(r)]
    if bad:
        raise ProtectedImageError(f"{len(bad)} protected images in a list, e.g. {bad[:3]}")


def leakage_result(items: list[Item], protected: list[Path], coco_cfg: CocoConfig) -> dict:
    """``data_tools.leakage.check`` of the listed images against the protected dirs."""
    from data_tools.leakage import check, scan_dir, scan_paths

    split_path = Path(coco_cfg["paths"]["split"])
    split = load_split(split_path) if split_path.exists() else None
    result = check(
        scan_paths(sorted({i.path for i in items})),
        [r for d in protected for r in scan_dir(d)],
        split,
        int(coco_cfg["leakage"]["dhash_max_distance"]),
    )
    result["protected_dirs"] = [p.as_posix() for p in protected]
    result["split"] = split_path.as_posix() if split is not None else None
    return result


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def summary(items: list[Item]) -> dict[str, Any]:
    by_source: dict[str, dict[str, int]] = {}
    for i in items:
        s = by_source.setdefault(i.source, Counter())
        s["images"] += 1
        s["boxes"] += i.boxes
        s["positives" if i.boxes else "negatives"] += 1
    groups = Counter(i.source for i in {(i.source, i.group): i for i in items}.values())
    return {
        "images": len(items),
        "boxes": sum(i.boxes for i in items),
        "by_source": {k: dict(v) for k, v in sorted(by_source.items())},
        "groups_by_source": dict(sorted(groups.items())),
    }


def anchor_share(train: list[Item]) -> dict[str, float | None]:
    anchor = [i for i in train if i.source.startswith("anchor")]
    boxes = sum(i.boxes for i in train)
    return {
        "images": len(anchor) / len(train) if train else None,
        "boxes": sum(i.boxes for i in anchor) / boxes if boxes else None,
    }


def list_text(items: list[Item]) -> str:
    return "".join(f"{p}\n" for p in sorted({i.path.as_posix() for i in items}))


def data_yaml(out: Path, variant: str) -> str:
    return (
        f"# {variant.upper()} training data (data_tools/build_dataset.py). Do not edit by hand.\n"
        "# Simulation validation, not arena. The COCO sanity set is not in these lists.\n"
        f"path: {out.as_posix()}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "names:\n"
        "  0: person\n"
    )


def build(
    config: str | Path = DEFAULT_CONFIG,
    train_runs: list[str] | None = None,
    val_runs: list[str] | None = None,
    out_dir: str | Path | None = None,
    allow_anchor_share: bool = False,
    plan_only: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Build (or with ``plan_only`` only count) the lists; returns (exit code, report)."""
    cfg = load_config(config).to_dict()
    coco_cfg = CocoConfig.from_yaml(cfg["coco_config"])
    out = Path(out_dir or cfg["out_dir"]).resolve()
    protected = [Path(p) for p in cfg["protected"] if Path(p).exists()]

    sources: dict[str, list[Item]] = {"sim_train": [], "sim_val": []}
    run_ids: dict[str, list[dict]] = {"sim_train": [], "sim_val": []}
    for source, runs in (
        ("sim_train", train_runs or cfg["sim"]["train_runs"]),
        ("sim_val", val_runs or cfg["sim"]["val_runs"]),
    ):
        for run in runs:
            items, identity = load_sim_run(run, source)
            sources[source] += items
            run_ids[source].append(identity)
    a = cfg["anchor"]
    sources.update(split_anchor(a["dir"], float(a["val_fraction"]), int(a["seed"])))
    sanity = eval_items(Path(cfg["eval"]["sanity_dir"]).resolve() / "images", "sanity")

    assert_runs_disjoint(run_ids["sim_train"], run_ids["sim_val"])
    variants = {
        v: {part: [i for s in names for i in sources[s]] for part, names in parts.items()}
        for v, parts in VARIANTS.items()
    }
    for v in variants.values():
        assert_groups_disjoint(v["train"], v["val"])
    listed = [i for v in variants.values() for part in v.values() for i in part]
    assert_not_protected(listed, protected)

    share = anchor_share(variants["e2"]["train"])
    report: dict[str, Any] = {
        "variants": {k: {p: summary(x) for p, x in v.items()} for k, v in variants.items()},
        "e2_anchor_share": share,
        "sanity_eval_images": len(sanity),
    }
    lo, hi = (float(x) for x in cfg["anchor_share_band"])
    print(json.dumps(report, indent=2))
    out_of_band = share["images"] is None or not lo <= share["images"] <= hi
    if out_of_band:
        log.warning(f"E2 anchor image share {share['images']} outside [{lo}, {hi}]")
        if not allow_anchor_share:
            log.error("stopping before writing: owner decides (--allow-anchor-share to override)")
            return EXIT_SHARE, report
    if plan_only:
        return 0, report

    check_disk(out, 10 * 2**20, coco_cfg, "build_dataset")
    leak = leakage_result(listed, protected, coco_cfg)
    report["leakage"] = {k: leak[k] for k in ("train_images", "protected_images", "protected_dirs")}
    report["leakage"].update(errors=len(leak["errors"]), warnings=leak["warnings"])
    if leak["errors"]:
        for e in leak["errors"]:
            log.error(f"leakage: {e}")
        log.error(f"{len(leak['errors'])} leakage errors: nothing written")
        return EXIT_LEAK, report

    tmp = out.with_name(out.name + ".building")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    common = {
        "built_utc": utc_now(),
        "git": git_state(),
        "config": Path(config).as_posix(),
        "config_hash": sha256_file(config),
        "allow_anchor_share": allow_anchor_share,
        "anchor": {
            "dir": Path(a["dir"]).as_posix(),
            "manifest_sha256": sha256_file(Path(a["dir"]) / "manifest.json"),
            "val_fraction": float(a["val_fraction"]),
            "seed": int(a["seed"]),
            "method": 'u = first 64 bits of sha256("<seed>:anchor_val:<coco id>") / 2^64 < val_fraction',
        },
        "sim_runs": {
            k: [{x: i[x] for x in i if x not in ("maze_seeds", "grids")} for i in v]
            for k, v in run_ids.items()
        },
        "leakage": report["leakage"],
        "group_disjointness": "asserted (groups, master seeds, maze seeds, grids, layouts)",
    }
    for v, parts in variants.items():
        d = tmp / v
        d.mkdir()
        lists = {}
        for part, items in parts.items():
            (d / f"{part}.txt").write_text(list_text(items), encoding="utf-8", newline="\n")
            lists[part] = sha256_file(d / f"{part}.txt")
        (d / "data.yaml").write_text(data_yaml(out / v, v), encoding="utf-8", newline="\n")
        manifest = dict(common, variant=v, lists_sha256=lists, **report["variants"][v])
        if v == "e2":
            manifest["anchor_share_of_train"] = share
        manifest["steps_note"] = "steps per epoch = ceil(train images / batch)"
        (d / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
    (tmp / "eval").mkdir()
    for name, items in (
        ("sim_val", sources["sim_val"]),
        ("anchor_val", sources["anchor_val"]),
        ("sanity", sanity),
    ):
        (tmp / "eval" / f"{name}.txt").write_text(list_text(items), encoding="utf-8", newline="\n")
    for sub in ("e1", "e2", "eval"):
        shutil.rmtree(out / sub, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        os.replace(tmp / sub, out / sub)
    shutil.rmtree(tmp, ignore_errors=True)
    log.info(f"wrote {out}/e1, e2, eval")
    return 0, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Training list files without copying images.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--train-run", action="append", help="sim train run dir (overrides config)")
    ap.add_argument("--val-run", action="append", help="sim val run dir (overrides config)")
    ap.add_argument("--out", default=None, help="output dir (default: config out_dir)")
    ap.add_argument("--plan", action="store_true", help="print counts and shares, write nothing")
    ap.add_argument(
        "--allow-anchor-share",
        action="store_true",
        help="build although E2's anchor image share is outside anchor_share_band",
    )
    args = ap.parse_args(argv)
    try:
        code, _ = build(
            args.config, args.train_run, args.val_run, args.out, args.allow_anchor_share, args.plan
        )
    except (GroupOverlapError, ProtectedImageError) as exc:
        log.error(f"{type(exc).__name__}: {exc}: nothing written")
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
