"""
data_tools/coco_split.py — Seeded split of COCO val2017 image ids
=================================================================

COCO sanity set, not arena. Puts every val2017 image id into exactly one of
three disjoint groups by a seeded hash of the id (``configs/coco.yaml split``):

- ``cutout_source``: person cutouts for the compositor (never in anchor or sanity);
- ``anchor``: real photos mixed into training;
- ``sanity``: held-out real photos, never trained on.

The group of an id depends only on the seed and the id, so the split is stable
when images are added or removed. COCO val2017 has one id per photo, so two
crops of the same photo cannot fall into different groups.

Writes ``data/coco/split.json`` and prints, per group: images, images with a
person annotation, non-crowd person boxes and person crowd regions.

Usage::

    python -m data_tools.coco_split
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from cv.logger import get_logger
from data_tools.coco_common import (
    GROUPS,
    CocoConfig,
    CocoImage,
    atomic_write_json,
    git_commit,
    load_index,
    split_group,
)

log = get_logger("coco_split", log_file=None)


def make_split(
    image_ids: list[int], seed: int, fractions: dict[str, float]
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {g: [] for g in GROUPS}
    for image_id in sorted(image_ids):
        groups[split_group(image_id, seed, fractions)].append(image_id)
    return groups


def group_counts(index: dict[int, CocoImage], groups: dict[str, list[int]]) -> dict[str, Any]:
    out = {}
    for g, ids in groups.items():
        images = [index[i] for i in ids]
        out[g] = {
            "images": len(images),
            "images_with_persons": sum(1 for im in images if im.persons),
            "person_boxes": sum(len(im.people) for im in images),
            "person_crowd_regions": sum(len(im.crowd) for im in images),
        }
    return out


def run(cfg: CocoConfig) -> dict[str, Any]:
    paths = cfg["paths"]
    index = load_index(paths["instances"])
    seed = int(cfg["split"]["seed"])
    fractions = {g: float(cfg["split"]["fractions"][g]) for g in GROUPS}
    groups = make_split(list(index), seed, fractions)
    counts = group_counts(index, groups)
    record = {
        "seed": seed,
        "fractions": fractions,
        "method": 'u = first 64 bits of sha256("<seed>:<image_id>") / 2^64, cumulative fractions',
        "config_hash": cfg.hash(),
        "git_commit": git_commit(),
        "counts": counts,
        "groups": groups,
        "note": "COCO sanity set, not arena",
    }
    atomic_write_json(Path(paths["split"]), record)
    log.info(f"wrote {paths['split']}")
    print(f"{'group':<14}{'images':>8}{'w/ persons':>12}{'person boxes':>14}{'crowd regions':>15}")
    for g in GROUPS:
        c = counts[g]
        print(
            f"{g:<14}{c['images']:>8}{c['images_with_persons']:>12}"
            f"{c['person_boxes']:>14}{c['person_crowd_regions']:>15}"
        )
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="configs/coco.yaml")
    args = ap.parse_args(argv)
    run(CocoConfig.from_yaml(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
