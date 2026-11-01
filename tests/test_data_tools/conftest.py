"""
tests/test_data_tools/conftest.py — Synthetic COCO fixture for the COCO tool tests
==================================================================================

``coco``: a tiny val2017.zip of generated JPEGs, hand-made instances and
captions JSON, and a CocoConfig pointing at them. The real download is never used.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from data_tools.coco_common import CocoConfig, SizeLimits, load_index
from tests.test_data_tools.coco_synth import REPO, SPEC, jpeg


@pytest.fixture
def coco(tmp_path: Path) -> dict[str, Any]:
    """Synthetic val2017.zip, instances and captions JSON, and a config pointing at them."""
    yaml = pytest.importorskip("yaml")
    root = tmp_path / "coco"
    (root / "annotations").mkdir(parents=True)
    images, anns, caps = [], [], []
    zpath = root / "val2017.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for image_id, (w, h, persons, captions, fill) in SPEC.items():
            name = f"{image_id:012d}.jpg"
            zf.writestr(f"val2017/{name}", jpeg(w, h, fill, image_id))
            images.append({"id": image_id, "file_name": name, "width": w, "height": h})
            for a in persons:
                anns.append(dict(a, image_id=image_id, area=a["bbox"][2] * a["bbox"][3]))
            caps += [
                {"id": 100 * image_id + k, "image_id": image_id, "caption": c}
                for k, c in enumerate(captions)
            ]
    # A crowd region of another category must not exclude image 1
    anns.append({"id": 99, "image_id": 1, "category_id": 18, "bbox": [0, 0, 50, 50], "iscrowd": 1})
    categories = [{"id": 1, "name": "person"}, {"id": 18, "name": "dog"}]
    (root / "annotations" / "instances_val2017.json").write_text(
        json.dumps({"images": images, "annotations": anns, "categories": categories})
    )
    (root / "annotations" / "captions_val2017.json").write_text(json.dumps({"annotations": caps}))

    raw = yaml.safe_load((REPO / "configs" / "coco.yaml").read_text(encoding="utf-8"))
    raw["arena_config"] = str(REPO / "configs" / "arena.yaml")
    raw["paths"] = {
        "root": str(root),
        "images_zip": str(zpath),
        "instances": str(root / "annotations" / "instances_val2017.json"),
        "captions": str(root / "annotations" / "captions_val2017.json"),
        "split": str(root / "split.json"),
        "person_sets": str(tmp_path / "coco_person"),
        "cutouts": str(tmp_path / "coco_cutouts"),
    }
    raw["person_sets"]["negatives"] = {"anchor": 2, "sanity": 1}
    cfg = CocoConfig(raw=raw, limits=SizeLimits.from_arena(raw["arena_config"]))
    index = load_index(raw["paths"]["instances"], raw["paths"]["captions"])
    return {"cfg": cfg, "index": index, "zip": zpath, "tmp": tmp_path}
