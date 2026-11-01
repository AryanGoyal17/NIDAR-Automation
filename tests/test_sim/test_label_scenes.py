"""
tests/test_sim/test_label_scenes.py — Heavy label checks on maze scenes
=======================================================================

Slow, so skipped by default. Run with::

    $env:NIDAR_SCENE_TESTS = "1"; python -m pytest -q tests/test_sim/test_label_scenes.py
    NIDAR_SCENE_TESTS=1 python -m pytest -q tests/test_sim/test_label_scenes.py   # bash

Over 6 hand-made layouts, 3 survivors and 5 sampled poses each, at 640x360:
- each SilhouetteRenderer mask equals an independent reference: the same
  survivor rendered in the main scene with every other body moved out of range;
- its outside-frame pixel count equals the reference on a canvas sized from
  the survivor's 3D box;
- visible pixels are a subset of the survivor-alone pixels, and truncation and
  occlusion lie in [0, 1].
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

p = pytest.importorskip("pybullet")

from sim.camera import CameraPose, PinholeCamera  # noqa: E402
from sim.labels import Silhouette, SilhouetteRenderer, decode_segmentation_mask  # noqa: E402
from sim.maze import MazeBuilder, load_layout  # noqa: E402
from sim.survivor import SurvivorPlacer  # noqa: E402
from sim.trajectory import TrajectorySampler  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("NIDAR_SCENE_TESTS") != "1",
    reason="heavy label scene checks: set NIDAR_SCENE_TESTS=1 to run",
)

ARENA_CONFIG = Path("configs/arena.yaml")
LAYOUTS = ["straight_corridor", "l_turn", "t_junction", "dead_end", "two_rooms", "loop"]
SEED = 1
N_POSES = 5
W, H = 640, 360
REFERENCE_MAX_AREA = 6.0  # skip the outside-count reference above this canvas size
PARK_M = 1000.0


@dataclass
class Record:
    layout: str
    pose: CameraPose
    survivor: int
    silhouette: Silhouette
    visible: np.ndarray
    reference_in_frame: np.ndarray
    reference_total: int | None = None
    notes: list[str] = field(default_factory=list)


def _alone_in_main(cam, cid, pose, keep: list[int], margins) -> np.ndarray:
    """Reference: render with every body except ``keep`` moved out of range, then restore."""
    others = [p.getBodyUniqueId(i, physicsClientId=cid) for i in range(p.getNumBodies(cid))]
    others = [b for b in others if b not in keep]
    saved = {b: p.getBasePositionAndOrientation(b, physicsClientId=cid) for b in others}
    try:
        for b, (pos, orn) in saved.items():
            p.resetBasePositionAndOrientation(
                b, [pos[0], pos[1], pos[2] + PARK_M], orn, physicsClientId=cid
            )
        seg = cam.render_segmentation(pose, physics_client=cid, margins=margins)
    finally:
        for b, (pos, orn) in saved.items():
            p.resetBasePositionAndOrientation(b, pos, orn, physicsClientId=cid)
    return np.isin(decode_segmentation_mask(seg), keep)


@pytest.fixture(scope="module")
def records() -> list[Record]:
    cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=W, height=H)
    builder, placer = MazeBuilder(ARENA_CONFIG), SurvivorPlacer(ARENA_CONFIG)
    out: list[Record] = []
    for name in LAYOUTS:
        grid = load_layout(f"sim/layouts/{name}.yaml")
        cid = p.connect(p.DIRECT)
        try:
            maze = builder.build_from_grid(grid, physics_client=cid)
            survivors = placer.place(maze, n=3, seed=SEED)
            sampler = TrajectorySampler(
                grid, arena_config_path=ARENA_CONFIG, survivor_boxes=[s.aabb for s in survivors]
            )
            poses = sampler.sample_poses(N_POSES, seed=SEED)
            with SilhouetteRenderer(cam, survivors, cid, max_canvas_area_factor=4.0) as sr:
                for pose in poses:
                    sils = sr.render(pose)
                    seg = decode_segmentation_mask(cam.render(pose, physics_client=cid).seg_mask)
                    for i, sil in enumerate(sils):
                        if sil is None:
                            continue
                        ids = survivors[i].body_ids
                        rec = Record(
                            layout=name,
                            pose=pose,
                            survivor=i,
                            silhouette=sil,
                            visible=np.isin(seg, ids),
                            reference_in_frame=_alone_in_main(cam, cid, pose, ids, (0, 0, 0, 0)),
                        )
                        rect = cam.project_aabb(pose, survivors[i].aabb, clip=False)
                        if rect is not None and sil.exact:
                            x1, y1, x2, y2 = rect
                            m = (
                                max(0, math.ceil(-x1) + 2),
                                max(0, math.ceil(-y1) + 2),
                                max(0, math.ceil(x2 - W) + 2),
                                max(0, math.ceil(y2 - H) + 2),
                            )
                            if (W + m[0] + m[2]) * (H + m[1] + m[3]) <= REFERENCE_MAX_AREA * W * H:
                                rec.reference_total = int(
                                    _alone_in_main(cam, cid, pose, ids, m).sum()
                                )
                        out.append(rec)
        finally:
            p.disconnect(cid)
    return out


def test_enough_survivors_in_view(records: list[Record]) -> None:
    assert len(records) >= 10, f"only {len(records)} survivors in view"


def test_alone_mask_equals_reference(records: list[Record]) -> None:
    bad = [
        (r.layout, r.survivor, int((r.silhouette.in_frame != r.reference_in_frame).sum()))
        for r in records
        if (r.silhouette.in_frame != r.reference_in_frame).any()
    ]
    assert not bad, f"(layout, survivor, px differing): {bad}"


def test_outside_count_equals_reference(records: list[Record]) -> None:
    checked = [r for r in records if r.reference_total is not None]
    assert len(checked) >= len(records) // 2
    # The reference canvas has other margins, so another projection matrix:
    # float rounding can flip isolated edge pixels in the margins (seen: 1 px
    # of 28994). 0.1 % of the silhouette bounds the truncation error at 0.001.
    bad = [
        (r.layout, r.survivor, r.silhouette.total_pixels, r.reference_total)
        for r in checked
        if abs(r.silhouette.total_pixels - r.reference_total) > 1e-3 * r.reference_total
    ]
    assert not bad, f"(layout, survivor, silhouette total, reference total): {bad}"


def test_visible_subset_of_alone(records: list[Record]) -> None:
    bad = [
        (r.layout, r.survivor, int((r.visible & ~r.silhouette.in_frame).sum()))
        for r in records
        if (r.visible & ~r.silhouette.in_frame).any()
    ]
    assert not bad, f"(layout, survivor, visible px outside the alone mask): {bad}"


def test_fractions_in_range(records: list[Record]) -> None:
    for r in records:
        assert 0.0 <= r.silhouette.truncation_fraction <= 1.0
        assert 0.0 <= r.silhouette.occlusion_fraction(r.visible) <= 1.0
