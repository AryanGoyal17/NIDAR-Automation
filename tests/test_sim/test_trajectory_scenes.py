"""
tests/test_sim/test_trajectory_scenes.py — TrajectorySampler over many survivor scenes
=====================================================================================

8 layouts (6 hand-made + 2 generated) x 20 seeds, 3 survivors placed per scene
with SurvivorPlacer (seed = scene seed), survivor boxes passed to the sampler.

Validates:
1. Survivors are never on the start cell.
2. sample_poses returns the requested count with no duplicate poses.
3. No pose closer than the survivor clearance to any survivor box.
4. No eye inside a survivor box or inside any survivor body part's box.
5. No pose where a survivor's projected box exceeds the configured fraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pybullet as p
import pytest

from sim.camera import CameraPose
from sim.layout_gen import MazeGenerator
from sim.maze import MazeBuilder, load_layout
from sim.survivor import SurvivorPlacer
from sim.trajectory import TrajectorySampler, point_to_aabb_distance, pose_key

ARENA_CONFIG = "configs/arena.yaml"
HANDMADE = ["straight_corridor", "l_turn", "t_junction", "dead_end", "two_rooms", "loop"]
SEEDS = range(20)
N_POSES = 60
N_SURVIVORS = 3


@dataclass
class Scene:
    name: str
    seed: int
    start_cell: tuple[int, int] | None
    survivor_cells: list[tuple[int, int]]
    survivor_boxes: list
    part_boxes: list
    sampler: TrajectorySampler  # shared per layout; survivor_boxes set to the last seed
    poses: list[CameraPose] = field(default_factory=list)


def _layouts() -> list[tuple[str, list[str]]]:
    out = [(n, load_layout(f"sim/layouts/{n}.yaml")) for n in HANDMADE]
    gen = MazeGenerator(ARENA_CONFIG)
    out.append(("gen_42", gen.generate(11, 11, seed=42).grid))
    out.append(("gen_101", gen.generate(13, 13, seed=101).grid))
    return out


@pytest.fixture(scope="module")
def scenes() -> list[Scene]:
    builder, placer = MazeBuilder(ARENA_CONFIG), SurvivorPlacer(ARENA_CONFIG)
    out: list[Scene] = []
    for name, grid in _layouts():
        # One maze per layout; placement depends only on the seed, so survivors
        # are placed and removed per seed.
        cid = p.connect(p.DIRECT)
        sampler = TrajectorySampler(grid, arena_config_path=ARENA_CONFIG)
        try:
            maze = builder.build_from_grid(grid, physics_client=cid)
            for seed in SEEDS:
                survivors = placer.place(maze, n=N_SURVIVORS, seed=seed)
                parts = [p.getAABB(b, physicsClientId=cid) for s in survivors for b in s.body_ids]
                placer.remove_all(survivors, cid)
                boxes = [s.aabb for s in survivors]
                sampler.set_survivor_boxes(boxes)
                scene = Scene(
                    name=name,
                    seed=seed,
                    start_cell=maze.start_cell,
                    survivor_cells=[s.cell for s in survivors],
                    survivor_boxes=boxes,
                    part_boxes=parts,
                    sampler=sampler,
                )
                scene.poses = sampler.sample_poses(N_POSES, seed=seed)
                out.append(scene)
        finally:
            p.disconnect(cid)
    return out


def test_scene_count(scenes: list[Scene]) -> None:
    assert len(scenes) == 8 * len(SEEDS)
    assert all(len(s.survivor_boxes) == N_SURVIVORS for s in scenes)


def test_survivors_never_on_start_cell(scenes: list[Scene]) -> None:
    for s in scenes:
        assert s.start_cell is not None
        assert s.start_cell not in s.survivor_cells, f"{s.name} seed {s.seed}"


def test_sample_poses_count_and_unique(scenes: list[Scene]) -> None:
    for s in scenes:
        assert len(s.poses) == N_POSES, f"{s.name} seed {s.seed}"
        assert len({pose_key(q) for q in s.poses}) == N_POSES, f"{s.name} seed {s.seed}"


def _check_clear(scene: Scene, poses: list[CameraPose]) -> None:
    clearance = scene.sampler.survivor_clearance_m
    for q in poses:
        eye = (q.x, q.y, q.z)
        for box in scene.survivor_boxes:
            assert point_to_aabb_distance(eye, box) >= clearance, f"{scene.name} seed {scene.seed}"
        for box in scene.part_boxes:
            assert point_to_aabb_distance(eye, box) > 0.0, f"{scene.name} seed {scene.seed}"


def _check_box_fraction(scene: Scene, poses: list[CameraPose]) -> None:
    sampler = scene.sampler
    for q in poses:
        fractions = sampler.camera.projected_box_fractions(q, scene.survivor_boxes)
        assert max(fractions) <= sampler.max_survivor_box_fraction, (
            f"{scene.name} seed {scene.seed}"
        )


def test_sample_poses_keep_clearance(scenes: list[Scene]) -> None:
    for s in scenes:
        _check_clear(s, s.poses)


def test_sample_poses_box_fraction(scenes: list[Scene]) -> None:
    for s in scenes:
        _check_box_fraction(s, s.poses)
