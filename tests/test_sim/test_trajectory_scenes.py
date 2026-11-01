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
6. generate_sequence: every sequence has at least min_sequence_frames frames or
   was skipped with a reason that was logged; consecutive frames are adjacent;
   no cell is re-entered; repeated (x, y, yaw) at most 2 % of all frames;
   checks 3 to 5 hold for every sequence pose.

Heavy (builds 160 scenes), so it is skipped in the default ``pytest`` run.
Run it on its own:

    $env:NIDAR_SCENE_TESTS = "1"; python -m pytest -q tests/test_sim/test_trajectory_scenes.py
    NIDAR_SCENE_TESTS=1 python -m pytest -q tests/test_sim/test_trajectory_scenes.py   # bash
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import pybullet as p
import pytest

from sim.camera import CameraPose
from sim.layout_gen import MazeGenerator
from sim.maze import MazeBuilder, load_layout
from sim.survivor import SurvivorPlacer
from sim.trajectory import SequenceSkipped, TrajectorySampler, point_to_aabb_distance, pose_key

pytestmark = pytest.mark.skipif(
    os.environ.get("NIDAR_SCENE_TESTS") != "1",
    reason="heavy 8 layouts x 20 seeds checks: set NIDAR_SCENE_TESTS=1 to run",
)

ARENA_CONFIG = "configs/arena.yaml"
HANDMADE = ["straight_corridor", "l_turn", "t_junction", "dead_end", "two_rooms", "loop"]
SEEDS = range(20)
N_POSES = 60
N_SURVIVORS = 3
N_FRAMES = 60
STEP_M = 0.08


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
    sequence: list[CameraPose] = field(default_factory=list)
    skip_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _layouts() -> list[tuple[str, list[str]]]:
    out = [(n, load_layout(f"sim/layouts/{n}.yaml")) for n in HANDMADE]
    gen = MazeGenerator(ARENA_CONFIG)
    out.append(("gen_42", gen.generate(11, 11, seed=42).grid))
    out.append(("gen_101", gen.generate(13, 13, seed=101).grid))
    return out


@pytest.fixture(scope="module")
def scenes() -> list[Scene]:
    builder, placer = MazeBuilder(ARENA_CONFIG), SurvivorPlacer(ARENA_CONFIG)
    records = _Records()
    logging.getLogger("nidar.trajectory").addHandler(records)
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
                records.messages.clear()
                try:
                    scene.sequence = sampler.generate_sequence(
                        num_frames=N_FRAMES, step_m=STEP_M, seed=seed
                    )
                except SequenceSkipped as skip:
                    scene.skip_reason = skip.reason
                scene.warnings = list(records.messages)
                out.append(scene)
        finally:
            p.disconnect(cid)
    logging.getLogger("nidar.trajectory").removeHandler(records)
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


# --------------------------------------------------------------------------- #
# generate_sequence
# --------------------------------------------------------------------------- #
def test_sequence_min_length_or_logged_skip(scenes: list[Scene]) -> None:
    for s in scenes:
        if s.skip_reason is not None:
            assert s.skip_reason and not s.sequence
            assert any(s.skip_reason in m for m in s.warnings), f"{s.name} seed {s.seed}"
        else:
            assert s.sampler.min_sequence_frames <= len(s.sequence) <= N_FRAMES, (
                f"{s.name} seed {s.seed}: {len(s.sequence)} frames"
            )


def test_sequence_frames_adjacent(scenes: list[Scene]) -> None:
    for s in scenes:
        for a, b in zip(s.sequence, s.sequence[1:], strict=False):
            assert math.hypot(b.x - a.x, b.y - a.y) <= STEP_M + 1e-9, f"{s.name} seed {s.seed}"


def test_sequence_never_reenters_a_cell(scenes: list[Scene]) -> None:
    for s in scenes:
        cs = s.sampler.cell_size
        cells = [(int(q.y // cs), int(q.x // cs)) for q in s.sequence]
        runs = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
        assert len(runs) == len(set(runs)), f"{s.name} seed {s.seed}: {runs}"


def test_sequence_repeats_at_most_two_percent(scenes: list[Scene]) -> None:
    frames = sum(len(s.sequence) for s in scenes)
    repeats = sum(len(s.sequence) - len({pose_key(q) for q in s.sequence}) for s in scenes)
    assert frames > 0
    assert repeats <= 0.02 * frames, f"{repeats} repeated of {frames}"


def test_sequence_keeps_clearance(scenes: list[Scene]) -> None:
    for s in scenes:
        _check_clear(s, s.sequence)


def test_sequence_box_fraction(scenes: list[Scene]) -> None:
    for s in scenes:
        _check_box_fraction(s, s.sequence)
