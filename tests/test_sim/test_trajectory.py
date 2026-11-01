"""
tests/test_sim/test_trajectory.py — Unit tests for sim.trajectory TrajectorySampler
==================================================================================

Validates:
1. Sampled poses are within free corridor boundaries.
2. Collision clearance from walls is strictly maintained.
3. Altitude z matches range around arena.yaml camera height.
4. Pitch is within specified downward tilt bounds.
5. Yaw is aligned along corridor axes (close to 0, 90, 180, or 270 deg).
6. Consecutive frames in generated sequences are spatially smooth and adjacent.
7. Survivor checks: clearance and projected box size from arena.yaml, rejection
   reasons, counts in the log and in the error, determinism by seed.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from sim.camera import CameraPose
from sim.trajectory import (
    REJECT_BOX_TOO_LARGE,
    REJECT_SURVIVOR_CLEARANCE,
    REJECT_WALL,
    PoseSamplingError,
    TrajectorySampler,
    point_to_aabb_distance,
    pose_key,
)

SAMPLE_LAYOUT = Path("sim/layouts/loop.yaml")
ARENA_CONFIG = Path("configs/arena.yaml")


@pytest.fixture
def sampler() -> TrajectorySampler:
    return TrajectorySampler(SAMPLE_LAYOUT, arena_config_path=ARENA_CONFIG)


class TestPoseSampling:
    def test_sampled_poses_are_collision_free(self, sampler: TrajectorySampler) -> None:
        poses = sampler.sample_poses(n=25, seed=123)
        assert len(poses) == 25
        for p in poses:
            assert sampler.is_pose_valid(p.x, p.y, p.z), (
                f"Pose ({p.x}, {p.y}, {p.z}) failed collision check!"
            )

    def test_altitude_within_bounds(self, sampler: TrajectorySampler) -> None:
        poses = sampler.sample_poses(
            n=20,
            seed=42,
            height_range=(0.7, 0.9),
        )
        for p in poses:
            assert 0.68 <= p.z <= 0.92

    def test_pitch_within_bounds(self, sampler: TrajectorySampler) -> None:
        poses = sampler.sample_poses(
            n=20,
            seed=42,
            pitch_range=(-35.0, -25.0),
        )
        for p in poses:
            assert -36.0 <= p.pitch <= -24.0

    def test_yaw_alignment_along_cardinal_axes(self, sampler: TrajectorySampler) -> None:
        """Yaw should be close to 0, 90, 180, or 270 deg (within ~20 deg jitter)."""
        poses = sampler.sample_poses(n=30, seed=777)
        cardinals = [0.0, 90.0, 180.0, 270.0, 360.0]
        for p in poses:
            min_diff = min(abs(p.yaw - c) for c in cardinals)
            assert min_diff < 25.0, f"Yaw {p.yaw}° is not aligned with corridors"


class TestSequenceGeneration:
    def test_sequence_length(self, sampler: TrajectorySampler) -> None:
        seq = sampler.generate_sequence(num_frames=20, seed=42)
        assert len(seq) == 20

    def test_sequence_spatial_continuity(self, sampler: TrajectorySampler) -> None:
        """Consecutive frames must have smooth spatial transitions without teleporting."""
        step = 0.08
        seq = sampler.generate_sequence(num_frames=25, step_m=step, seed=42)
        for i in range(len(seq) - 1):
            p0 = seq[i]
            p1 = seq[i + 1]
            dist_xy = math.hypot(p1.x - p0.x, p1.y - p0.y)
            assert dist_xy < (step * 2.5), (
                f"Consecutive frame gap too large: {dist_xy:.3f}m between frame {i} and {i + 1}"
            )
            # Yaw difference should be smooth
            yaw_diff = abs((p1.yaw - p0.yaw + 180.0) % 360.0 - 180.0)
            assert yaw_diff < 35.0, f"Yaw jump {yaw_diff:.1f}° too abrupt between frames"

    def test_sequence_poses_valid(self, sampler: TrajectorySampler) -> None:
        seq = sampler.generate_sequence(num_frames=15, seed=99)
        for p in seq:
            assert sampler.is_pose_valid(p.x, p.y, p.z)


# --------------------------------------------------------------------------- #
# Survivor checks and sample_poses contract
# --------------------------------------------------------------------------- #
# A low, wide box like a lying survivor, in the loop layout's top corridor.
LYING_BOX = ((3.4, 1.6, 0.0), (4.6, 2.9, 0.36))


@pytest.fixture
def survivor_sampler() -> TrajectorySampler:
    return TrajectorySampler(
        SAMPLE_LAYOUT, arena_config_path=ARENA_CONFIG, survivor_boxes=[LYING_BOX]
    )


class TestSurvivorChecks:
    def test_values_come_from_arena_yaml(self, sampler: TrajectorySampler) -> None:
        from cv.config import load_config

        cfg = load_config(ARENA_CONFIG).trajectory
        assert sampler.survivor_clearance_m == float(cfg.survivor_clearance_m)
        assert sampler.max_survivor_box_fraction == float(cfg.max_survivor_box_fraction)

    def test_rejection_reasons(self, survivor_sampler: TrajectorySampler) -> None:
        s = survivor_sampler
        # Inside a wall cell.
        assert s.pose_rejection(CameraPose(x=0.2, y=0.2, z=0.8)) == REJECT_WALL
        # Above the box, 0.2 m from its top.
        assert s.pose_rejection(CameraPose(x=4.0, y=2.25, z=0.56)) == REJECT_SURVIVOR_CLEARANCE
        # 0.4 m from the box, looking straight at it: it fills most of the image.
        close = CameraPose(x=3.0, y=2.25, z=0.6, yaw=0.0, pitch=-45.0)
        assert s.position_rejection(close.x, close.y, close.z) is None
        assert s.pose_rejection(close) == REJECT_BOX_TOO_LARGE
        # Far away, looking along the corridor.
        assert s.pose_rejection(CameraPose(x=9.75, y=2.25, z=0.8, yaw=180.0)) is None

    def test_sampled_poses_respect_survivor(self, survivor_sampler: TrajectorySampler) -> None:
        s = survivor_sampler
        poses = s.sample_poses(n=200, seed=5)
        assert len(poses) == 200
        assert len({pose_key(p) for p in poses}) == 200
        for p in poses:
            d = point_to_aabb_distance((p.x, p.y, p.z), LYING_BOX)
            assert d >= s.survivor_clearance_m
            assert s.camera.projected_box_fraction(p, LYING_BOX) <= s.max_survivor_box_fraction

    def test_impossible_request_reports_counts(self) -> None:
        # One box covering the whole maze: every candidate fails the clearance.
        everything = ((-1.0, -1.0, -1.0), (50.0, 50.0, 5.0))
        s = TrajectorySampler(
            SAMPLE_LAYOUT, arena_config_path=ARENA_CONFIG, survivor_boxes=[everything]
        )
        with pytest.raises(PoseSamplingError) as err:
            s.sample_poses(n=5, seed=0, max_attempts_per_pose=10)
        msg = str(err.value)
        assert "found 0 of 5" in msg
        assert "wall=0" in msg and "survivor_clearance=50" in msg and "box_too_large=0" in msg

    def test_counts_logged_at_info_every_call(
        self, survivor_sampler: TrajectorySampler, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="nidar.trajectory")
        survivor_sampler.sample_poses(n=10, seed=1)
        survivor_sampler.sample_poses(n=10, seed=2)
        lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        lines = [m for m in lines if m.startswith("sample_poses:")]
        assert len(lines) == 2
        for m in lines:
            for reason in ("wall=", "survivor_clearance=", "box_too_large=", "duplicate="):
                assert reason in m


class TestDeterminism:
    def test_sample_poses_same_seed_same_poses(self, survivor_sampler: TrajectorySampler) -> None:
        other = TrajectorySampler(
            SAMPLE_LAYOUT, arena_config_path=ARENA_CONFIG, survivor_boxes=[LYING_BOX]
        )
        a = survivor_sampler.sample_poses(n=50, seed=11)
        b = other.sample_poses(n=50, seed=11)
        assert [p.as_tuple() for p in a] == [p.as_tuple() for p in b]
        c = other.sample_poses(n=50, seed=12)
        assert [p.as_tuple() for p in a] != [p.as_tuple() for p in c]
