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
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from sim.trajectory import TrajectorySampler

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
                f"Consecutive frame gap too large: {dist_xy:.3f}m between frame {i} and {i+1}"
            )
            # Yaw difference should be smooth
            yaw_diff = abs((p1.yaw - p0.yaw + 180.0) % 360.0 - 180.0)
            assert yaw_diff < 35.0, f"Yaw jump {yaw_diff:.1f}° too abrupt between frames"

    def test_sequence_poses_valid(self, sampler: TrajectorySampler) -> None:
        seq = sampler.generate_sequence(num_frames=15, seed=99)
        for p in seq:
            assert sampler.is_pose_valid(p.x, p.y, p.z)
