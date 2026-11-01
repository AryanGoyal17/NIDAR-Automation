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
8. generate_sequence: stops at the clearance boundary and rotates in place, never
   walks back, skips with a logged reason below the minimum, deterministic.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

# sim.camera and sim.maze import pybullet.
pytest.importorskip("pybullet")

from sim.camera import CameraPose  # noqa: E402
from sim.trajectory import (  # noqa: E402
    REJECT_BOX_TOO_LARGE,
    REJECT_SURVIVOR_CLEARANCE,
    REJECT_WALL,
    PoseSamplingError,
    SequenceSkipped,
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


# --------------------------------------------------------------------------- #
# generate_sequence contract
# --------------------------------------------------------------------------- #
CORRIDOR = ["########", "#S.....#", "########"]
# A standing-height box in the last corridor cell (centre x = 9.75 m, y = 2.25 m).
STANDING_BOX = ((9.5, 2.0, 0.0), (10.0, 2.5, 1.75))


def _corridor(boxes: list | None = None) -> TrajectorySampler:
    return TrajectorySampler(CORRIDOR, arena_config_path=ARENA_CONFIG, survivor_boxes=boxes)


class TestSequenceContract:
    def test_stops_at_survivor_and_rotates(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="nidar.trajectory")
        s = _corridor([STANDING_BOX])
        seq = s.generate_sequence(num_frames=200, seed=3, start_cell=(1, 1))
        assert any("ended: blocked (" in r.getMessage() for r in caplog.records)
        # Trailing in-place frames: same position, yaw stepping by rotate_yaw_step_deg.
        last = seq[-1]
        rot = [q for q in seq if (q.x, q.y) == (last.x, last.y)]
        assert 2 <= len(rot) <= s.rotate_max_frames + 1
        assert seq[-len(rot) :] == rot
        for a, b in zip(rot, rot[1:], strict=False):
            step = abs((b.yaw - a.yaw + 180.0) % 360.0 - 180.0)
            assert step == pytest.approx(s.rotate_yaw_step_deg)
        # Stopped at the survivor boundary: one more step along the corridor
        # (same orientation) is rejected by a survivor rule.
        stop = rot[0]
        step = CameraPose(stop.x + 0.08, stop.y, stop.z, stop.yaw, stop.pitch, stop.roll)
        assert s.pose_rejection(step) in (REJECT_SURVIVOR_CLEARANCE, REJECT_BOX_TOO_LARGE)
        for q in seq:
            assert s.pose_rejection(q) is None

    def test_stops_within_one_step_of_clearance(self) -> None:
        s = _corridor([STANDING_BOX])
        s.max_survivor_box_fraction = 1.0  # isolate the clearance rule
        s.rotate_in_place = False
        seq = s.generate_sequence(num_frames=200, seed=3, start_cell=(1, 1))
        last = seq[-1]
        d = point_to_aabb_distance((last.x, last.y, last.z), STANDING_BOX)
        assert s.survivor_clearance_m <= d < s.survivor_clearance_m + 0.08 + 1e-9

    def test_no_rotation_when_disabled(self) -> None:
        s = _corridor([STANDING_BOX])
        s.rotate_in_place = False
        seq = s.generate_sequence(num_frames=200, seed=3, start_cell=(1, 1))
        assert (seq[-1].x, seq[-1].y) != (seq[-2].x, seq[-2].y)

    def test_ends_without_walking_back(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="nidar.trajectory")
        s = _corridor()
        seq = s.generate_sequence(num_frames=500, seed=0, start_cell=(1, 1))
        assert any("ended: no unvisited continuation" in r.getMessage() for r in caplog.records)
        xs = [q.x for q in seq]
        assert xs == sorted(xs) and len(set(xs)) == len(xs)
        assert len(seq) < 500

    def test_too_short_is_skipped_with_logged_reason(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="nidar.trajectory")
        s = _corridor([STANDING_BOX])
        with pytest.raises(SequenceSkipped) as err:
            s.generate_sequence(num_frames=500, seed=3, start_cell=(1, 1), min_frames=400)
        assert err.value.reason
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(err.value.reason in m for m in warnings)

    def test_never_one_frame(self) -> None:
        # Start right next to the survivor: the default minimum still holds or it skips.
        s = _corridor([STANDING_BOX])
        for seed in range(10):
            try:
                seq = s.generate_sequence(num_frames=60, seed=seed)
            except SequenceSkipped:
                continue
            assert len(seq) >= s.min_sequence_frames

    def test_values_come_from_arena_yaml(self, sampler: TrajectorySampler) -> None:
        from cv.config import load_config

        cfg = load_config(ARENA_CONFIG).trajectory
        assert sampler.min_sequence_frames == int(cfg.min_sequence_frames)
        assert sampler.rotate_in_place == bool(cfg.rotate_in_place)
        assert sampler.rotate_yaw_step_deg == float(cfg.rotate_yaw_step_deg)
        assert sampler.rotate_max_frames == int(cfg.rotate_max_frames)


class TestSequenceDeterminism:
    def test_same_seed_same_sequence(self, survivor_sampler: TrajectorySampler) -> None:
        other = TrajectorySampler(
            SAMPLE_LAYOUT, arena_config_path=ARENA_CONFIG, survivor_boxes=[LYING_BOX]
        )
        for seed in (0, 7, 42):
            a = survivor_sampler.generate_sequence(num_frames=60, seed=seed)
            b = other.generate_sequence(num_frames=60, seed=seed)
            assert [p.as_tuple() for p in a] == [p.as_tuple() for p in b]
        a = survivor_sampler.generate_sequence(num_frames=60, seed=1)
        c = survivor_sampler.generate_sequence(num_frames=60, seed=2)
        assert [p.as_tuple() for p in a] != [p.as_tuple() for p in c]
