"""
tests/test_sim/test_survivor.py — Unit tests for sim.survivor
===============================================================

Validates:
1. Deterministic placement — same seed → identical results.
2. Count bounds — respects min/max and exact n.
3. Posture coverage — all postures produce at least one body ID.
4. Scale jitter — scale stays within ±15 % of nominal.
5. Placement validity — survivors land on free cells only.
6. Body ID return — every placed survivor has body IDs.
7. Removal — remove_all clears all bodies without error.
"""

from __future__ import annotations

import math
from pathlib import Path

import pybullet as p
import pytest

from sim.maze import MazeBuilder, MazeResult, FREE_CHARS
from sim.survivor import (
    SurvivorInfo,
    SurvivorPlacer,
    POSTURES,
    _POSTURE_BUILDERS,
)

# Shared inline grid for testing.
INLINE_GRID = [
    "###########",
    "#S........#",
    "#.........#",
    "#.........#",
    "#.........#",
    "#.........#",
    "#.........#",
    "#.........#",
    "#.........#",
    "###########",
]


@pytest.fixture(scope="module")
def maze() -> MazeResult:
    """Build a test maze once for all tests in this module."""
    builder = MazeBuilder()
    result = builder.build_from_grid(INLINE_GRID)
    yield result
    p.disconnect(result.physics_client)


@pytest.fixture
def placer() -> SurvivorPlacer:
    return SurvivorPlacer()


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
class TestDeterminism:
    """Same seed must produce identical survivors."""

    def test_same_seed_identical(self, maze: MazeResult, placer: SurvivorPlacer) -> None:
        s1 = placer.place(maze, seed=42)
        placer.remove_all(s1, maze.physics_client)

        s2 = placer.place(maze, seed=42)
        placer.remove_all(s2, maze.physics_client)

        assert len(s1) == len(s2), "Count mismatch"
        for a, b in zip(s1, s2):
            assert a.posture == b.posture
            assert a.cell == b.cell
            assert abs(a.yaw_deg - b.yaw_deg) < 1e-9
            assert abs(a.scale - b.scale) < 1e-9

    def test_different_seeds_differ(self, maze: MazeResult, placer: SurvivorPlacer) -> None:
        s1 = placer.place(maze, n=2, seed=42)
        placer.remove_all(s1, maze.physics_client)

        s2 = placer.place(maze, n=2, seed=99)
        placer.remove_all(s2, maze.physics_client)

        # At least one attribute should differ between different seeds.
        any_diff = any(
            a.posture != b.posture or a.cell != b.cell
            or abs(a.yaw_deg - b.yaw_deg) > 0.01
            for a, b in zip(s1, s2)
        )
        assert any_diff, "Different seeds produced identical survivors"


# --------------------------------------------------------------------------- #
# Count bounds
# --------------------------------------------------------------------------- #
class TestCounts:
    """Survivor count respects limits."""

    def test_zero_survivors(self, maze: MazeResult, placer: SurvivorPlacer) -> None:
        s = placer.place(maze, n=0, seed=1)
        assert len(s) == 0

    def test_exact_count(self, maze: MazeResult, placer: SurvivorPlacer) -> None:
        for n in (1, 2, 3):
            s = placer.place(maze, n=n, seed=10 + n)
            placer.remove_all(s, maze.physics_client)
            assert len(s) == n, f"Expected {n} survivors, got {len(s)}"

    def test_random_count_in_bounds(self, maze: MazeResult) -> None:
        placer = SurvivorPlacer(min_survivors=1, max_survivors=3)
        for seed in range(20):
            s = placer.place(maze, seed=seed)
            placer.remove_all(s, maze.physics_client)
            assert 1 <= len(s) <= 3, f"Count {len(s)} out of [1,3]"


# --------------------------------------------------------------------------- #
# Posture coverage
# --------------------------------------------------------------------------- #
class TestPostures:
    """Every supported posture produces valid bodies."""

    @pytest.mark.parametrize("posture", POSTURES)
    def test_posture_creates_bodies(self, maze: MazeResult, posture: str) -> None:
        """Force each posture and verify body_ids are produced."""
        # Use direct builder to isolate posture testing.
        builder_fn = _POSTURE_BUILDERS[posture]
        cx, cy = 3.0, 3.0
        yaw_rad = 0.0
        h, sw, td = 1.75, 0.45, 0.25
        skin = (0.8, 0.7, 0.6)
        cloth = (0.3, 0.3, 0.3)

        ids = builder_fn(
            maze.physics_client, cx, cy, yaw_rad,
            h, sw, td, skin, cloth,
        )
        assert isinstance(ids, list)
        assert len(ids) >= 2, f"{posture} should have at least 2 body parts"

        # Clean up.
        for bid in ids:
            p.removeBody(bid, physicsClientId=maze.physics_client)


# --------------------------------------------------------------------------- #
# Scale jitter
# --------------------------------------------------------------------------- #
class TestScale:
    """Scale factor remains within configured jitter range."""

    def test_scale_bounds(self, maze: MazeResult, placer: SurvivorPlacer) -> None:
        for seed in range(50):
            survivors = placer.place(maze, n=1, seed=seed)
            for s in survivors:
                assert 0.85 <= s.scale <= 1.15, (
                    f"Scale {s.scale} out of [0.85, 1.15]"
                )
            placer.remove_all(survivors, maze.physics_client)


# --------------------------------------------------------------------------- #
# Placement validity
# --------------------------------------------------------------------------- #
class TestPlacement:
    """Survivors must land on free cells."""

    def test_cells_are_free(self, maze: MazeResult, placer: SurvivorPlacer) -> None:
        for seed in range(30):
            survivors = placer.place(maze, n=3, seed=seed)
            for s in survivors:
                r, c = s.cell
                assert maze.grid[r][c] in FREE_CHARS, (
                    f"Survivor at ({r},{c}) is on a wall cell"
                )
            placer.remove_all(survivors, maze.physics_client)

    def test_no_overlapping_cells(self, maze: MazeResult, placer: SurvivorPlacer) -> None:
        survivors = placer.place(maze, n=3, seed=42)
        cells = [s.cell for s in survivors]
        assert len(cells) == len(set(cells)), "Survivors overlap on the same cell"
        placer.remove_all(survivors, maze.physics_client)


# --------------------------------------------------------------------------- #
# Body IDs
# --------------------------------------------------------------------------- #
class TestBodyIDs:
    """Every survivor has non-empty body_ids."""

    def test_body_ids_not_empty(self, maze: MazeResult, placer: SurvivorPlacer) -> None:
        survivors = placer.place(maze, n=3, seed=42)
        for s in survivors:
            assert len(s.body_ids) > 0
        placer.remove_all(survivors, maze.physics_client)


# --------------------------------------------------------------------------- #
# Removal
# --------------------------------------------------------------------------- #
class TestRemoval:
    """remove_all should run without error."""

    def test_remove_all(self, maze: MazeResult, placer: SurvivorPlacer) -> None:
        survivors = placer.place(maze, n=3, seed=42)
        # Should not raise.
        placer.remove_all(survivors, maze.physics_client)

    def test_remove_individual(self, maze: MazeResult, placer: SurvivorPlacer) -> None:
        survivors = placer.place(maze, n=2, seed=7)
        for s in survivors:
            placer.remove(s, maze.physics_client)
