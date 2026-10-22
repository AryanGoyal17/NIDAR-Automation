
"""
tests/test_sim/test_maze.py — Unit tests for sim.maze MazeBuilder
=================================================================

Validates that:
1. The sample maze layout parses and loads correctly.
2. The wall count matches the number of '#' cells in the grid.
3. The free cell count matches the number of '.' and 'S' cells.
4. The world-space bounds match (num_cols * cell_size, num_rows * cell_size, wall_height).
5. The start cell ('S') is found at the correct grid position.
6. PyBullet body IDs are created for every wall.
7. A minimal inline layout produces the correct counts.
8. Invalid characters in a layout are rejected.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pybullet as p
import pytest
import yaml

from sim.maze import MazeBuilder, load_layout

SAMPLE_LAYOUT = Path("configs/maze_layouts/sample_maze.yaml")
ARENA_CONFIG = Path("configs/arena.yaml")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _count_char(grid: list[str], ch: str) -> int:
    return sum(row.count(ch) for row in grid)


def _write_temp_layout(grid_rows: list[str]) -> Path:
    """Write a temporary YAML layout file and return its path."""
    data = {"layout": {"grid": grid_rows}}
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.safe_dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_result():
    """Build the sample maze once for the entire test module."""
    builder = MazeBuilder(arena_config_path=ARENA_CONFIG)
    result = builder.build(layout_path=SAMPLE_LAYOUT)
    yield result
    p.disconnect(result.physics_client)


@pytest.fixture(scope="module")
def sample_grid() -> list[str]:
    """Load the sample grid for counting checks."""
    return load_layout(SAMPLE_LAYOUT)


# --------------------------------------------------------------------------- #
# Layout parser tests
# --------------------------------------------------------------------------- #
class TestLoadLayout:
    def test_sample_layout_loads(self) -> None:
        grid = load_layout(SAMPLE_LAYOUT)
        assert len(grid) > 0
        assert all(isinstance(row, str) for row in grid)

    def test_all_rows_same_width(self) -> None:
        grid = load_layout(SAMPLE_LAYOUT)
        widths = {len(row) for row in grid}
        assert len(widths) == 1, f"Rows have inconsistent widths: {widths}"

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_layout("configs/maze_layouts/nonexistent_12345.yaml")

    def test_invalid_character_raises(self) -> None:
        path = _write_temp_layout(["##X#"])
        try:
            with pytest.raises(ValueError, match="Invalid character"):
                load_layout(path)
        finally:
            path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Wall count tests
# --------------------------------------------------------------------------- #
class TestWallCount:
    def test_wall_count_matches_grid(
        self, sample_result, sample_grid: list[str]
    ) -> None:
        expected_walls = _count_char(sample_grid, "#")
        assert sample_result.num_walls == expected_walls, (
            f"Expected {expected_walls} walls from grid, got {sample_result.num_walls}"
        )

    def test_free_count_matches_grid(
        self, sample_result, sample_grid: list[str]
    ) -> None:
        expected_free = _count_char(sample_grid, ".") + _count_char(sample_grid, "S")
        assert sample_result.num_free == expected_free, (
            f"Expected {expected_free} free cells, got {sample_result.num_free}"
        )

    def test_wall_plus_free_equals_total(self, sample_result) -> None:
        total = sample_result.num_rows * sample_result.num_cols
        assert sample_result.num_walls + sample_result.num_free == total

    def test_body_id_count_matches_wall_count(self, sample_result) -> None:
        assert len(sample_result.wall_body_ids) == sample_result.num_walls


# --------------------------------------------------------------------------- #
# Bounds tests
# --------------------------------------------------------------------------- #
class TestBounds:
    def test_bounds_match_layout(self, sample_result, sample_grid: list[str]) -> None:
        num_cols = len(sample_grid[0])
        num_rows = len(sample_grid)
        cell_size = sample_result.cell_size
        expected_bounds = (
            num_cols * cell_size,
            num_rows * cell_size,
            sample_result.wall_height,
        )
        assert sample_result.bounds == pytest.approx(expected_bounds, abs=1e-6), (
            f"Expected bounds {expected_bounds}, got {sample_result.bounds}"
        )

    def test_bounds_positive(self, sample_result) -> None:
        bx, by, bz = sample_result.bounds
        assert bx > 0 and by > 0 and bz > 0


# --------------------------------------------------------------------------- #
# Start cell tests
# --------------------------------------------------------------------------- #
class TestStartCell:
    def test_start_cell_found(self, sample_result) -> None:
        assert sample_result.start_cell is not None, "Start cell 'S' not found"

    def test_start_cell_position(self, sample_result, sample_grid: list[str]) -> None:
        # Find the expected 'S' position from the grid
        expected = None
        for r, row in enumerate(sample_grid):
            for c, ch in enumerate(row):
                if ch == "S":
                    expected = (r, c)
                    break
            if expected:
                break
        assert sample_result.start_cell == expected

    def test_start_world_position(self, sample_result) -> None:
        assert sample_result.start_world is not None
        x, y, z = sample_result.start_world
        assert z == 0.0
        assert x > 0 and y > 0


# --------------------------------------------------------------------------- #
# Floor and ceiling tests
# --------------------------------------------------------------------------- #
class TestFloorAndCeiling:
    def test_floor_body_exists(self, sample_result) -> None:
        assert sample_result.floor_id >= 0

    def test_ceiling_body_exists_when_configured(self, sample_result) -> None:
        # arena.yaml has ceiling.present: true
        assert sample_result.ceiling_id is not None
        assert sample_result.ceiling_id >= 0


# --------------------------------------------------------------------------- #
# Minimal inline layout test
# --------------------------------------------------------------------------- #
class TestMinimalLayout:
    """Build a tiny 3×3 maze to verify counts independently."""

    def test_tiny_maze(self) -> None:
        grid_rows = ["###", "#S#", "###"]
        path = _write_temp_layout(grid_rows)
        try:
            builder = MazeBuilder(arena_config_path=ARENA_CONFIG)
            result = builder.build(layout_path=path)

            assert result.num_walls == 8  # 8 '#' cells
            assert result.num_free == 1   # 1 'S' cell
            assert result.start_cell == (1, 1)
            assert result.num_rows == 3
            assert result.num_cols == 3

            bx, by, bz = result.bounds
            assert bx == pytest.approx(3 * result.cell_size)
            assert by == pytest.approx(3 * result.cell_size)

            p.disconnect(result.physics_client)
        finally:
            path.unlink(missing_ok=True)
