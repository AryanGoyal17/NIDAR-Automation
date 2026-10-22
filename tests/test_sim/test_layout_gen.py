"""
tests/test_sim/test_layout_gen.py — Unit tests for sim.layout_gen
================================================================

Validates:
1. Determinism: Same seed generates exact same grid.
2. Diversity: Different seeds generate different layouts.
3. Grid validity: Only valid characters ('#', '.', 'S'), all rows equal length.
4. Outer boundary is properly walled.
5. Exactly one 'S' start position exists.
6. Corridor width is loaded from configs/arena.yaml.
7. YAML saving and roundtrip loading with sim.maze.load_layout works.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sim.layout_gen import MazeGenerator
from sim.maze import load_layout

ARENA_CONFIG = Path("configs/arena.yaml")


@pytest.fixture
def generator() -> MazeGenerator:
    return MazeGenerator(arena_config_path=ARENA_CONFIG)


class TestMazeGeneratorDeterminism:
    def test_same_seed_gives_identical_layout(self, generator: MazeGenerator) -> None:
        """Requirement: Same seed gives the same layout."""
        layout_a = generator.generate(rows=11, cols=11, seed=12345)
        layout_b = generator.generate(rows=11, cols=11, seed=12345)
        assert layout_a.grid == layout_b.grid
        assert layout_a.num_rows == layout_b.num_rows
        assert layout_a.num_cols == layout_b.num_cols
        assert layout_a.bounds_m == layout_b.bounds_m

    def test_different_seeds_give_different_layouts(
        self, generator: MazeGenerator
    ) -> None:
        layout_a = generator.generate(rows=13, cols=13, seed=101)
        layout_b = generator.generate(rows=13, cols=13, seed=202)
        assert layout_a.grid != layout_b.grid


class TestMazeGeneratorProperties:
    def test_corridor_width_from_arena_config(self, generator: MazeGenerator) -> None:
        # configs/arena.yaml has corridor_width: 1.5
        assert generator.corridor_width == pytest.approx(1.5)

    def test_dimensions_and_row_lengths(self, generator: MazeGenerator) -> None:
        layout = generator.generate(rows=11, cols=15, seed=42)
        assert layout.num_rows == 11
        assert layout.num_cols == 15
        assert len(layout.grid) == 11
        assert all(len(row) == 15 for row in layout.grid)

    def test_even_dimensions_are_promoted_to_odd(
        self, generator: MazeGenerator
    ) -> None:
        layout = generator.generate(rows=10, cols=12, seed=42)
        assert layout.num_rows == 11
        assert layout.num_cols == 13

    def test_valid_characters_only(self, generator: MazeGenerator) -> None:
        layout = generator.generate(rows=11, cols=11, seed=99)
        valid_chars = {"#", ".", "S"}
        for r_idx, row in enumerate(layout.grid):
            for c_idx, ch in enumerate(row):
                assert ch in valid_chars, f"Invalid ch '{ch}' at ({r_idx}, {c_idx})"

    def test_exactly_one_start_marker(self, generator: MazeGenerator) -> None:
        layout = generator.generate(rows=11, cols=11, seed=7)
        total_s = sum(row.count("S") for row in layout.grid)
        assert total_s == 1

    def test_outer_perimeter_is_walls(self, generator: MazeGenerator) -> None:
        layout = generator.generate(rows=11, cols=11, seed=42)
        top = layout.grid[0]
        bottom = layout.grid[-1]
        assert all(ch == "#" for ch in top)
        assert all(ch == "#" for ch in bottom)
        assert all(row[0] == "#" and row[-1] == "#" for row in layout.grid)

    def test_bounds_calculation(self, generator: MazeGenerator) -> None:
        layout = generator.generate(rows=11, cols=11, seed=42)
        expected_w = 11 * 1.5
        expected_l = 11 * 1.5
        assert layout.bounds_m == pytest.approx((expected_w, expected_l))

    def test_save_and_roundtrip_load(self, generator: MazeGenerator) -> None:
        layout = generator.generate(rows=11, cols=11, seed=777)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            layout.save(tmp_path)
            loaded_grid = load_layout(tmp_path)
            assert loaded_grid == layout.grid
        finally:
            tmp_path.unlink(missing_ok=True)
