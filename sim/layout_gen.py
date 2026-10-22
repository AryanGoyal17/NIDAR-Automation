"""
sim/layout_gen.py — Seeded random maze generator for NIDAR synthetic data
========================================================================

Generates random ASCII grid mazes using the recursive backtracker algorithm
(depth-first search with backtracking). Ensures exact determinism when a
random seed is provided.

Physical cell sizing (corridor clear passage width) is read from
``configs/arena.yaml`` (key: ``arena.corridor_width``).

ASCII Legend:
    '#' — wall cell
    '.' — free corridor passage
    'S' — start / drone entry point (at the initial carved cell)

Usage (CLI)::

    python -m sim.layout_gen --seed 42 --rows 11 --cols 11
    python -m sim.layout_gen --seed 42 --output configs/maze_layouts/gen_42.yaml

Programmatic::

    from sim.layout_gen import MazeGenerator
    gen = MazeGenerator()
    grid = gen.generate(rows=11, cols=11, seed=42)
    # or dimension-based (in metres):
    grid = gen.generate_from_dimensions(width_m=15.0, height_m=15.0, seed=42)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import yaml

from cv.config import load_config

DEFAULT_ARENA_CONFIG = Path("configs/arena.yaml")
WALL_CHAR = "#"
FREE_CHAR = "."
START_CHAR = "S"


@dataclass
class GeneratedLayout:
    """Container for a generated maze layout and its physical metadata."""

    grid: list[str]
    seed: int | None
    num_rows: int
    num_cols: int
    corridor_width: float
    bounds_m: tuple[float, float]
    """(width_x, length_y) in metres."""

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to dict representation matching NIDAR layout YAML format."""
        return {
            "layout": {
                "grid": self.grid,
                "metadata": {
                    "generator": "recursive_backtracker",
                    "seed": self.seed,
                    "rows": self.num_rows,
                    "cols": self.num_cols,
                    "corridor_width_m": self.corridor_width,
                    "bounds_m": {
                        "width_x": self.bounds_m[0],
                        "length_y": self.bounds_m[1],
                    },
                },
            }
        }

    def save(self, output_path: str | Path) -> Path:
        """Write layout to a YAML file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_yaml_dict(), f, sort_keys=False)
        return out


class MazeGenerator:
    """
    Seeded recursive backtracker maze generator.

    Parameters
    ----------
    arena_config_path : str or Path
        Path to ``configs/arena.yaml``. Corridor width is loaded from this file.
    """

    def __init__(self, arena_config_path: str | Path = DEFAULT_ARENA_CONFIG) -> None:
        self.arena_config_path = Path(arena_config_path)
        self._cfg = load_config(self.arena_config_path)
        self.corridor_width: float = float(self._cfg.arena.corridor_width)

    def generate(
        self,
        rows: int = 11,
        cols: int = 11,
        seed: int | None = None,
    ) -> GeneratedLayout:
        """
        Generate an ASCII grid maze of specified rows and columns.

        Parameters
        ----------
        rows : int
            Total number of rows in the resulting ASCII grid (must be >= 5;
            even numbers are automatically increased to odd).
        cols : int
            Total number of columns in the resulting ASCII grid (must be >= 5;
            even numbers are automatically increased to odd).
        seed : int, optional
            Random seed for repeatable generation.

        Returns
        -------
        GeneratedLayout
            Object containing the grid strings, seed, and physical metrics.
        """
        # Ensure dimensions are odd and >= 5 for valid wall-corridor grid
        rows = max(5, rows)
        cols = max(5, cols)
        if rows % 2 == 0:
            rows += 1
        if cols % 2 == 0:
            cols += 1

        rng = random.Random(seed)

        # Initialise grid with all walls
        grid: list[list[str]] = [[WALL_CHAR for _ in range(cols)] for _ in range(rows)]

        # Cells in recursive backtracker graph (odd coordinates)
        start_r, start_c = 1, 1
        grid[start_r][start_c] = START_CHAR

        visited = {(start_r, start_c)}
        stack: list[tuple[int, int]] = [(start_r, start_c)]

        # Cardinal moves: 2 steps in grid (jump over intermediate wall)
        directions = [
            (-2, 0),  # North
            (2, 0),   # South
            (0, 2),   # East
            (0, -2),  # West
        ]

        while stack:
            curr_r, curr_c = stack[-1]

            # Find unvisited 2-step neighbors
            neighbors: list[tuple[int, int, int, int]] = []
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1:
                    if (nr, nc) not in visited:
                        # (neighbor_row, neighbor_col, wall_row, wall_col)
                        neighbors.append((nr, nc, curr_r + dr // 2, curr_c + dc // 2))

            if neighbors:
                nr, nc, wr, wc = rng.choice(neighbors)
                # Carve wall and neighbor cell
                grid[wr][wc] = FREE_CHAR
                grid[nr][nc] = FREE_CHAR
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

        grid_strings = ["".join(row) for row in grid]
        bounds_m = (cols * self.corridor_width, rows * self.corridor_width)

        return GeneratedLayout(
            grid=grid_strings,
            seed=seed,
            num_rows=rows,
            num_cols=cols,
            corridor_width=self.corridor_width,
            bounds_m=bounds_m,
        )

    def generate_from_dimensions(
        self,
        width_m: float = 15.0,
        height_m: float = 15.0,
        seed: int | None = None,
    ) -> GeneratedLayout:
        """
        Generate a maze scaled to target physical bounding dimensions in metres.

        Number of rows and columns is derived directly from the corridor width.

        Parameters
        ----------
        width_m : float
            Target total width in metres (East-West / cols).
        height_m : float
            Target total height in metres (North-South / rows).
        seed : int, optional
            Random seed for repeatable generation.
        """
        cols = int(round(width_m / self.corridor_width))
        rows = int(round(height_m / self.corridor_width))
        return self.generate(rows=rows, cols=cols, seed=seed)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="NIDAR Seeded Maze Generator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rows", type=int, default=11, help="ASCII grid rows")
    parser.add_argument("--cols", type=int, default=11, help="ASCII grid cols")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save layout YAML",
    )
    parser.add_argument(
        "--arena",
        type=str,
        default=str(DEFAULT_ARENA_CONFIG),
        help="Path to arena.yaml",
    )
    args = parser.parse_args()

    generator = MazeGenerator(arena_config_path=args.arena)
    layout = generator.generate(rows=args.rows, cols=args.cols, seed=args.seed)

    print("=" * 60)
    print("  NIDAR Seeded Maze Generator (Recursive Backtracker)")
    print("=" * 60)
    print(f"  Seed           : {layout.seed}")
    print(f"  Grid Size      : {layout.num_rows} rows × {layout.num_cols} cols")
    print(f"  Corridor Width : {layout.corridor_width} m (from {args.arena})")
    print(f"  Physical Bounds: {layout.bounds_m[0]:.2f}m × {layout.bounds_m[1]:.2f}m")
    print()
    print("ASCII Grid:")
    for row in layout.grid:
        print(f"    {row}")
    print()

    if args.output:
        saved_path = layout.save(args.output)
        print(f"Layout saved to: {saved_path}")


if __name__ == "__main__":
    main()
