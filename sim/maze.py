"""
sim/maze.py — PyBullet maze builder for NIDAR synthetic data generation
=======================================================================

Builds a 3D maze in a headless PyBullet simulation from an ASCII grid
layout (stored in a YAML file) and physical dimensions read from
``configs/arena.yaml``.

Grid legend:
    '#'  — wall cell (a box body is placed here)
    '.'  — free corridor (passable space)
    'S'  — start / entry cell (free space; marks the drone launch point)

Each cell is a square whose side length equals ``arena.corridor_width``
from the arena config.  Walls are vertical boxes spanning the full cell
footprint with height ``arena.wall_height``.

The grid origin (row 0, col 0) maps to world position ``(0, 0, 0)``
at the top-left corner.  Axes follow the project convention:
    x — rightward (column index)
    y — forward   (row index, increasing downward in the grid = increasing y)
    z — upward

Usage (standalone smoke test)::

    python -m sim.maze
    python -m sim.maze --layout configs/maze_layouts/sample_maze.yaml

Programmatic::

    from sim.maze import MazeBuilder
    builder = MazeBuilder()
    result = builder.build("configs/maze_layouts/sample_maze.yaml")
    print(result.num_walls, result.bounds)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pybullet as p
import yaml

from cv.config import load_config

# Default file paths (relative to project root)
DEFAULT_ARENA_CONFIG = Path("configs/arena.yaml")
DEFAULT_LAYOUT = Path("configs/maze_layouts/sample_maze.yaml")

WALL_CHAR = "#"
FREE_CHARS = {".", "S"}


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class MazeResult:
    """Holds the result of building a maze in PyBullet."""

    physics_client: int
    """PyBullet physics client ID."""

    grid: list[str]
    """The parsed ASCII grid rows."""

    num_rows: int
    """Number of rows in the grid."""

    num_cols: int
    """Number of columns in the grid (width of the widest row)."""

    cell_size: float
    """Side length of each grid cell in metres (= corridor_width)."""

    wall_height: float
    """Height of wall bodies in metres."""

    wall_thickness: float
    """Thickness of wall bodies in metres."""

    num_walls: int
    """Total number of wall bodies placed."""

    num_free: int
    """Total number of free (passable) cells."""

    start_cell: tuple[int, int] | None
    """(row, col) of the 'S' cell, or None if absent."""

    floor_id: int
    """PyBullet body ID of the floor plane."""

    ceiling_id: int | None
    """PyBullet body ID of the ceiling slab, or None if ceiling is disabled."""

    wall_body_ids: list[int] = field(default_factory=list)
    """PyBullet body IDs for every wall box."""

    @property
    def bounds(self) -> tuple[float, float, float]:
        """World-space bounding box dimensions (width_x, depth_y, height_z) in metres."""
        return (
            self.num_cols * self.cell_size,
            self.num_rows * self.cell_size,
            self.wall_height,
        )

    @property
    def start_world(self) -> tuple[float, float, float] | None:
        """World position (x, y, z) of the start cell centre, or None."""
        if self.start_cell is None:
            return None
        row, col = self.start_cell
        return (
            (col + 0.5) * self.cell_size,
            (row + 0.5) * self.cell_size,
            0.0,
        )


# --------------------------------------------------------------------------- #
# Layout parser
# --------------------------------------------------------------------------- #
def load_layout(layout_path: str | Path) -> list[str]:
    """
    Load and validate an ASCII grid layout from a YAML file.

    The YAML must contain::

        layout:
          grid:
            - "###..."
            - "#.#..."

    Returns a list of grid row strings, all padded to the same width.

    Raises:
        FileNotFoundError: If the layout file does not exist.
        ValueError: If the grid is empty or contains invalid characters.
    """
    path = Path(layout_path)
    if not path.exists():
        raise FileNotFoundError(f"Layout file not found: {path.resolve()}")

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    grid_data = raw.get("layout", {}).get("grid")
    if not grid_data or not isinstance(grid_data, list):
        raise ValueError(f"Layout file must contain layout.grid list: {path}")

    # Validate characters and normalise width
    max_width = max(len(row) for row in grid_data)
    grid: list[str] = []
    valid_chars = {WALL_CHAR} | FREE_CHARS
    for i, row in enumerate(grid_data):
        for ch in row:
            if ch not in valid_chars:
                raise ValueError(
                    f"Invalid character '{ch}' at row {i} in {path}. "
                    f"Valid: {valid_chars}"
                )
        # Pad short rows with wall characters (safe boundary)
        grid.append(row.ljust(max_width, WALL_CHAR))

    if len(grid) == 0:
        raise ValueError(f"Empty grid in {path}")

    return grid


# --------------------------------------------------------------------------- #
# Maze builder
# --------------------------------------------------------------------------- #
class MazeBuilder:
    """
    Constructs a PyBullet maze from a YAML layout and arena configuration.

    Parameters
    ----------
    arena_config_path : str or Path
        Path to ``configs/arena.yaml``.
    """

    def __init__(self, arena_config_path: str | Path = DEFAULT_ARENA_CONFIG) -> None:
        self._cfg = load_config(arena_config_path)

    # -- public API --

    def build(
        self,
        layout_path: str | Path = DEFAULT_LAYOUT,
        physics_client: int | None = None,
    ) -> MazeResult:
        """
        Parse a layout and build the maze in PyBullet.

        Parameters
        ----------
        layout_path : str or Path
            Path to the YAML layout file.
        physics_client : int, optional
            An existing PyBullet physics client ID.  If ``None``, a new
            ``p.DIRECT`` (headless) client is created.

        Returns
        -------
        MazeResult
            Dataclass with physics client, body IDs, grid metadata, and bounds.
        """
        grid = load_layout(layout_path)
        num_rows = len(grid)
        num_cols = len(grid[0])

        arena = self._cfg.arena
        cell_size: float = arena.corridor_width
        wall_height: float = arena.wall_height
        wall_thickness: float = arena.wall_thickness
        wall_color: list[float] = arena.walls.color_rgb
        floor_color: list[float] = arena.floor.color_rgb
        ceiling_present: bool = arena.ceiling.present
        ceiling_height: float = arena.ceiling.height
        ceiling_color: list[float] = arena.ceiling.color_rgb

        # Physics client
        if physics_client is None:
            physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=physics_client)

        # -- Floor --
        floor_half = [
            (num_cols * cell_size) / 2,
            (num_rows * cell_size) / 2,
            0.005,  # thin slab
        ]
        floor_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=floor_half, physicsClientId=physics_client
        )
        floor_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=floor_half,
            rgbaColor=[*floor_color, 1.0],
            physicsClientId=physics_client,
        )
        floor_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_col,
            baseVisualShapeIndex=floor_vis,
            basePosition=[floor_half[0], floor_half[1], -0.005],
            physicsClientId=physics_client,
        )

        # -- Ceiling (optional) --
        ceiling_id = None
        if ceiling_present:
            ceil_half = [floor_half[0], floor_half[1], 0.005]
            ceil_col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=ceil_half, physicsClientId=physics_client
            )
            ceil_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=ceil_half,
                rgbaColor=[*ceiling_color, 1.0],
                physicsClientId=physics_client,
            )
            ceiling_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=ceil_col,
                baseVisualShapeIndex=ceil_vis,
                basePosition=[floor_half[0], floor_half[1], ceiling_height + 0.005],
                physicsClientId=physics_client,
            )

        # -- Walls --
        wall_half = [cell_size / 2, cell_size / 2, wall_height / 2]
        wall_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=wall_half, physicsClientId=physics_client
        )
        wall_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=wall_half,
            rgbaColor=[*wall_color, 1.0],
            physicsClientId=physics_client,
        )

        wall_body_ids: list[int] = []
        num_walls = 0
        num_free = 0
        start_cell: tuple[int, int] | None = None

        for row_idx, row in enumerate(grid):
            for col_idx, ch in enumerate(row):
                if ch == WALL_CHAR:
                    # Centre of the cell in world coordinates
                    cx = (col_idx + 0.5) * cell_size
                    cy = (row_idx + 0.5) * cell_size
                    cz = wall_height / 2  # half-height above floor

                    body_id = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=wall_col,
                        baseVisualShapeIndex=wall_vis,
                        basePosition=[cx, cy, cz],
                        physicsClientId=physics_client,
                    )
                    wall_body_ids.append(body_id)
                    num_walls += 1
                else:
                    num_free += 1
                    if ch == "S":
                        start_cell = (row_idx, col_idx)

        return MazeResult(
            physics_client=physics_client,
            grid=grid,
            num_rows=num_rows,
            num_cols=num_cols,
            cell_size=cell_size,
            wall_height=wall_height,
            wall_thickness=wall_thickness,
            num_walls=num_walls,
            num_free=num_free,
            start_cell=start_cell,
            floor_id=floor_id,
            ceiling_id=ceiling_id,
            wall_body_ids=wall_body_ids,
        )


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="NIDAR maze builder smoke test")
    parser.add_argument(
        "--layout",
        default=str(DEFAULT_LAYOUT),
        help="Path to YAML layout file (default: %(default)s)",
    )
    parser.add_argument(
        "--arena",
        default=str(DEFAULT_ARENA_CONFIG),
        help="Path to arena config (default: %(default)s)",
    )
    args = parser.parse_args()

    builder = MazeBuilder(arena_config_path=args.arena)
    result = builder.build(layout_path=args.layout)

    print("=" * 60)
    print("  NIDAR AirMouse — PyBullet Maze Builder")
    print("=" * 60)
    print(f"  Layout file   : {args.layout}")
    print(f"  Grid size     : {result.num_rows} rows × {result.num_cols} cols")
    print(f"  Cell size     : {result.cell_size} m")
    print(f"  Wall height   : {result.wall_height} m")
    print(f"  Wall count    : {result.num_walls}")
    print(f"  Free cells    : {result.num_free}")
    print(f"  Start cell    : {result.start_cell}")
    print(f"  Start world   : {result.start_world}")
    print(f"  Bounds (x,y,z): {result.bounds}")
    print(f"  Floor body ID : {result.floor_id}")
    print(f"  Ceiling body  : {result.ceiling_id}")
    print(f"  Wall body IDs : {len(result.wall_body_ids)} bodies")
    print()
    print("Grid:")
    for row in result.grid:
        print(f"    {row}")
    print()

    p.disconnect(result.physics_client)
    print("PyBullet disconnected. Maze build OK.")


if __name__ == "__main__":
    main()
