"""
sim/survivor.py — Procedural survivor (human target) generator for PyBullet
=============================================================================

Builds simple human-like survivors from PyBullet geometric primitives
(capsules, spheres, boxes) suitable for generating labelled training data
via the segmentation mask.

Supported postures
------------------
- **standing**       — upright, arms at sides.
- **sitting**        — legs forward, torso upright, back against a wall.
- **lying_supine**   — flat on back, face up.
- **lying_prone**    — flat on stomach, face down.
- **lying_lateral**  — on one side (recovery position).
- **curled**         — foetal / curled-up ball (most compact).

Each survivor is randomly:
- Scaled by ±15 % from the nominal ``arena.yaml`` dimensions.
- Coloured with a random skin / mannequin base and clothing colour.
- Given a random yaw orientation.
- Placed on the floor in a randomly chosen free cell (no wall overlap).

The module returns a list of :class:`SurvivorInfo` records containing the
PyBullet body IDs so that post-render code can map segmentation-mask IDs
back to individual survivors.

Usage (CLI smoke test)::

    python -m sim.survivor

Programmatic::

    from sim.survivor import SurvivorPlacer
    placer = SurvivorPlacer()
    survivors = placer.place(maze_result, n=2, seed=42)
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pybullet as p

from cv.config import load_config
from sim.maze import FREE_CHARS, MazeBuilder, MazeResult

DEFAULT_ARENA_CONFIG = Path("configs/arena.yaml")

# --------------------------------------------------------------------------- #
# Colour palettes
# --------------------------------------------------------------------------- #

# Skin / mannequin base colours (RGB floats 0-1).
SKIN_COLORS = [
    (0.96, 0.82, 0.71),   # Fair
    (0.87, 0.72, 0.58),   # Light tan
    (0.76, 0.60, 0.44),   # Medium
    (0.55, 0.38, 0.25),   # Brown
    (0.36, 0.24, 0.16),   # Dark brown
    (0.70, 0.70, 0.70),   # Mannequin grey
    (0.85, 0.82, 0.78),   # Mannequin beige
]

# Default clothing palette (used if arena.yaml palette is missing/empty).
DEFAULT_CLOTHING = [
    (0.15, 0.25, 0.55),   # Navy
    (0.70, 0.15, 0.15),   # Red
    (0.20, 0.45, 0.25),   # Dark green
    (0.25, 0.25, 0.25),   # Charcoal grey
    (0.85, 0.80, 0.70),   # Light khaki
    (0.40, 0.30, 0.22),   # Brown
    (0.12, 0.12, 0.12),   # Near black
    (0.90, 0.90, 0.88),   # Off white
]

# Posture identifiers.
POSTURES = [
    "standing",
    "sitting",
    "lying_supine",
    "lying_prone",
    "lying_lateral",
    "curled",
]


# --------------------------------------------------------------------------- #
# Data class
# --------------------------------------------------------------------------- #
@dataclass
class SurvivorInfo:
    """Record for a single placed survivor."""

    body_ids: list[int]
    """PyBullet body IDs for every primitive composing this survivor."""

    posture: str
    """Posture name (e.g. 'lying_prone')."""

    cell: tuple[int, int]
    """Grid cell (row, col) where the survivor was placed."""

    world_pos: tuple[float, float, float]
    """Centre of the survivor base on the floor (x, y, z=0)."""

    yaw_deg: float
    """Random yaw orientation in degrees."""

    scale: float
    """Multiplicative scale factor applied to nominal dimensions."""


# --------------------------------------------------------------------------- #
# Helpers — primitive body creation
# --------------------------------------------------------------------------- #

def _rgba(rgb: tuple[float, ...], a: float = 1.0) -> list[float]:
    """Convert an RGB tuple to an RGBA list."""
    return [rgb[0], rgb[1], rgb[2], a]


def _create_box(
    cid: int,
    half_extents: tuple[float, float, float],
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float] | None = None,
    color_rgb: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> int:
    """Create a static box body and return its PyBullet body ID."""
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=list(half_extents), physicsClientId=cid
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=list(half_extents),
        rgbaColor=_rgba(color_rgb),
        physicsClientId=cid,
    )
    orn = orientation or (0, 0, 0, 1)
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(position),
        baseOrientation=list(orn),
        physicsClientId=cid,
    )
    return body


def _create_capsule(
    cid: int,
    radius: float,
    height: float,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float] | None = None,
    color_rgb: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> int:
    """Create a static capsule body (cylinder with hemispherical caps)."""
    col = p.createCollisionShape(
        p.GEOM_CAPSULE, radius=radius, height=height, physicsClientId=cid
    )
    vis = p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=radius,
        length=height,
        rgbaColor=_rgba(color_rgb),
        physicsClientId=cid,
    )
    orn = orientation or (0, 0, 0, 1)
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(position),
        baseOrientation=list(orn),
        physicsClientId=cid,
    )
    return body


def _create_sphere(
    cid: int,
    radius: float,
    position: tuple[float, float, float],
    color_rgb: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> int:
    """Create a static sphere body."""
    col = p.createCollisionShape(
        p.GEOM_SPHERE, radius=radius, physicsClientId=cid
    )
    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=_rgba(color_rgb),
        physicsClientId=cid,
    )
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(position),
        baseOrientation=(0, 0, 0, 1),
        physicsClientId=cid,
    )
    return body


def _yaw_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    """Quaternion for a rotation about the z-axis by *yaw_rad*."""
    return p.getQuaternionFromEuler([0, 0, yaw_rad])


def _euler_quaternion(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """Quaternion from Euler angles (radians)."""
    return p.getQuaternionFromEuler([roll, pitch, yaw])


def _rotate_xy(
    dx: float, dy: float, yaw_rad: float
) -> tuple[float, float]:
    """Rotate (dx, dy) offset by *yaw_rad* around the z-axis."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return (dx * c - dy * s, dx * s + dy * c)


# --------------------------------------------------------------------------- #
# Posture builders — each returns a list of body IDs
# --------------------------------------------------------------------------- #

def _build_standing(
    cid: int,
    cx: float, cy: float,
    yaw_rad: float,
    h: float, sw: float, td: float,
    skin: tuple[float, ...],
    cloth: tuple[float, ...],
) -> list[int]:
    """Standing posture — upright, arms at sides.

    Body parts (bottom-up):
    - Two leg capsules (hip to ground).
    - Torso box (from hips to shoulders).
    - Head sphere on top.
    - Two arm capsules hanging beside torso.
    """
    ids: list[int] = []

    # Proportions.
    leg_len = h * 0.47
    torso_len = h * 0.33
    head_r = h * 0.075
    arm_len = h * 0.38
    leg_r = sw * 0.14
    arm_r = sw * 0.10

    # Legs (two capsules, standing vertically).
    for side in (-1, 1):
        lx, ly = _rotate_xy(side * sw * 0.20, 0, yaw_rad)
        ids.append(_create_capsule(
            cid, leg_r, leg_len,
            (cx + lx, cy + ly, leg_len / 2 + leg_r),
            color_rgb=cloth,
        ))

    # Torso box.
    torso_z = leg_len + leg_r + torso_len / 2
    ids.append(_create_box(
        cid,
        (sw / 2, td / 2, torso_len / 2),
        (cx, cy, torso_z),
        _yaw_quaternion(yaw_rad),
        color_rgb=cloth,
    ))

    # Head sphere.
    head_z = torso_z + torso_len / 2 + head_r
    ids.append(_create_sphere(
        cid, head_r,
        (cx, cy, head_z),
        color_rgb=skin,
    ))

    # Arms (hanging at sides).
    shoulder_z = torso_z + torso_len / 2 - arm_r
    for side in (-1, 1):
        ax, ay = _rotate_xy(side * (sw / 2 + arm_r), 0, yaw_rad)
        ids.append(_create_capsule(
            cid, arm_r, arm_len,
            (cx + ax, cy + ay, shoulder_z - arm_len / 2),
            color_rgb=cloth,
        ))

    return ids


def _build_sitting(
    cid: int,
    cx: float, cy: float,
    yaw_rad: float,
    h: float, sw: float, td: float,
    skin: tuple[float, ...],
    cloth: tuple[float, ...],
) -> list[int]:
    """Sitting posture — legs forward on floor, torso upright."""
    ids: list[int] = []

    leg_len = h * 0.47
    torso_len = h * 0.33
    head_r = h * 0.075
    arm_len = h * 0.30
    leg_r = sw * 0.14
    arm_r = sw * 0.10

    # Legs — horizontal capsules on floor.
    for side in (-1, 1):
        lx_off, ly_off = _rotate_xy(side * sw * 0.20, leg_len / 2, yaw_rad)
        ids.append(_create_capsule(
            cid, leg_r, leg_len,
            (cx + lx_off, cy + ly_off, leg_r),
            _euler_quaternion(0, math.pi / 2, yaw_rad),
            color_rgb=cloth,
        ))

    # Torso — vertical box starting at floor.
    torso_z = torso_len / 2
    ids.append(_create_box(
        cid,
        (sw / 2, td / 2, torso_len / 2),
        (cx, cy, torso_z),
        _yaw_quaternion(yaw_rad),
        color_rgb=cloth,
    ))

    # Head.
    head_z = torso_z + torso_len / 2 + head_r
    ids.append(_create_sphere(
        cid, head_r,
        (cx, cy, head_z),
        color_rgb=skin,
    ))

    # Arms — resting on floor beside body.
    arm_z = arm_len / 2
    for side in (-1, 1):
        ax, ay = _rotate_xy(side * (sw / 2 + arm_r), 0, yaw_rad)
        ids.append(_create_capsule(
            cid, arm_r, arm_len,
            (cx + ax, cy + ay, arm_z),
            color_rgb=cloth,
        ))

    return ids


def _build_lying(
    cid: int,
    cx: float, cy: float,
    yaw_rad: float,
    h: float, sw: float, td: float,
    skin: tuple[float, ...],
    cloth: tuple[float, ...],
    face_roll: float = 0.0,
) -> list[int]:
    """Lying posture — flat on floor.

    *face_roll*: 0 = supine (face up), π = prone (face down),
                 ±π/2 = lateral (on side).
    """
    ids: list[int] = []

    leg_len = h * 0.47
    torso_len = h * 0.33
    head_r = h * 0.075
    arm_len = h * 0.38
    leg_r = sw * 0.14
    arm_r = sw * 0.10
    body_r = max(sw, td) / 2

    # Body roll lifts the centreline slightly above floor.
    base_z = body_r

    # Torso — horizontal capsule.
    torso_orn = _euler_quaternion(face_roll, math.pi / 2, yaw_rad)
    ids.append(_create_capsule(
        cid, td / 2, torso_len,
        (cx, cy, base_z),
        torso_orn,
        color_rgb=cloth,
    ))

    # Legs — extend forward from torso along yaw.
    leg_dx, leg_dy = _rotate_xy(0, (torso_len / 2 + leg_len / 2), yaw_rad)
    for side in (-1, 1):
        sx, sy = _rotate_xy(side * sw * 0.15, 0, yaw_rad)
        ids.append(_create_capsule(
            cid, leg_r, leg_len,
            (cx + leg_dx + sx, cy + leg_dy + sy, base_z),
            _euler_quaternion(face_roll, math.pi / 2, yaw_rad),
            color_rgb=cloth,
        ))

    # Head sphere — extends behind torso.
    hx, hy = _rotate_xy(0, -(torso_len / 2 + head_r), yaw_rad)
    ids.append(_create_sphere(
        cid, head_r,
        (cx + hx, cy + hy, base_z),
        color_rgb=skin,
    ))

    # Arms — beside torso.
    for side in (-1, 1):
        ax, ay = _rotate_xy(side * (sw / 2 + arm_r * 2), 0, yaw_rad)
        ids.append(_create_capsule(
            cid, arm_r, arm_len,
            (cx + ax, cy + ay, base_z),
            _euler_quaternion(face_roll, math.pi / 2, yaw_rad),
            color_rgb=cloth,
        ))

    return ids


def _build_curled(
    cid: int,
    cx: float, cy: float,
    yaw_rad: float,
    h: float, sw: float, td: float,
    skin: tuple[float, ...],
    cloth: tuple[float, ...],
) -> list[int]:
    """Curled / foetal posture — tight ball on the floor."""
    ids: list[int] = []

    torso_len = h * 0.25
    head_r = h * 0.07
    leg_r = sw * 0.14
    body_r = max(sw, td) / 2
    base_z = body_r

    # Main torso — shorter, tucked.
    ids.append(_create_capsule(
        cid, td / 2, torso_len,
        (cx, cy, base_z),
        _euler_quaternion(math.pi / 6, math.pi / 2, yaw_rad),
        color_rgb=cloth,
    ))

    # Knees tucked up — a box representing folded legs.
    kx, ky = _rotate_xy(0, torso_len * 0.3, yaw_rad)
    ids.append(_create_box(
        cid,
        (sw / 2, torso_len * 0.4, body_r * 0.6),
        (cx + kx, cy + ky, base_z),
        _euler_quaternion(0, 0, yaw_rad),
        color_rgb=cloth,
    ))

    # Head — tucked down at one end.
    hx, hy = _rotate_xy(0, -(torso_len * 0.5 + head_r * 0.5), yaw_rad)
    ids.append(_create_sphere(
        cid, head_r,
        (cx + hx, cy + hy, base_z),
        color_rgb=skin,
    ))

    return ids


# Map posture name -> builder callable.
_POSTURE_BUILDERS = {
    "standing": lambda cid, cx, cy, yr, h, sw, td, sk, cl:
        _build_standing(cid, cx, cy, yr, h, sw, td, sk, cl),
    "sitting": lambda cid, cx, cy, yr, h, sw, td, sk, cl:
        _build_sitting(cid, cx, cy, yr, h, sw, td, sk, cl),
    "lying_supine": lambda cid, cx, cy, yr, h, sw, td, sk, cl:
        _build_lying(cid, cx, cy, yr, h, sw, td, sk, cl, face_roll=0.0),
    "lying_prone": lambda cid, cx, cy, yr, h, sw, td, sk, cl:
        _build_lying(cid, cx, cy, yr, h, sw, td, sk, cl, face_roll=math.pi),
    "lying_lateral": lambda cid, cx, cy, yr, h, sw, td, sk, cl:
        _build_lying(cid, cx, cy, yr, h, sw, td, sk, cl, face_roll=math.pi / 2),
    "curled": lambda cid, cx, cy, yr, h, sw, td, sk, cl:
        _build_curled(cid, cx, cy, yr, h, sw, td, sk, cl),
}


# --------------------------------------------------------------------------- #
# Survivor Placer
# --------------------------------------------------------------------------- #
class SurvivorPlacer:
    """
    Places procedural survivors in a built PyBullet maze.

    Parameters
    ----------
    arena_config_path : str or Path
        Path to ``configs/arena.yaml``.
    min_survivors : int
        Minimum number of survivors per scene (default 0).
    max_survivors : int
        Maximum number of survivors per scene (default 3).
    scale_jitter : float
        Maximum multiplicative deviation from nominal size (default 0.15 = ±15 %).
    """

    def __init__(
        self,
        arena_config_path: str | Path = DEFAULT_ARENA_CONFIG,
        min_survivors: int = 0,
        max_survivors: int = 3,
        scale_jitter: float = 0.15,
    ) -> None:
        cfg = load_config(arena_config_path)
        surv = cfg.survivor

        self.nominal_height: float = float(surv.dimensions.height_m)
        self.nominal_shoulder: float = float(surv.dimensions.shoulder_width_m)
        self.nominal_depth: float = float(surv.dimensions.torso_depth_m)

        # Clothing palette from config (fall back to built-in).
        self.clothing_palette: list[tuple[float, ...]] = []
        if hasattr(surv, "clothing_color_palette"):
            for c in surv.clothing_color_palette:
                self.clothing_palette.append(tuple(float(v) for v in c))
        if not self.clothing_palette:
            self.clothing_palette = list(DEFAULT_CLOTHING)

        self.min_survivors = min_survivors
        self.max_survivors = max_survivors
        self.scale_jitter = scale_jitter

        # Arena grid params (loaded per-place call from MazeResult).
        self.cell_size: float = float(cfg.arena.corridor_width)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def place(
        self,
        maze: MazeResult,
        n: int | None = None,
        seed: int | None = None,
    ) -> list[SurvivorInfo]:
        """
        Place survivors in the maze.

        Parameters
        ----------
        maze : MazeResult
            A built maze (provides physics client, grid, cell_size).
        n : int, optional
            Exact number of survivors.  If *None*, a random count in
            ``[min_survivors, max_survivors]`` is chosen.
        seed : int, optional
            Random seed for full reproducibility.

        Returns
        -------
        list[SurvivorInfo]
            Records for each placed survivor (body IDs, posture, cell, etc.).
        """
        rng = random.Random(seed)
        cid = maze.physics_client

        # Determine count.
        if n is None:
            n = rng.randint(self.min_survivors, self.max_survivors)
        n = max(0, n)

        # Gather free cells (exclude walls and the start cell).
        free_cells: list[tuple[int, int]] = []
        for r in range(maze.num_rows):
            for c in range(maze.num_cols):
                if maze.grid[r][c] in FREE_CHARS:
                    free_cells.append((r, c))

        if not free_cells:
            return []

        # Cannot place more survivors than free cells.
        n = min(n, len(free_cells))

        # Shuffle and pick non-overlapping cells.
        chosen_cells = rng.sample(free_cells, n)

        survivors: list[SurvivorInfo] = []
        for cell in chosen_cells:
            row, col = cell
            cx = (col + 0.5) * maze.cell_size
            cy = (row + 0.5) * maze.cell_size

            # Random posture.
            posture = rng.choice(POSTURES)

            # Random yaw.
            yaw_deg = rng.uniform(0, 360)
            yaw_rad = math.radians(yaw_deg)

            # Random scale ±15 %.
            scale = 1.0 + rng.uniform(-self.scale_jitter, self.scale_jitter)
            h = self.nominal_height * scale
            sw = self.nominal_shoulder * scale
            td = self.nominal_depth * scale

            # Random colours.
            skin = rng.choice(SKIN_COLORS)
            cloth = rng.choice(self.clothing_palette)

            # Build posture.
            builder_fn = _POSTURE_BUILDERS[posture]
            body_ids = builder_fn(cid, cx, cy, yaw_rad, h, sw, td, skin, cloth)

            survivors.append(SurvivorInfo(
                body_ids=body_ids,
                posture=posture,
                cell=cell,
                world_pos=(cx, cy, 0.0),
                yaw_deg=yaw_deg,
                scale=scale,
            ))

        return survivors

    def remove(self, survivor: SurvivorInfo, cid: int) -> None:
        """Remove a single survivor's bodies from the simulation."""
        for bid in survivor.body_ids:
            p.removeBody(bid, physicsClientId=cid)

    def remove_all(
        self, survivors: list[SurvivorInfo], cid: int
    ) -> None:
        """Remove all survivors from the simulation."""
        for s in survivors:
            self.remove(s, cid)


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #
def _main() -> None:
    """Quick smoke test: build a maze and place survivors."""
    import textwrap

    print("=== sim.survivor smoke test ===\n")

    builder = MazeBuilder()
    grid = [
        "#########",
        "#S......#",
        "#.#####.#",
        "#.#####.#",
        "#.#####.#",
        "#.......#",
        "#########",
    ]
    maze = builder.build_from_grid(grid)
    print(f"Maze built: {maze.num_rows}x{maze.num_cols}, {maze.num_walls} walls")

    placer = SurvivorPlacer()

    for seed in (42, 99, 7):
        survivors = placer.place(maze, seed=seed)
        print(f"\nSeed {seed}: placed {len(survivors)} survivors")
        for i, s in enumerate(survivors):
            print(textwrap.indent(
                f"#{i}: posture={s.posture}, cell={s.cell}, "
                f"yaw={s.yaw_deg:.1f}°, scale={s.scale:.2f}, "
                f"parts={len(s.body_ids)}",
                "  ",
            ))
        # Clean up.
        placer.remove_all(survivors, maze.physics_client)

    # Verify determinism.
    s1 = placer.place(maze, seed=42)
    placer.remove_all(s1, maze.physics_client)
    s2 = placer.place(maze, seed=42)
    placer.remove_all(s2, maze.physics_client)

    assert len(s1) == len(s2), "Determinism check failed: different counts"
    for a, b in zip(s1, s2):
        assert a.posture == b.posture, f"Posture mismatch: {a.posture} vs {b.posture}"
        assert a.cell == b.cell, f"Cell mismatch: {a.cell} vs {b.cell}"
        assert abs(a.yaw_deg - b.yaw_deg) < 1e-9, "Yaw mismatch"
        assert abs(a.scale - b.scale) < 1e-9, "Scale mismatch"
    print("\n[OK] Determinism check passed (seed=42 reproduces identical results)")

    p.disconnect(maze.physics_client)
    print("\n=== Done ===")


if __name__ == "__main__":
    _main()
