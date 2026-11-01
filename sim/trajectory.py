"""
sim/trajectory.py — Drone pose sampling and trajectory generation
================================================================

Generates camera/drone poses inside PyBullet maze layouts according to physical
constraints defined in ``configs/arena.yaml``:
- Flight altitude and tilt ranges from ``arena.yaml``.
- Yaw aligned predominantly with corridor axes plus configurable jitter.
- Small roll jitter (and optional banking during turns).
- Strict collision/safety distance checks rejecting any pose too close to walls.
- Survivor checks (when survivor 3D boxes are given): a minimum 3D distance from
  the camera to every survivor box, and a rejection when one survivor's
  projected box covers too much of the image. Both values come from the
  ``trajectory`` section of ``arena.yaml``.
- Independent pose sampling (``sample_poses``): the source of training frames.
- Consecutive sequence generator (``generate_sequence``) for replay and
  tracking tests only: a self-avoiding walk over the corridor graph that stops
  at a survivor's clearance boundary (optionally rotating in place) and never
  returns fewer than ``trajectory.min_sequence_frames`` frames.

Usage::

    from sim.trajectory import TrajectorySampler
    sampler = TrajectorySampler("sim/layouts/loop.yaml")

    # Sample random individual poses (the source of training frames):
    sampler = TrajectorySampler("sim/layouts/loop.yaml", survivor_boxes=[s.aabb for s in survivors])
    poses = sampler.sample_poses(n=10, seed=42)

    # Generate a smooth consecutive flight sequence:
    sequence = sampler.generate_sequence(num_frames=60, step_m=0.08, seed=42)
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from cv.config import load_config
from cv.logger import get_logger
from sim.camera import CameraPose, PinholeCamera
from sim.maze import FREE_CHARS, WALL_CHAR, load_layout

DEFAULT_ARENA_CONFIG = Path("configs/arena.yaml")

AABB = tuple[tuple[float, float, float], tuple[float, float, float]]

# Pose rejection reasons (counted and logged by sample_poses).
REJECT_WALL = "wall"
REJECT_SURVIVOR_CLEARANCE = "survivor_clearance"
REJECT_BOX_TOO_LARGE = "box_too_large"
REJECT_DUPLICATE = "duplicate"
REJECTION_REASONS = (REJECT_WALL, REJECT_SURVIVOR_CLEARANCE, REJECT_BOX_TOO_LARGE, REJECT_DUPLICATE)

log = get_logger("trajectory")


class PoseSamplingError(RuntimeError):
    """sample_poses could not find the requested number of valid unique poses."""


class SequenceSkipped(RuntimeError):
    """generate_sequence could not produce the minimum number of frames."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def pose_key(pose: CameraPose) -> tuple[float, float, int]:
    """Identity of a pose for duplicate checks: (x, y) to 1 cm, yaw to 1 degree."""
    return (round(pose.x, 2), round(pose.y, 2), round(pose.yaw) % 360)


def point_to_aabb_distance(point: tuple[float, float, float], aabb: AABB) -> float:
    """Euclidean distance from a 3D point to a box (0.0 inside the box)."""
    lo, hi = aabb
    d = [max(lo[i] - point[i], 0.0, point[i] - hi[i]) for i in range(3)]
    return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


# --------------------------------------------------------------------------- #
# Trajectory Sampler
# --------------------------------------------------------------------------- #
class TrajectorySampler:
    """
    Samples valid collision-free poses and smooth trajectories in a maze.

    Parameters
    ----------
    layout : str, Path, or list[str]
        Path to layout YAML or pre-loaded ASCII grid strings.
    arena_config_path : str or Path
        Path to arena.yaml.
    min_wall_dist : float, optional
        Minimum clearance distance in metres from any wall surface (safety buffer).
        Default is 0.35m.
    survivor_boxes : sequence of AABB, optional
        Real 3D boxes of the placed survivors (``SurvivorInfo.aabb``). Without
        them no survivor checks are made.
    """

    def __init__(
        self,
        layout: str | Path | list[str],
        arena_config_path: str | Path = DEFAULT_ARENA_CONFIG,
        min_wall_dist: float = 0.35,
        survivor_boxes: Sequence[AABB] | None = None,
    ) -> None:
        if isinstance(layout, list):
            self.grid = list(layout)
        else:
            self.grid = load_layout(layout)

        self.num_rows = len(self.grid)
        self.num_cols = len(self.grid[0])

        cfg = load_config(arena_config_path)
        self.cell_size: float = float(cfg.arena.corridor_width)
        self.wall_height: float = float(cfg.arena.wall_height)
        self.nominal_height: float = float(cfg.camera.height)
        self.nominal_tilt: float = float(cfg.camera.tilt_deg)
        self.min_wall_dist: float = float(min_wall_dist)
        self.survivor_clearance_m: float = float(cfg.trajectory.survivor_clearance_m)
        self.max_survivor_box_fraction: float = float(cfg.trajectory.max_survivor_box_fraction)
        self.min_sequence_frames: int = int(cfg.trajectory.min_sequence_frames)
        self.rotate_in_place: bool = bool(cfg.trajectory.rotate_in_place)
        self.rotate_yaw_step_deg: float = float(cfg.trajectory.rotate_yaw_step_deg)
        self.rotate_max_frames: int = int(cfg.trajectory.rotate_max_frames)
        # Projection only (no rendering): same FOV and aspect as the renders.
        self.camera = PinholeCamera(arena_config_path)
        self.survivor_boxes: list[AABB] = []
        self.set_survivor_boxes(survivor_boxes)

        # Precompute free cells and wall bounding boxes
        self.free_cells: list[tuple[int, int]] = []
        self.start_cell: tuple[int, int] | None = None
        self.wall_cells: list[tuple[int, int]] = []

        for r in range(self.num_rows):
            for c in range(self.num_cols):
                ch = self.grid[r][c]
                if ch in FREE_CHARS:
                    self.free_cells.append((r, c))
                    if ch == "S":
                        self.start_cell = (r, c)
                else:
                    self.wall_cells.append((r, c))

        if not self.free_cells:
            raise ValueError("Layout has no free corridor cells.")

    def set_survivor_boxes(self, boxes: Sequence[AABB] | None) -> None:
        """Replace the survivor 3D boxes used by the survivor checks."""
        self.survivor_boxes = [
            (
                (float(lo[0]), float(lo[1]), float(lo[2])),
                (float(hi[0]), float(hi[1]), float(hi[2])),
            )
            for lo, hi in (boxes or [])
        ]

    # -- Coordinate Transformations & Collision Checking --

    def cell_to_world_center(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid cell (row, col) to world center coordinates (x, y)."""
        x = (col + 0.5) * self.cell_size
        y = (row + 0.5) * self.cell_size
        return (x, y)

    def is_pose_valid(self, x: float, y: float, z: float) -> bool:
        """
        Check if (x, y, z) keeps min_wall_dist from walls and survivor_clearance_m
        from every survivor box.
        """
        return self.position_rejection(x, y, z) is None

    def position_rejection(self, x: float, y: float, z: float) -> str | None:
        """Reason the camera position is rejected, or None if it is valid."""
        if not self._clear_of_walls(x, y, z):
            return REJECT_WALL
        for box in self.survivor_boxes:
            if point_to_aabb_distance((x, y, z), box) < self.survivor_clearance_m:
                return REJECT_SURVIVOR_CLEARANCE
        return None

    def pose_rejection(self, pose: CameraPose) -> str | None:
        """
        Reason the full pose is rejected, or None if it is valid.

        Checks walls (with altitude and world bounds), survivor clearance, then
        whether any survivor's projected 3D box covers more than
        max_survivor_box_fraction of the image. The projected AABB is
        conservative for rotated bodies (see ``PinholeCamera.project_aabb``).
        """
        reason = self.position_rejection(pose.x, pose.y, pose.z)
        if reason is not None:
            return reason
        if self.survivor_boxes:
            fractions = self.camera.projected_box_fractions(pose, self.survivor_boxes)
            if max(fractions) > self.max_survivor_box_fraction:
                return REJECT_BOX_TOO_LARGE
        return None

    def _clear_of_walls(self, x: float, y: float, z: float) -> bool:
        """Altitude, world bounds and min_wall_dist clearance from wall cells."""
        # Vertical bounds check
        if z < 0.2 or z > (self.wall_height - 0.2):
            return False

        # World bounds check
        total_w = self.num_cols * self.cell_size
        total_l = self.num_rows * self.cell_size
        if (
            x < self.min_wall_dist
            or x > (total_w - self.min_wall_dist)
            or y < self.min_wall_dist
            or y > (total_l - self.min_wall_dist)
        ):
            return False

        # Grid cell query
        c = int(x // self.cell_size)
        r = int(y // self.cell_size)
        if not (0 <= r < self.num_rows and 0 <= c < self.num_cols):
            return False

        if self.grid[r][c] == WALL_CHAR:
            return False

        # Exact distance check to neighboring wall boundaries
        # A wall cell at (wr, wc) spans [wc*cs, (wc+1)*cs] x [wr*cs, (wr+1)*cs]
        # Check 3x3 neighborhood around (r, c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.num_rows and 0 <= nc < self.num_cols:
                    if self.grid[nr][nc] == WALL_CHAR:
                        w_min_x = nc * self.cell_size
                        w_max_x = (nc + 1) * self.cell_size
                        w_min_y = nr * self.cell_size
                        w_max_y = (nr + 1) * self.cell_size

                        # Closest point on wall box to (x, y)
                        cx = min(max(x, w_min_x), w_max_x)
                        cy = min(max(y, w_min_y), w_max_y)

                        dist = math.hypot(x - cx, y - cy)
                        if dist < self.min_wall_dist:
                            return False

        return True

    # -- Yaw Alignment Along Corridors --

    def get_corridor_headings(self, row: int, col: int) -> list[float]:
        """
        Determine nominal cardinal heading angles (in degrees) along corridors
        from cell (row, col).
        """
        headings: list[float] = []
        # Check 4 cardinal neighbors
        # East (+x): col + 1 -> 0 deg
        if col + 1 < self.num_cols and self.grid[row][col + 1] in FREE_CHARS:
            headings.append(0.0)
        # North (+y): row + 1 -> 90 deg
        if row + 1 < self.num_rows and self.grid[row + 1][col] in FREE_CHARS:
            headings.append(90.0)
        # West (-x): col - 1 -> 180 deg
        if col - 1 >= 0 and self.grid[row][col - 1] in FREE_CHARS:
            headings.append(180.0)
        # South (-y): row - 1 -> 270 deg
        if row - 1 >= 0 and self.grid[row - 1][col] in FREE_CHARS:
            headings.append(270.0)

        return headings if headings else [0.0]

    # -- Discrete Pose Sampling --

    def _draw_pose(
        self,
        rng: random.Random,
        height_range: tuple[float, float],
        pitch_range: tuple[float, float],
        yaw_jitter_deg: float,
        roll_jitter_deg: float,
        centerline_bias: bool,
    ) -> CameraPose:
        """Draw one candidate pose in a random free cell (not checked)."""
        r, c = rng.choice(self.free_cells)
        cx, cy = self.cell_to_world_center(r, c)

        # Small position perturbation within cell
        max_offset = max(0.0, (self.cell_size / 2.0) - self.min_wall_dist)
        if centerline_bias and max_offset > 0.0:
            sigma = max_offset / 3.0
            dx = min(max(rng.gauss(0.0, sigma), -max_offset), max_offset)
            dy = min(max(rng.gauss(0.0, sigma), -max_offset), max_offset)
        else:
            dx = rng.uniform(-max_offset, max_offset)
            dy = rng.uniform(-max_offset, max_offset)

        pz = rng.uniform(height_range[0], height_range[1])

        # Yaw: corridor heading + jitter
        base_yaw = rng.choice(self.get_corridor_headings(r, c))
        yaw = (base_yaw + rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)) % 360.0

        pitch = rng.uniform(pitch_range[0], pitch_range[1])
        roll = rng.uniform(-roll_jitter_deg, roll_jitter_deg)

        return CameraPose(x=cx + dx, y=cy + dy, z=pz, yaw=yaw, pitch=pitch, roll=roll)

    def sample_poses(
        self,
        n: int,
        seed: int | None = None,
        height_range: tuple[float, float] = (0.6, 1.0),
        pitch_range: tuple[float, float] = (-35.0, -25.0),
        yaw_jitter_deg: float = 12.0,
        roll_jitter_deg: float = 3.0,
        centerline_bias: bool = True,
        max_attempts_per_pose: int = 200,
    ) -> list[CameraPose]:
        """
        Sample n valid, unique, independent camera poses.

        This is the source of training frames. Every pose passes
        :meth:`pose_rejection` (walls, survivor clearance, projected box size)
        and no two poses share a :func:`pose_key`. The result depends only on
        the seed, the arguments, the layout and the survivor boxes. Rejection
        counts by reason are logged at INFO level on every call.

        Parameters
        ----------
        centerline_bias : bool
            If True, drone position offsets are concentrated towards the
            corridor centerline using a Gaussian distribution, reflecting
            natural quadcopter navigation rather than wall-hugging.
        max_attempts_per_pose : int
            Candidate budget is ``n * max_attempts_per_pose``.

        Raises
        ------
        PoseSamplingError
            If n poses are not found within the budget. The message gives the
            rejection counts by reason.
        """
        rng = random.Random(seed)
        rejected: Counter[str] = Counter({reason: 0 for reason in REJECTION_REASONS})
        poses: list[CameraPose] = []
        keys: set[tuple[float, float, int]] = set()
        attempts = 0
        budget = max(1, n * max_attempts_per_pose)

        while len(poses) < n and attempts < budget:
            attempts += 1
            pose = self._draw_pose(
                rng, height_range, pitch_range, yaw_jitter_deg, roll_jitter_deg, centerline_bias
            )
            reason = self.pose_rejection(pose)
            if reason is None and pose_key(pose) in keys:
                reason = REJECT_DUPLICATE
            if reason is not None:
                rejected[reason] += 1
                continue
            keys.add(pose_key(pose))
            poses.append(pose)

        counts = ", ".join(f"{reason}={rejected[reason]}" for reason in REJECTION_REASONS)
        log.info(
            f"sample_poses: n={n} seed={seed} accepted={len(poses)} attempts={attempts} "
            f"survivors={len(self.survivor_boxes)} rejected: {counts}"
        )
        if len(poses) < n:
            raise PoseSamplingError(
                f"sample_poses found {len(poses)} of {n} poses in {attempts} attempts "
                f"(seed={seed}); rejected: {counts}"
            )
        return poses

    # -- Consecutive Sequence Generation (replay and tracking tests only) --

    def _neighbours(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """Free 4-neighbours of a cell, in a fixed order."""
        r, c = cell
        out = []
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.num_rows and 0 <= nc < self.num_cols:
                if self.grid[nr][nc] in FREE_CHARS:
                    out.append((nr, nc))
        return out

    def _segment_points(
        self, a: tuple[int, int], b: tuple[int, int], step_m: float
    ) -> list[tuple[float, float]]:
        """Points from the centre of cell a (excluded) to the centre of cell b (included)."""
        x0, y0 = self.cell_to_world_center(*a)
        x1, y1 = self.cell_to_world_center(*b)
        n_sub = max(1, int(round(math.hypot(x1 - x0, y1 - y0) / step_m)))
        return [
            (x0 + (x1 - x0) * k / n_sub, y0 + (y1 - y0) * k / n_sub) for k in range(1, n_sub + 1)
        ]

    def _walk(
        self,
        start: tuple[int, int],
        rng: random.Random,
        num_frames: int,
        min_frames: int,
        step_m: float,
        noise: tuple[float, float, float, float],
    ) -> tuple[list[CameraPose], str]:
        """
        One self-avoiding walk over the corridor graph from ``start``.

        Returns the poses and why the walk ended.
        """
        height_noise, pitch_noise, yaw_noise, roll_noise = noise
        poses: list[CameraPose] = []
        visited = {start}
        cur = start
        z = self.nominal_height
        pitch = -self.nominal_tilt
        yaw: float | None = None

        def make_pose(px: float, py: float, target_yaw: float) -> CameraPose:
            nonlocal z, pitch, yaw
            if yaw is None:
                yaw = target_yaw
            else:
                # Smooth angle interpolation (shortest angular distance)
                diff = (target_yaw - yaw + 180.0) % 360.0 - 180.0
                yaw = (yaw + 0.25 * diff) % 360.0
            frame_yaw = (yaw + rng.uniform(-yaw_noise, yaw_noise)) % 360.0
            z = min(max(z + rng.uniform(-height_noise, height_noise), 0.4), self.wall_height - 0.4)
            pitch = min(max(pitch + rng.uniform(-pitch_noise, pitch_noise), -45.0), -15.0)
            # Bank roll slightly into turns
            delta_yaw = (target_yaw - yaw + 180.0) % 360.0 - 180.0
            roll = -0.15 * delta_yaw + rng.uniform(-roll_noise, roll_noise)
            return CameraPose(x=px, y=py, z=z, yaw=frame_yaw, pitch=pitch, roll=roll)

        while True:
            candidates = [n for n in self._neighbours(cur) if n not in visited]
            if not candidates:
                return poses, "no unvisited continuation"
            clear = [
                n
                for n in candidates
                if all(
                    self.position_rejection(px, py, z) is None
                    for px, py in self._segment_points(cur, n, step_m)
                )
            ]
            # Until the minimum length is reached, avoid heading into a survivor.
            nxt = rng.choice(clear if clear and len(poses) < min_frames else candidates)

            cx, cy = self.cell_to_world_center(*cur)
            nx, ny = self.cell_to_world_center(*nxt)
            heading = math.degrees(math.atan2(ny - cy, nx - cx)) % 360.0
            points = self._segment_points(cur, nxt, step_m)
            if not poses:
                points = [(cx, cy)] + points

            for px, py in points:
                pose = make_pose(px, py, heading)
                reason = self.pose_rejection(pose)
                if reason is not None:
                    # Stop at the boundary: the last accepted pose.
                    if poses and reason in (REJECT_SURVIVOR_CLEARANCE, REJECT_BOX_TOO_LARGE):
                        poses.extend(self._rotate_in_place(poses[-1], rng, num_frames - len(poses)))
                    return poses, f"blocked ({reason})"
                poses.append(pose)
                if len(poses) >= num_frames:
                    return poses, "num_frames reached"

            visited.add(nxt)
            cur = nxt

    def _rotate_in_place(self, last: CameraPose, rng: random.Random, room: int) -> list[CameraPose]:
        """Yaw steps in place from ``last``; stops at the first rejected pose."""
        if not self.rotate_in_place or room <= 0:
            return []
        # Never come back to a yaw already used at this position.
        max_steps = max(0, math.ceil(360.0 / self.rotate_yaw_step_deg) - 1)
        n = min(self.rotate_max_frames, max_steps, room)
        direction = rng.choice((-1.0, 1.0))
        out = []
        for k in range(1, n + 1):
            pose = CameraPose(
                x=last.x,
                y=last.y,
                z=last.z,
                yaw=(last.yaw + direction * k * self.rotate_yaw_step_deg) % 360.0,
                pitch=last.pitch,
                roll=last.roll,
            )
            if self.pose_rejection(pose) is not None:
                break
            out.append(pose)
        return out

    def generate_sequence(
        self,
        num_frames: int = 30,
        step_m: float = 0.08,
        seed: int | None = None,
        height_noise: float = 0.03,
        pitch_noise: float = 1.5,
        yaw_noise: float = 3.0,
        roll_noise: float = 1.5,
        start_cell: tuple[int, int] | None = None,
        min_frames: int | None = None,
        max_start_attempts: int = 20,
    ) -> list[CameraPose]:
        """
        Generate consecutive camera poses along the corridor graph.

        For replay and tracking tests only; training frames come from
        :meth:`sample_poses`.

        The walk moves cell to cell through free 4-neighbours and never enters
        a visited cell, so positions are not repeated. Every pose passes
        :meth:`pose_rejection`. When the next pose would be too close to a
        survivor (or its box too large in the image), the sequence stops at the
        last accepted pose and, if ``trajectory.rotate_in_place`` is set, adds
        up to ``rotate_max_frames`` in-place yaw steps of
        ``rotate_yaw_step_deg`` (distinct frames). It also ends when there is
        no unvisited neighbour or when ``num_frames`` is reached.

        Start cells are tried in an order drawn from ``seed`` (or only
        ``start_cell`` if given); the first walk reaching ``num_frames`` is
        returned, otherwise the longest. Same seed, layout and survivor boxes
        give the same sequence.

        Parameters
        ----------
        num_frames : int
            Maximum number of frames.
        step_m : float
            Distance step in metres between consecutive frames.
        seed : int, optional
            Random seed for repeatability.
        height_noise, pitch_noise, yaw_noise, roll_noise : float
            Per-frame perturbations (metres, degrees).
        start_cell : tuple[int, int], optional
            Fixed start cell; otherwise valid cells are tried in seeded order.
        min_frames : int, optional
            Minimum length; defaults to ``trajectory.min_sequence_frames``
            (capped at ``num_frames``).
        max_start_attempts : int
            Maximum number of start cells tried.

        Returns
        -------
        list[CameraPose]
            Between ``min_frames`` and ``num_frames`` consecutive poses.

        Raises
        ------
        SequenceSkipped
            If no start cell gives ``min_frames`` frames. ``reason`` says why.
        """
        rng = random.Random(seed)
        min_len = min(num_frames, self.min_sequence_frames if min_frames is None else min_frames)
        noise = (height_noise, pitch_noise, yaw_noise, roll_noise)

        if start_cell is not None:
            starts = [start_cell]
        else:
            starts = [
                cell
                for cell in self.free_cells
                if self.position_rejection(*self.cell_to_world_center(*cell), self.nominal_height)
                is None
            ]
            rng.shuffle(starts)
            starts = starts[:max_start_attempts]

        best: tuple[list[CameraPose], str, tuple[int, int]] | None = None
        for start in starts:
            walk_rng = random.Random(rng.getrandbits(64))
            poses, end = self._walk(start, walk_rng, num_frames, min_len, step_m, noise)
            if best is None or len(poses) > len(best[0]):
                best = (poses, end, start)
            if len(poses) >= num_frames:
                break

        if best is None or len(best[0]) < min_len:
            if best is None:
                reason = "no valid start cell"
            else:
                reason = (
                    f"longest of {len(starts)} start cells has {len(best[0])} < {min_len} "
                    f"frames (start {best[2]}, ended: {best[1]})"
                )
            log.warning(f"generate_sequence skipped: seed={seed}: {reason}")
            raise SequenceSkipped(reason)

        poses, end, start = best
        log.info(
            f"generate_sequence: seed={seed} frames={len(poses)}/{num_frames} start={start} "
            f"ended: {end}"
        )
        return poses


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="NIDAR Trajectory Sampler")
    parser.add_argument(
        "--layout",
        type=str,
        default="sim/layouts/loop.yaml",
        help="Path to layout YAML",
    )
    parser.add_argument("--samples", type=int, default=5, help="Number of random poses")
    parser.add_argument("--seq-len", type=int, default=10, help="Frames in smooth sequence")
    args = parser.parse_args()

    sampler = TrajectorySampler(args.layout)
    print("=" * 60)
    print("  NIDAR Trajectory & Pose Sampler")
    print("=" * 60)
    print(f"  Layout       : {args.layout}")
    print(f"  Free cells   : {len(sampler.free_cells)}")
    print(f"  Cell size    : {sampler.cell_size} m")
    print(f"  Wall clear   : {sampler.min_wall_dist} m")
    print()

    print(f"Sampled {args.samples} Discrete Poses:")
    poses = sampler.sample_poses(args.samples, seed=42)
    for i, p in enumerate(poses):
        print(
            f"  [{i}] pos=({p.x:.2f}, {p.y:.2f}, {p.z:.2f})m, "
            f"yaw={p.yaw:.1f}°, pitch={p.pitch:.1f}°, roll={p.roll:.1f}°"
        )
    print()

    print(f"Generated Smooth Sequence ({args.seq_len} consecutive frames):")
    seq = sampler.generate_sequence(num_frames=args.seq_len, seed=42)
    for i, p in enumerate(seq):
        print(
            f"  frame {i:02d}: pos=({p.x:.2f}, {p.y:.2f}, {p.z:.2f})m, "
            f"yaw={p.yaw:.1f}°, pitch={p.pitch:.1f}°, roll={p.roll:.1f}°"
        )


if __name__ == "__main__":
    main()
