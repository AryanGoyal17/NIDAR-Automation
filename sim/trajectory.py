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
- Smooth trajectory generator producing temporally consecutive sequences of
  poses suitable for multi-frame synthetic video generation.

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
from collections import Counter, deque
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

    # -- Smooth Trajectory Sequence Generation --

    def _find_random_walk_path(
        self,
        min_length: int = 5,
        rng: random.Random | None = None,
    ) -> list[tuple[int, int]]:
        """
        Find a connected sequence of adjacent free cells using self-avoiding
        random walk or BFS search.
        """
        if rng is None:
            rng = random.Random()

        start = self.start_cell if self.start_cell else rng.choice(self.free_cells)
        path = [start]
        visited = {start}

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for _ in range(min_length * 3):
            curr_r, curr_c = path[-1]
            candidates = []
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.num_rows and 0 <= nc < self.num_cols:
                    if self.grid[nr][nc] in FREE_CHARS and (nr, nc) not in visited:
                        candidates.append((nr, nc))

            if candidates:
                nxt = rng.choice(candidates)
                path.append(nxt)
                visited.add(nxt)
                if len(path) >= min_length:
                    break
            else:
                # If stuck, allow backtrack or stop
                break

        # If walk was too short, find a longest shortest-path via BFS
        if len(path) < 3:
            path = self._bfs_farthest_path(start)

        return path

    def _bfs_farthest_path(self, start: tuple[int, int]) -> list[tuple[int, int]]:
        """Find path to the farthest reachable cell from start using BFS."""
        queue = deque([[start]])
        visited = {start}
        longest_path = [start]

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while queue:
            curr_path = queue.popleft()
            if len(curr_path) > len(longest_path):
                longest_path = curr_path

            curr_r, curr_c = curr_path[-1]
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.num_rows and 0 <= nc < self.num_cols:
                    if self.grid[nr][nc] in FREE_CHARS and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(curr_path + [(nr, nc)])

        return longest_path

    def generate_sequence(
        self,
        num_frames: int = 30,
        step_m: float = 0.08,
        seed: int | None = None,
        height_noise: float = 0.03,
        pitch_noise: float = 1.5,
        yaw_noise: float = 3.0,
        roll_noise: float = 1.5,
    ) -> list[CameraPose]:
        """
        Generate a smooth consecutive sequence of camera poses along a corridor path.

        Parameters
        ----------
        num_frames : int
            Total number of consecutive poses/frames to generate.
        step_m : float
            Distance step in metres between consecutive frames.
        seed : int, optional
            Random seed for repeatability.
        height_noise : float
            Max altitude perturbation per frame.
        pitch_noise : float
            Max pitch perturbation in degrees.
        yaw_noise : float
            Max heading perturbation in degrees.
        roll_noise : float
            Max roll perturbation in degrees.

        Returns
        -------
        list[CameraPose]
            Ordered list of consecutive poses simulating smooth drone flight.
        """
        rng = random.Random(seed)
        cell_path = self._find_random_walk_path(min_length=max(4, num_frames // 8), rng=rng)

        # Convert cells to world waypoints
        waypoints = [self.cell_to_world_center(r, c) for r, c in cell_path]

        # Interpolate waypoints with dense sampling
        dense_points: list[tuple[float, float]] = []
        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i]
            x1, y1 = waypoints[i + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            n_sub = max(1, int(round(seg_len / step_m)))
            for s in range(n_sub):
                t = s / float(n_sub)
                px = x0 + t * (x1 - x0)
                py = y0 + t * (y1 - y0)
                dense_points.append((px, py))

        # Include final waypoint
        dense_points.append(waypoints[-1])

        # If dense_points has fewer points than num_frames, extend along direction
        while len(dense_points) < num_frames:
            # Repeat or oscillate
            dense_points.extend(reversed(dense_points[:-1]))

        # Trim to exactly num_frames
        points = dense_points[:num_frames]

        # Compute smooth poses
        poses: list[CameraPose] = []
        nominal_z = self.nominal_height
        nominal_pitch = -self.nominal_tilt

        curr_z = nominal_z
        curr_pitch = nominal_pitch
        curr_yaw = 0.0

        for i in range(len(points)):
            px, py = points[i]

            # Compute heading from path tangent
            if i < len(points) - 1:
                dx = points[i + 1][0] - px
                dy = points[i + 1][1] - py
            else:
                dx = px - points[i - 1][0]
                dy = py - points[i - 1][1]

            target_yaw = math.degrees(math.atan2(dy, dx)) % 360.0

            if i == 0:
                curr_yaw = target_yaw
            else:
                # Smooth angle interpolation (shortest angular distance)
                diff = (target_yaw - curr_yaw + 180.0) % 360.0 - 180.0
                curr_yaw = (curr_yaw + 0.25 * diff) % 360.0

            # Add smooth small perturbations
            jitter_y = rng.uniform(-yaw_noise, yaw_noise)
            frame_yaw = (curr_yaw + jitter_y) % 360.0

            curr_z += rng.uniform(-height_noise, height_noise)
            curr_z = min(max(curr_z, 0.4), self.wall_height - 0.4)

            curr_pitch += rng.uniform(-pitch_noise, pitch_noise)
            curr_pitch = min(max(curr_pitch, -45.0), -15.0)

            # Bank roll slightly into turns
            delta_yaw = (target_yaw - curr_yaw + 180.0) % 360.0 - 180.0
            bank_roll = -0.15 * delta_yaw
            frame_roll = bank_roll + rng.uniform(-roll_noise, roll_noise)

            poses.append(
                CameraPose(
                    x=px,
                    y=py,
                    z=curr_z,
                    yaw=frame_yaw,
                    pitch=curr_pitch,
                    roll=frame_roll,
                )
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
