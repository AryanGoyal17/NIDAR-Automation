"""
scripts/sim_preview.py — Generate 24-frame simulation contact sheet
===================================================================

Renders 24 camera viewpoints across different hand-made and procedurally
generated maze layouts and stitches them into a single contact-sheet image
at ``outputs/sim_preview.png``.

Features:
- Samples from 6 hand-made layouts (straight, L-turn, T-junction, dead end,
  two rooms, loop) and 2 procedurally generated recursive backtracker mazes.
- Uses PinholeCamera with arena.yaml parameters.
- Uses TrajectorySampler for valid, collision-free viewpoints.
- Assembles a clean 4x6 grid contact sheet with viewpoint labels.

Usage::

    python scripts/sim_preview.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import pybullet as p

from sim.camera import PinholeCamera
from sim.layout_gen import MazeGenerator
from sim.maze import MazeBuilder
from sim.trajectory import TrajectorySampler

OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "sim_preview.png"
ARENA_CONFIG = Path("configs/arena.yaml")

HANDMADE_LAYOUTS = [
    ("straight_corridor", Path("sim/layouts/straight_corridor.yaml")),
    ("l_turn", Path("sim/layouts/l_turn.yaml")),
    ("t_junction", Path("sim/layouts/t_junction.yaml")),
    ("dead_end", Path("sim/layouts/dead_end.yaml")),
    ("two_rooms", Path("sim/layouts/two_rooms.yaml")),
    ("loop", Path("sim/layouts/loop.yaml")),
]


def render_layout_frames(
    builder: MazeBuilder,
    camera: PinholeCamera,
    layout_name: str,
    layout_source: Path | list[str],
    num_frames: int,
    seed: int,
    physics_client: int,
) -> list[tuple[str, np.ndarray]]:
    """Build layout in PyBullet, sample collision-free poses, and render frames."""
    # Build maze
    if isinstance(layout_source, list):
        maze_res = builder.build_from_grid(layout_source, physics_client=physics_client)
    else:
        maze_res = builder.build(layout_source, physics_client=physics_client)

    sampler = TrajectorySampler(maze_res.grid, arena_config_path=ARENA_CONFIG)
    poses = sampler.sample_poses(
        n=num_frames,
        seed=seed,
        pitch_range=(-28.0, -20.0),
        centerline_bias=True,
    )

    rendered: list[tuple[str, np.ndarray]] = []
    for idx, pose in enumerate(poses):
        out = camera.render(pose, physics_client=physics_client)
        frame_bgr = out.bgr.copy()

        # Add visual label overlay
        label = f"{layout_name} #{idx+1} [yaw={pose.yaw:.0f} p={pose.pitch:.0f}]"
        cv2.rectangle(frame_bgr, (5, 5), (5 + len(label) * 9 + 10, 28), (20, 20, 20), -1)
        cv2.putText(
            frame_bgr,
            label,
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        rendered.append((label, frame_bgr))

    # Clean up physics bodies for next layout
    p.resetSimulation(physicsClientId=physics_client)

    return rendered


def generate_contact_sheet(
    frames: list[np.ndarray],
    grid_rows: int = 4,
    grid_cols: int = 6,
    border_px: int = 3,
) -> np.ndarray:
    """Combine individual frame images into a single grid image."""
    h, w, _ = frames[0].shape
    border_color = [45, 45, 45]  # Dark grey border

    # Add borders around each tile
    bordered = [
        cv2.copyMakeBorder(
            f,
            border_px,
            border_px,
            border_px,
            border_px,
            cv2.BORDER_CONSTANT,
            value=border_color,
        )
        for f in frames
    ]

    tile_h, tile_w, _ = bordered[0].shape

    sheet = np.zeros(
        (grid_rows * tile_h, grid_cols * tile_w, 3),
        dtype=np.uint8,
    )

    for idx, tile in enumerate(bordered):
        if idx >= grid_rows * grid_cols:
            break
        r = idx // grid_cols
        c = idx % grid_cols
        sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 24-frame sim preview contact sheet")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="Output PNG path")
    parser.add_argument("--width", type=int, default=480, help="Per-tile width")
    parser.add_argument("--height", type=int, default=270, help="Per-tile height")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  NIDAR Simulation Multi-Layout Preview Generator")
    print("=" * 65)
    print(f"  Target output : {out_path.resolve()}")
    print(f"  Tile size     : {args.width} × {args.height}")
    print("  Collecting layouts...")

    # Plan: 6 hand-made layouts (3 frames each = 18 frames)
    #     + 2 generated mazes (3 frames each = 6 frames)
    # Total = 24 frames
    layouts_to_render: list[tuple[str, Path | list[str], int, int]] = []

    # 1. Hand-made layouts
    for name, path in HANDMADE_LAYOUTS:
        layouts_to_render.append((name, path, 3, 100))

    # 2. Procedural generated mazes
    gen = MazeGenerator(arena_config_path=ARENA_CONFIG)
    gen_42 = gen.generate(rows=11, cols=11, seed=42)
    gen_101 = gen.generate(rows=13, cols=13, seed=101)

    layouts_to_render.append(("gen_maze_42", gen_42.grid, 3, 42))
    layouts_to_render.append(("gen_maze_101", gen_101.grid, 3, 101))

    total_expected = sum(item[2] for item in layouts_to_render)
    print(f"  Total frames to render: {total_expected} across {len(layouts_to_render)} layouts")

    cid = p.connect(p.DIRECT)
    builder = MazeBuilder(arena_config_path=ARENA_CONFIG)
    camera = PinholeCamera(
        arena_config_path=ARENA_CONFIG,
        width=args.width,
        height=args.height,
    )

    all_frames: list[np.ndarray] = []
    t_start = time.perf_counter()

    try:
        for name, src, num, seed in layouts_to_render:
            print(f"  -> Rendering {num} frames for: {name}...")
            rendered = render_layout_frames(
                builder=builder,
                camera=camera,
                layout_name=name,
                layout_source=src,
                num_frames=num,
                seed=seed,
                physics_client=cid,
            )
            for _, img in rendered:
                all_frames.append(img)
    finally:
        p.disconnect(cid)

    elapsed = time.perf_counter() - t_start
    print(f"  Rendered {len(all_frames)} frames in {elapsed:.2f}s ({len(all_frames)/elapsed:.1f} FPS)")

    # Build 4x6 contact sheet
    print("  Assembling 4×6 contact sheet...")
    sheet = generate_contact_sheet(all_frames, grid_rows=4, grid_cols=6)

    cv2.imwrite(str(out_path), sheet)
    print("=" * 65)
    print(f"  SUCCESS: Contact sheet saved to:\n  {out_path.resolve()}")
    print(f"  Image size: {sheet.shape[1]} × {sheet.shape[0]} px")
    print("=" * 65)


if __name__ == "__main__":
    main()
