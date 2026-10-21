#!/usr/bin/env python3
"""
scripts/check_sim.py — PyBullet Simulation Smoke Test

Verifies that PyBullet is functional in headless (DIRECT) mode:
1. Connects to PyBullet using p.DIRECT.
2. Creates a ground plane and a colored 3D box object.
3. Computes synthetic camera view and projection matrices.
4. Renders a synthetic camera frame via p.getCameraImage.
5. Saves the rendered image to outputs/sim_check.png and verifies its integrity.

Usage:
    python scripts/check_sim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

OUTPUT_PATH = Path("outputs/sim_check.png")
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def main() -> int:
    print("=" * 64)
    print("       NIDAR AIRMOUSE — PYBULLET SIMULATION SMOKE TEST")
    print("=" * 64)

    # 1. Connect in headless DIRECT mode
    physics_client = p.connect(p.DIRECT)
    if physics_client < 0:
        print("[FAIL] Could not connect to PyBullet in DIRECT mode.")
        return 1
    print("  [OK] Connected to PyBullet (DIRECT headless mode)")

    try:
        # Set search path for default URDF assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # 2. Add ground plane
        plane_id = p.loadURDF("plane.urdf")
        print(f"  [OK] Loaded ground plane (URDF id={plane_id})")

        # 3. Add a colored 3D box (simulating a maze block / obstacle)
        half_extents = [0.5, 0.5, 0.5]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.85, 0.25, 0.20, 1.0],  # Red box
            specularColor=[0.4, 0.4, 0.4],
        )
        box_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[0.0, 0.0, 0.5],
        )
        print(f"  [OK] Created 3D box obstacle (body id={box_id}) at [0.0, 0.0, 0.5]")

        # 4. Camera matrices (looking at the box from an elevated diagonal angle)
        cam_target = [0.0, 0.0, 0.5]
        cam_dist = 2.5
        yaw = 45.0
        pitch = -25.0
        roll = 0.0
        up_axis_index = 2

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target,
            distance=cam_dist,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=up_axis_index,
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(IMAGE_WIDTH) / float(IMAGE_HEIGHT),
            nearVal=0.1,
            farVal=100.0,
        )

        # 5. Render image
        width, height, rgb_data, depth_data, seg_data = p.getCameraImage(
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        print(f"  [OK] Rendered frame with getCameraImage: {width}x{height}")

        # 6. Convert and save image
        rgba_array = np.reshape(np.array(rgb_data, dtype=np.uint8), (height, width, 4))
        rgb_array = rgba_array[:, :, :3]  # Drop alpha channel for standard RGB PNG

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(rgb_array)
        img.save(OUTPUT_PATH)
        print(f"  [OK] Saved rendered frame to: {OUTPUT_PATH.resolve()}")

        # 7. Verification
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
            print("[FAIL] Rendered image file is missing or zero-sized.")
            return 1

        reloaded = Image.open(OUTPUT_PATH)
        print(f"  [OK] Verified saved image: format={reloaded.format}, size={reloaded.size}, mode={reloaded.mode}")

    finally:
        p.disconnect(physics_client)
        print("  [OK] Disconnected PyBullet session cleanly")

    print("=" * 64)
    print("  ALL CHECKS PASSED — PyBullet simulation rendering is ready.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
