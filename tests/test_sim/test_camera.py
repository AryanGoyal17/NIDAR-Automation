"""
tests/test_sim/test_camera.py — Unit tests for sim.camera PinholeCamera
======================================================================

Validates:
1. Camera loads resolution and FOV from configs/arena.yaml.
2. View matrix produces valid 16-element PyBullet format.
3. Projection matrix produces valid 16-element PyBullet format.
4. Render returns BGR uint8 (H, W, 3), depth float32 (H, W), seg int32 (H, W).
5. Benchmark method executes and returns positive FPS.
6. Renders of a fixed scene and fixed poses are byte-identical to recorded hashes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

from sim.camera import CameraPose, PinholeCamera

ARENA_CONFIG = Path("configs/arena.yaml")


@pytest.fixture(scope="module")
def pybullet_session():
    cid = p.connect(p.DIRECT)
    yield cid
    p.disconnect(cid)


class TestCameraInitialization:
    def test_loads_arena_config_defaults(self) -> None:
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG)
        assert cam.width == 1280
        assert cam.height == 720
        assert cam.vertical_fov_deg == pytest.approx(54.0)
        assert cam.near_clip == pytest.approx(0.1)
        assert cam.far_clip == pytest.approx(15.0)

    def test_override_resolution(self) -> None:
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=640, height=360)
        assert cam.width == 640
        assert cam.height == 360
        assert len(cam.projection_matrix) == 16


class TestMatrixComputation:
    def test_view_matrix_structure(self) -> None:
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=240)
        pose = CameraPose(x=1.0, y=2.0, z=0.8, yaw=45.0, pitch=-30.0, roll=5.0)
        vm = cam.compute_view_matrix(pose)
        assert len(vm) == 16
        assert all(isinstance(v, float) for v in vm)

    def test_view_matrix_from_tuple(self) -> None:
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=240)
        vm1 = cam.compute_view_matrix((1.0, 2.0, 0.8, 45.0, -30.0, 5.0))
        vm2 = cam.compute_view_matrix(CameraPose(1.0, 2.0, 0.8, 45.0, -30.0, 5.0))
        assert vm1 == pytest.approx(vm2)


class TestRendering:
    def test_render_output_shapes_and_types(self, pybullet_session: int) -> None:
        # Use low-res for fast test execution
        w, h = 160, 90
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=w, height=h)
        pose = CameraPose(x=0.0, y=0.0, z=1.0, yaw=0.0, pitch=-30.0)

        out = cam.render(pose, physics_client=pybullet_session)

        # 1. BGR image
        assert out.bgr.shape == (h, w, 3)
        assert out.bgr.dtype == np.uint8

        # 2. Depth buffer
        assert out.depth.shape == (h, w)
        assert out.depth.dtype == np.float32

        # 3. Segmentation mask
        assert out.seg_mask.shape == (h, w)
        assert out.seg_mask.dtype == np.int32

    def test_benchmark_returns_positive_fps(self, pybullet_session: int) -> None:
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=160, height=90)
        fps = cam.benchmark(num_frames=5, physics_client=pybullet_session)
        assert fps > 0.0

    def test_lit_floor_brighter_than_ambient_only(self, pybullet_session: int) -> None:
        """Verify that PyBullet lightDirection (z > 0 pointing to overhead light)
        produces direct illumination on the floor clearly above ambient-only."""
        import cv2

        from sim.labels import decode_segmentation_mask
        from sim.maze import MazeBuilder, load_layout

        layout_path = Path("sim/layouts/straight_corridor.yaml")
        grid = load_layout(layout_path)
        builder = MazeBuilder(arena_config_path=ARENA_CONFIG)
        maze = builder.build_from_grid(grid, physics_client=pybullet_session)

        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
        pose = CameraPose(x=2.25, y=2.25, z=0.8, yaw=0.0, pitch=-30.0)

        try:
            # 1. Lit render (default overhead directional light, z > 0)
            out_lit = cam.render(pose, physics_client=pybullet_session)
            gray_lit = cv2.cvtColor(out_lit.bgr, cv2.COLOR_BGR2GRAY)
            mask = decode_segmentation_mask(out_lit.seg_mask)
            floor_mask = mask == maze.floor_id

            assert floor_mask.any(), "Floor must be visible in render"
            mean_floor_lit = float(gray_lit[floor_mask].mean())

            # 2. Ambient-only render (diffuse_coeff=0.0)
            out_ambient = cam.render(pose, physics_client=pybullet_session, diffuse_coeff=0.0)
            gray_ambient = cv2.cvtColor(out_ambient.bgr, cv2.COLOR_BGR2GRAY)
            mean_floor_ambient = float(gray_ambient[floor_mask].mean())

            # In a properly lit scene, the floor grey should be clearly above ambient-only
            assert mean_floor_lit > mean_floor_ambient + 20.0, (
                f"Lit floor ({mean_floor_lit:.1f}) not clearly brighter than "
                f"ambient-only ({mean_floor_ambient:.1f})"
            )
        finally:
            p.resetSimulation(physicsClientId=pybullet_session)


# --------------------------------------------------------------------------- #
# Render regression pin
# --------------------------------------------------------------------------- #
# Fixed poses covering yaw, pitch (down to nadir) and roll in both directions.
PINNED_POSES = [
    CameraPose(x=1.0, y=0.0, z=0.8, yaw=0.0, pitch=-30.0, roll=0.0),
    CameraPose(x=1.5, y=-0.3, z=0.7, yaw=10.0, pitch=-45.0, roll=5.0),
    CameraPose(x=2.0, y=0.4, z=1.0, yaw=350.0, pitch=-15.0, roll=-8.0),
    CameraPose(x=3.0, y=-1.0, z=0.9, yaw=90.0, pitch=-30.0, roll=3.0),
    CameraPose(x=4.5, y=1.2, z=0.6, yaw=265.0, pitch=-60.0, roll=-12.0),
    CameraPose(x=5.5, y=0.0, z=1.2, yaw=180.0, pitch=-5.0, roll=20.0),
    CameraPose(x=3.5, y=0.0, z=1.5, yaw=135.0, pitch=-90.0, roll=0.0),
]

# sha256 of (bgr, depth, seg_mask) bytes per pose at 320x180, recorded on the
# camera code before the axis helper refactor (pybullet 3.2.7 built from source,
# dev laptop). Only regenerate (with _pinned_render_hashes) when the renderer
# itself changes, never to make a sim/camera.py change pass.
PINNED_HASHES = {
    0: "8eec5b7c2d4d4767927c4e589f0ab72d8e43c919bb432f7c249c0af015516e57",
    1: "bfae9ee8e1ee9eeb62e5d50423c4932ec37e7a17c3b14b6af1b6954854a1a61b",
    2: "ab2a5ff0d32c5e61cac9c14abdd06e008ab607a073fb54259f9ec286a8378b8c",
    3: "e2e8ebfe45df47a92223ab321f1603915a1ab89504d3a56d6cfea7876bd85575",
    4: "a295f14c71449458888c705eb707b4ffd6ba1e24e6d569221e2fea4655ccec1b",
    5: "2fb4784ab144bd414b549201b11e76d6d2eb4b2156b62c8a350c974d2d9d2df6",
    6: "906dfcdddbec93a52fa6d43993db58b96d606a94e5cef9817a5a8d94a49e0ab3",
}


def _build_pinned_scene(cid: int) -> None:
    """Floor, two walls, a person-sized box and a sphere, fixed colours."""

    def box(half: list[float], pos: list[float], rgb: list[float], yaw: float = 0.0) -> None:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=rgb + [1.0], physicsClientId=cid
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            physicsClientId=cid,
        )

    box([5.0, 3.0, 0.01], [3.0, 0.0, -0.01], [0.5, 0.5, 0.52])
    box([5.0, 0.05, 1.0], [3.0, 1.6, 1.0], [0.8, 0.8, 0.78])
    box([5.0, 0.05, 1.0], [3.0, -1.6, 1.0], [0.7, 0.75, 0.8])
    box([0.9, 0.25, 0.15], [3.5, 0.2, 0.15], [0.7, 0.15, 0.15], yaw=0.6)
    vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.12, rgbaColor=[0.9, 0.75, 0.6, 1.0], physicsClientId=cid
    )
    p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=vis, basePosition=[4.5, -0.2, 0.12], physicsClientId=cid
    )


def _pinned_render_hashes() -> dict[int, str]:
    cid = p.connect(p.DIRECT)
    try:
        _build_pinned_scene(cid)
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
        hashes = {}
        for i, pose in enumerate(PINNED_POSES):
            out = cam.render(pose, physics_client=cid)
            h = hashlib.sha256()
            for arr in (out.bgr, out.depth, out.seg_mask):
                h.update(np.ascontiguousarray(arr).tobytes())
            hashes[i] = h.hexdigest()
        return hashes
    finally:
        p.disconnect(cid)


class TestRenderPinned:
    def test_renders_match_recorded_hashes(self) -> None:
        assert _pinned_render_hashes() == PINNED_HASHES
