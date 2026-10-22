"""
tests/test_sim/test_camera.py — Unit tests for sim.camera PinholeCamera
======================================================================

Validates:
1. Camera loads resolution and FOV from configs/arena.yaml.
2. View matrix produces valid 16-element PyBullet format.
3. Projection matrix produces valid 16-element PyBullet format.
4. Render returns BGR uint8 (H, W, 3), depth float32 (H, W), seg int32 (H, W).
5. Benchmark method executes and returns positive FPS.
"""

from __future__ import annotations

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
