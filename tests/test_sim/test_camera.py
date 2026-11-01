"""
tests/test_sim/test_camera.py — Unit tests for sim.camera PinholeCamera
======================================================================

Validates:
1. Camera loads resolution and FOV from configs/arena.yaml.
2. View matrix produces valid 16-element PyBullet format.
3. Projection matrix produces valid 16-element PyBullet format.
4. Render returns BGR uint8 (H, W, 3), depth float32 (H, W), seg int32 (H, W).
5. Benchmark method executes and returns positive FPS.
6. Renders of a fixed scene and fixed poses match renders made with an
   independently computed view matrix (axis maths pin, portable across builds).
7. project_aabb matches the rendered segmentation box of an axis-aligned box.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

p = pytest.importorskip("pybullet")

from sim.camera import CameraPose, PinholeCamera  # noqa: E402

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

# The reference render uses the same build, lighting and projection as
# PinholeCamera.render, so only the view matrix (camera_axes and its use) is
# pinned, not lighting or projection changes. Threshold on the per-pose mean
# absolute BGR difference (0-255), measured 1 Nov 2026 at 320x180 (dev laptop,
# pybullet 3.2.7 built from source):
#   repeat render of the same pose:                         0.0
#   reference vs render (view matrices bit-identical):      0.0
#   view-matrix noise 1e-6 / 1e-5 relative (other build):  0.0017 / 0.0135
#   smallest change to catch, one axis at a time, least sensitive pose:
#     x +-1 cm 0.139, roll +-0.5 deg 0.183, yaw +-0.5 deg 0.285
# 0.05 is 3.7x above the 1e-5 noise and 2.8x below the weakest perturbation.
PIN_MAX_MAD = 0.05


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


def _reference_view_matrix(pose: CameraPose) -> list[float]:
    """View matrix from Rz(yaw) Ry(-pitch) Rx(roll) applied to fwd=+x, up=+z.

    Written independently of sim.camera.camera_axes; same convention.
    """
    cy, sy = math.cos(math.radians(pose.yaw)), math.sin(math.radians(pose.yaw))
    cp, sp = math.cos(math.radians(-pose.pitch)), math.sin(math.radians(-pose.pitch))
    cr, sr = math.cos(math.radians(pose.roll)), math.sin(math.radians(pose.roll))
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot = rz @ ry @ rx
    eye = np.array([pose.x, pose.y, pose.z])
    return list(p.computeViewMatrix(eye.tolist(), (eye + rot[:, 0]).tolist(), rot[:, 2].tolist()))


class TestRenderPinned:
    def test_renders_match_reference_view(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cid = p.connect(p.DIRECT)
        try:
            _build_pinned_scene(cid)
            cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
            mads = []
            for pose in PINNED_POSES:
                out = cam.render(pose, physics_client=cid)
                with monkeypatch.context() as m:
                    m.setattr(
                        cam, "compute_view_matrix", lambda _pose, q=pose: _reference_view_matrix(q)
                    )
                    ref = cam.render(pose, physics_client=cid)
                diff = np.abs(out.bgr.astype(np.int16) - ref.bgr.astype(np.int16))
                mads.append(float(diff.mean()))
        finally:
            p.disconnect(cid)
        assert max(mads) < PIN_MAX_MAD, f"per-pose BGR MAD {[round(v, 4) for v in mads]}"


# --------------------------------------------------------------------------- #
# Box projection
# --------------------------------------------------------------------------- #
BOX_HALF = [0.3, 0.2, 0.4]
BOX_POS = [3.0, 0.0, 0.4]
BOX_AABB = ((2.7, -0.2, 0.0), (3.3, 0.2, 0.8))

PROJECTION_POSES = [
    CameraPose(x=1.0, y=0.0, z=0.8, yaw=0.0, pitch=-30.0, roll=0.0),
    CameraPose(x=1.5, y=0.5, z=1.0, yaw=340.0, pitch=-35.0, roll=10.0),
    CameraPose(x=2.0, y=-0.6, z=0.6, yaw=20.0, pitch=-10.0, roll=-15.0),
    CameraPose(x=3.0, y=-1.5, z=1.2, yaw=90.0, pitch=-40.0, roll=5.0),
    CameraPose(x=4.5, y=0.2, z=0.9, yaw=180.0, pitch=-25.0, roll=0.0),
    CameraPose(x=3.0, y=0.0, z=1.6, yaw=0.0, pitch=-90.0, roll=30.0),
    # Box face 0.05 m away: part of the box is closer than the near plane.
    CameraPose(x=2.0, y=-0.25, z=0.5, yaw=30.0, pitch=-10.0, roll=0.0),
]


@pytest.fixture(scope="module")
def box_scene():
    cid = p.connect(p.DIRECT)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=BOX_HALF, physicsClientId=cid)
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=BOX_HALF, rgbaColor=[0.7, 0.2, 0.2, 1.0], physicsClientId=cid
    )
    bid = p.createMultiBody(0, col, vis, BOX_POS, physicsClientId=cid)
    yield cid, bid
    p.disconnect(cid)


class TestBoxProjection:
    @pytest.mark.parametrize("pose", PROJECTION_POSES)
    def test_matches_rendered_mask_box(self, box_scene, pose: CameraPose) -> None:
        from sim.labels import decode_segmentation_mask

        cid, bid = box_scene
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
        mask = decode_segmentation_mask(cam.render(pose, physics_client=cid).seg_mask) == bid
        ys, xs = np.nonzero(mask)
        assert len(xs) > 0
        rect = cam.project_aabb(pose, BOX_AABB)
        assert rect is not None
        mask_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        for got, want in zip(rect, mask_box, strict=True):
            assert abs(got - want) <= 2.0, f"projected {rect} vs mask {mask_box}"

    def test_box_behind_camera(self) -> None:
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
        pose = CameraPose(x=5.0, y=0.0, z=0.8, yaw=0.0, pitch=-30.0)
        assert cam.project_aabb(pose, BOX_AABB) is None
        assert cam.projected_box_fraction(pose, BOX_AABB) == 0.0

    def test_eye_inside_box_fills_image(self) -> None:
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
        pose = CameraPose(x=3.0, y=0.0, z=0.5, yaw=0.0, pitch=-30.0)
        assert cam.projected_box_fraction(pose, BOX_AABB) == pytest.approx(1.0)

    def test_fraction_independent_of_resolution(self) -> None:
        pose = PROJECTION_POSES[0]
        small = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
        large = PinholeCamera(arena_config_path=ARENA_CONFIG, width=1280, height=720)
        assert small.projected_box_fraction(pose, BOX_AABB) == pytest.approx(
            large.projected_box_fraction(pose, BOX_AABB)
        )

    def test_project_box_corners_equals_project_aabb(self) -> None:
        from sim.camera import _BOX_CORNER_BITS

        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
        corners = np.where(_BOX_CORNER_BITS, BOX_AABB[1], BOX_AABB[0])
        for pose in PROJECTION_POSES:
            for clip in (True, False):
                assert cam.project_box(pose, corners, clip=clip) == cam.project_aabb(
                    pose, BOX_AABB, clip=clip
                )

    def test_unclipped_extends_beyond_image(self) -> None:
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
        rect = cam.project_aabb(STRADDLE_POSE, BOX_AABB, clip=False)
        clipped = cam.project_aabb(STRADDLE_POSE, BOX_AABB)
        assert rect is not None and clipped is not None
        assert rect[0] < cam.width < rect[2]
        assert clipped[2] == cam.width
        assert rect[:2] == clipped[:2]


# --------------------------------------------------------------------------- #
# Segmentation canvas with margins
# --------------------------------------------------------------------------- #
# The box straddles the right image edge from this pose.
STRADDLE_POSE = CameraPose(x=1.0, y=0.0, z=0.8, yaw=40.0, pitch=-20.0, roll=0.0)


class TestCanvasRender:
    @pytest.mark.parametrize(
        "margins", [(0, 0, 0, 0), (40, 0, 0, 0), (0, 0, 64, 30), (17, 23, 5, 91), (50, 25, 50, 25)]
    )
    @pytest.mark.parametrize("pose", [PROJECTION_POSES[1], STRADDLE_POSE])
    def test_crop_equals_frame_mask(self, box_scene, pose: CameraPose, margins) -> None:
        cid, _ = box_scene
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
        frame = cam.render(pose, physics_client=cid).seg_mask
        canvas = cam.render_segmentation(pose, physics_client=cid, margins=margins)
        left, top, right, bottom = margins
        assert canvas.shape == (180 + top + bottom, 320 + left + right)
        crop = canvas[top : top + 180, left : left + 320]
        assert int((crop != frame).sum()) == 0

    def test_margin_shows_the_part_outside_the_frame(self, box_scene) -> None:
        cid, bid = box_scene
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
        x1, y1, x2, y2 = cam.project_aabb(STRADDLE_POSE, BOX_AABB, clip=False)
        pad = 2  # pixel rounding at the silhouette edge
        margins = (
            max(0, math.ceil(-x1) + pad),
            max(0, math.ceil(-y1) + pad),
            max(0, math.ceil(x2 - cam.width) + pad),
            max(0, math.ceil(y2 - cam.height) + pad),
        )
        assert margins[2] > 0
        canvas = cam.render_segmentation(STRADDLE_POSE, physics_client=cid, margins=margins)
        ys, xs = np.nonzero((canvas & 0xFFFFFF) == bid)
        # The silhouette continues into the right margin and stays inside the canvas.
        assert xs.max() >= margins[0] + cam.width
        assert xs.max() < canvas.shape[1] - 1
        # Its box matches the unclipped projection, shifted by the margins.
        mask_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        shifted = (x1 + margins[0], y1 + margins[1], x2 + margins[0], y2 + margins[1])
        for got, want in zip(shifted, mask_box, strict=True):
            assert abs(got - want) <= 2.0, f"projected {shifted} vs canvas mask {mask_box}"

    def test_zero_margin_matrix_equals_fov_matrix(self) -> None:
        for w, h in ((320, 180), (640, 360), (1280, 720)):
            cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=w, height=h)
            np.testing.assert_allclose(
                cam.canvas_projection_matrix((0, 0, 0, 0)), cam.projection_matrix, atol=1e-6
            )

    def test_negative_margin_rejected(self, box_scene) -> None:
        cid, _ = box_scene
        cam = PinholeCamera(arena_config_path=ARENA_CONFIG, width=320, height=180)
        with pytest.raises(ValueError, match="margins"):
            cam.render_segmentation(STRADDLE_POSE, physics_client=cid, margins=(0, -1, 0, 0))
