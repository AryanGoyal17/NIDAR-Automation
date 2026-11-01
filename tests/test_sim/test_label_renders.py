"""
tests/test_sim/test_label_renders.py — Labels measured on small real renders
============================================================================

Validates, with PyBullet renders at 320x180:
1. SilhouetteRenderer copies sphere, box, capsule and cylinder bodies exactly
   (0 px mask difference, same collision AABB) and rejects other shapes.
2. A plate half outside the frame has the geometrically expected truncation.
3. A fully visible survivor has truncation 0 and occlusion near 0; one behind
   a wall is occluded; one behind another survivor is measured alone.

The slow scene checks at the end run only with NIDAR_SCENE_TESTS=1.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

p = pytest.importorskip("pybullet")

from sim.camera import CameraPose, PinholeCamera  # noqa: E402
from sim.labels import (  # noqa: E402
    SilhouetteRenderer,
    UnsupportedShapeError,
    compute_survivor_labels,
    decode_segmentation_mask,
)
from sim.survivor import _POSTURE_BUILDERS  # noqa: E402

ARENA_CONFIG = Path("configs/arena.yaml")
W, H = 320, 180
CANVAS_CAP = 4.0  # explicit, so these tests do not depend on arena.yaml edits
SKIN = (0.80, 0.62, 0.50)
CLOTH = (0.70, 0.15, 0.15)


@pytest.fixture
def cid():
    client = p.connect(p.DIRECT)
    yield client
    p.disconnect(client)


@pytest.fixture
def cam() -> PinholeCamera:
    return PinholeCamera(arena_config_path=ARENA_CONFIG, width=W, height=H)


def _box(cid: int, half: list[float], pos: list[float], rgb=(0.6, 0.6, 0.6)) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half, rgbaColor=[*rgb, 1.0], physicsClientId=cid
    )
    return p.createMultiBody(0, col, vis, pos, physicsClientId=cid)


def _floor(cid: int) -> int:
    return _box(cid, [10.0, 10.0, 0.005], [0.0, 0.0, -0.005])


def _standing(cid: int, x: float, y: float, yaw_deg: float = 180.0) -> list[int]:
    """Standing survivor facing the camera at the origin (yaw 180 = facing -x)."""
    return _POSTURE_BUILDERS["standing"](
        cid, x, y, math.radians(yaw_deg), 1.75, 0.45, 0.25, SKIN, CLOTH
    )


def _labels(cam, cid, pose, survivors):
    with SilhouetteRenderer(cam, survivors, cid, max_canvas_area_factor=CANVAS_CAP) as sr:
        sils = sr.render(pose)
    seg = cam.render(pose, physics_client=cid).seg_mask
    return compute_survivor_labels(seg, survivors, silhouettes=sils), sils, seg


# --------------------------------------------------------------------------- #
# Generic copy
# --------------------------------------------------------------------------- #
SHAPES = {
    "sphere": ({"radius": 0.2}, {"radius": 0.2}),
    "box": ({"halfExtents": [0.1, 0.2, 0.3]}, {"halfExtents": [0.1, 0.2, 0.3]}),
    "capsule": ({"radius": 0.1, "length": 0.5}, {"radius": 0.1, "height": 0.5}),
    "cylinder": ({"radius": 0.15, "length": 0.4}, {"radius": 0.15, "height": 0.4}),
}
SHAPE_TYPES = {
    "sphere": p.GEOM_SPHERE,
    "box": p.GEOM_BOX,
    "capsule": p.GEOM_CAPSULE,
    "cylinder": p.GEOM_CYLINDER,
}
COPY_POSES = [
    CameraPose(x=0.0, y=0.0, z=1.0, yaw=0.0, pitch=-15.0),
    CameraPose(x=1.6, y=-1.0, z=1.6, yaw=90.0, pitch=-45.0, roll=10.0),
    CameraPose(x=1.5, y=1.2, z=0.4, yaw=-60.0, pitch=5.0),
]


def _synthetic_body(cid: int, shape: str) -> int:
    """A body of one primitive with a visual frame offset and an inertial offset."""
    vis_kw, col_kw = SHAPES[shape]
    gtype = SHAPE_TYPES[shape]
    frame_orn = p.getQuaternionFromEuler([0.4, -0.2, 0.7])
    col = p.createCollisionShape(
        gtype,
        collisionFramePosition=[0.05, -0.02, 0.03],
        collisionFrameOrientation=frame_orn,
        physicsClientId=cid,
        **col_kw,
    )
    vis = p.createVisualShape(
        gtype,
        rgbaColor=[0.2, 0.5, 0.8, 1.0],
        visualFramePosition=[0.05, -0.02, 0.03],
        visualFrameOrientation=frame_orn,
        physicsClientId=cid,
        **vis_kw,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[1.5, 0.1, 0.6],
        baseOrientation=p.getQuaternionFromEuler([0.3, 0.2, 0.1]),
        baseInertialFramePosition=[0.0, 0.03, -0.01],
        physicsClientId=cid,
    )


class TestShapeCopy:
    @pytest.mark.parametrize("shape", list(SHAPES))
    def test_copy_renders_identically(self, cid, cam, shape: str) -> None:
        bid = _synthetic_body(cid, shape)
        with SilhouetteRenderer(cam, [[bid]], cid, max_canvas_area_factor=CANVAS_CAP) as sr:
            copy = sr.copy_ids[0][0]
            for pose in COPY_POSES:
                orig = cam.render_segmentation(pose, physics_client=cid)
                dup = cam.render_segmentation(pose, physics_client=sr.copy_client)
                a = decode_segmentation_mask(orig) == bid
                b = decode_segmentation_mask(dup) == copy
                assert a.sum() > 50, f"{shape} not in view from {pose}"
                assert int((a != b).sum()) == 0
            aabb_orig = np.array(p.getAABB(bid, physicsClientId=cid))
            aabb_copy = np.array(p.getAABB(copy, physicsClientId=sr.copy_client))
            np.testing.assert_allclose(aabb_copy, aabb_orig, atol=1e-6)

    @pytest.mark.parametrize("shape", list(SHAPES))
    def test_part_box_bounds_silhouette(self, cid, cam, shape: str) -> None:
        bid = _synthetic_body(cid, shape)
        with SilhouetteRenderer(cam, [[bid]], cid, max_canvas_area_factor=CANVAS_CAP) as sr:
            for pose in COPY_POSES:
                sil = sr.render(pose)[0]
                assert sil is not None and sil.exact
                visible = decode_segmentation_mask(cam.render(pose, physics_client=cid).seg_mask)
                visible = visible == bid
                # Alone render == real render here (nothing else in the scene).
                assert int((visible != sil.in_frame).sum()) == 0

    def test_mesh_rejected_with_type_name(self, cid, cam) -> None:
        pybullet_data = pytest.importorskip("pybullet_data")
        mesh = os.path.join(pybullet_data.getDataPath(), "cube.obj")
        vis = p.createVisualShape(p.GEOM_MESH, fileName=mesh, physicsClientId=cid)
        bid = p.createMultiBody(0, -1, vis, [1.5, 0.0, 0.5], physicsClientId=cid)
        with pytest.raises(UnsupportedShapeError, match="GEOM_MESH"):
            SilhouetteRenderer(cam, [[bid]], cid, max_canvas_area_factor=CANVAS_CAP)

    def test_visual_only_body_copied(self, cid, cam) -> None:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, physicsClientId=cid)
        bid = p.createMultiBody(0, -1, vis, [1.5, 0.0, 0.6], physicsClientId=cid)
        with SilhouetteRenderer(cam, [[bid]], cid, max_canvas_area_factor=CANVAS_CAP) as sr:
            sil = sr.render(COPY_POSES[0])[0]
            assert sil is not None and sil.in_frame_pixels > 50


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #
class TestTruncation:
    def test_plate_half_outside_matches_geometry(self, cid, cam) -> None:
        # A thin plate facing a level camera, centred on the camera's height and
        # cut by the left image edge so that a share `target` of it is outside.
        pose = CameraPose(x=0.0, y=0.0, z=1.0, yaw=0.0, pitch=0.0)
        half, half_t, depth, target = 0.5, 0.001, 2.0, 0.4
        f, d_front = cam.focal_length_px, depth - half_t
        width_px = f * 2 * half / d_front
        # Image column u = W/2 - f * y / d (world +y is image left).
        y_left = (W / 2 + target * width_px) * d_front / f
        bid = _box(cid, [half_t, half, half], [depth, y_left - half, 1.0])

        u_left = W / 2 - f * y_left / d_front
        u_right = W / 2 - f * (y_left - 2 * half) / d_front
        expected = (0.0 - u_left) / (u_right - u_left)
        # Rasterisation includes a pixel by its centre, so each count can be off
        # by half a pixel per unit of edge length: the error is at most
        # perimeter / area. The plate's side face adds under one pixel column.
        area = width_px * width_px
        tolerance = (4 * width_px + width_px) / area

        labels, sils, _ = _labels(cam, cid, pose, [[bid]])
        assert sils[0] is not None and sils[0].exact
        assert len(labels) == 1
        assert abs(labels[0].truncation_fraction - expected) <= tolerance, (
            f"truncation {labels[0].truncation_fraction:.4f}, expected {expected:.4f} "
            f"+/- {tolerance:.4f}"
        )
        assert labels[0].occlusion_fraction == 0.0

    def test_standing_survivor_on_frame_edge(self, cid, cam) -> None:
        _floor(cid)
        pose = CameraPose(x=0.0, y=0.0, z=1.0, yaw=0.0, pitch=-10.0)
        # Body axis on the left image edge: about half outside. Loose bounds,
        # because the posture is not symmetric under perspective.
        depth = 3.0
        y_edge = depth * (W / 2) / cam.focal_length_px
        ids = _standing(cid, depth, y_edge)
        labels, sils, _ = _labels(cam, cid, pose, [ids])
        assert sils[0] is not None and sils[0].exact
        assert len(labels) == 1
        assert 0.25 < labels[0].truncation_fraction < 0.75
        assert labels[0].truncation_exact


# --------------------------------------------------------------------------- #
# Occlusion
# --------------------------------------------------------------------------- #
VIEW = CameraPose(x=0.0, y=0.0, z=1.0, yaw=0.0, pitch=-10.0)


class TestOcclusion:
    def test_fully_visible_survivor(self, cid, cam) -> None:
        _floor(cid)
        ids = _standing(cid, 3.0, 0.0)
        labels, sils, _ = _labels(cam, cid, VIEW, [ids])
        assert len(labels) == 1
        assert sils[0].outside_pixels == 0
        assert labels[0].truncation_fraction == 0.0
        assert labels[0].occlusion_fraction <= 0.01

    def test_survivor_behind_wall(self, cid, cam) -> None:
        _floor(cid)
        ids = _standing(cid, 3.0, 0.0)
        _box(cid, [0.05, 0.5, 1.0], [2.0, -0.5, 1.0])  # covers the right half
        labels, sils, seg = _labels(cam, cid, VIEW, [ids])
        assert sils[0] is not None
        assert len(labels) == 1
        assert 0.2 < labels[0].occlusion_fraction < 0.8
        assert labels[0].truncation_fraction == 0.0

    def test_survivor_almost_hidden_is_dropped(self, cid, cam) -> None:
        _floor(cid)
        ids = _standing(cid, 3.0, 0.0)
        _box(cid, [0.05, 1.0, 1.5], [2.0, 0.0, 1.0])  # covers all of it
        _box(cid, [0.05, 1.0, 0.2], [2.0, 0.0, 0.05])
        seg = cam.render(VIEW, physics_client=cid).seg_mask
        visible = np.isin(decode_segmentation_mask(seg), ids).sum()
        labels, sils, _ = _labels(cam, cid, VIEW, [ids])
        assert sils[0] is not None and sils[0].in_frame_pixels > 0
        occlusion = 1.0 - visible / sils[0].in_frame_pixels
        assert occlusion > 0.9
        assert labels == []

    def test_survivor_behind_survivor_measured_alone(self, cid, cam) -> None:
        _floor(cid)
        front = _standing(cid, 2.5, 0.05)
        rear = _standing(cid, 4.0, -0.1)  # its left part is behind the front one
        labels, sils, _ = _labels(cam, cid, VIEW, [front, rear])
        by_id = {lb.survivor_id: lb for lb in labels}
        assert by_id[0].occlusion_fraction <= 0.01
        assert by_id[1].occlusion_fraction > 0.1
        # The rear survivor's silhouette equals the one rendered with it alone.
        with SilhouetteRenderer(cam, [rear], cid, max_canvas_area_factor=CANVAS_CAP) as sr:
            alone = sr.render(VIEW)[0]
        assert int((alone.in_frame != sils[1].in_frame).sum()) == 0


# --------------------------------------------------------------------------- #
# Visibility (thresholds from arena.yaml)
# --------------------------------------------------------------------------- #
NEAR_BLACK = {
    "ambient_coeff": 0.03,
    "diffuse_coeff": 0.03,
    "specular_coeff": 0.0,
    "light_color": [0.3, 0.3, 0.3],
}


def _visibility_scene(cid: int) -> list[int]:
    """Closed room (empty background renders white): floor, ceiling, wall behind."""
    _floor(cid)
    _box(cid, [10.0, 10.0, 0.005], [0.0, 0.0, 3.0])  # ceiling
    _box(cid, [0.05, 10.0, 1.6], [4.5, 0.0, 1.5], rgb=(0.7, 0.7, 0.68))  # wall behind
    return _standing(cid, 3.0, 0.0)


class TestVisibility:
    def _label(self, cam, cid, ids, light: dict, low_light: bool):
        pytest.importorskip("cv2")
        pytest.importorskip("yaml")
        from sim.labels import LabelConfig, label_frame
        from sim.randomize import FrameAugmentor

        cfg = LabelConfig.from_arena_config(ARENA_CONFIG)
        out = cam.render(VIEW, physics_client=cid, **light)
        img = FrameAugmentor().augment(out.bgr, is_low_light=low_light, seed=0)
        with SilhouetteRenderer(cam, [ids], cid, max_canvas_area_factor=CANVAS_CAP) as sr:
            result = label_frame(out.seg_mask, [ids], sr.render(VIEW), img, cfg)
        return result, img, cfg

    def test_near_black_low_light_dropped_and_ambiguous(self, cid, cam) -> None:
        ids = _visibility_scene(cid)
        result, img, cfg = self._label(cam, cid, ids, NEAR_BLACK, low_light=True)
        assert img.mean() < 20, f"scene not near-black: mean grey {img.mean():.1f}"
        assert result.labels == []
        assert len(result.dropped) == 1
        assert result.dropped[0].reasons == ("low_contrast",)
        assert result.dropped[0].label.contrast < cfg.min_contrast
        assert result.ambiguous
        assert result.ambiguous_reasons == ("low_contrast",)

    def test_same_scene_in_normal_light_kept(self, cid, cam) -> None:
        ids = _visibility_scene(cid)
        result, _, cfg = self._label(cam, cid, ids, {}, low_light=False)
        assert not result.ambiguous
        assert len(result.labels) == 1
        assert result.labels[0].contrast >= cfg.min_contrast
