"""
sim/camera.py — Pinhole camera simulator for NIDAR synthetic data
================================================================

Simulates a drone-mounted pinhole camera inside PyBullet environments using
physical camera parameters (resolution, FOV, clipping planes, tilt) loaded
from ``configs/arena.yaml``.

Features:
- Configurable pose: ``(x, y, z, yaw, pitch, roll)`` in world coordinates.
- Uses ``p.computeViewMatrix``, ``p.computeProjectionMatrixFOV``, and
  ``p.getCameraImage``.
- Default renderer: CPU TinyRenderer (``p.ER_TINY_RENDERER``).
- Output format:
  - BGR uint8 array of shape ``(H, W, 3)`` ready for OpenCV.
  - Linear depth buffer float32 array of shape ``(H, W)``.
  - Segmentation mask int32 array of shape ``(H, W)`` (flags:
    ``p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX``).
- Built-in FPS benchmarking tool.

Coordinate conventions:
- World axes: x = East/rightward, y = North/forward, z = Upward (ROS 2 REP 103).
- Yaw: Heading in degrees around the z-axis (0 deg = facing +x East; 90 deg = +y North).
- Pitch: Elevation/tilt in degrees. Negative values represent downward tilt
  towards the ground (e.g. -30.0 deg for nominal arena downward search angle).
- Roll: Rotation around the camera optical / forward axis in degrees.

Usage (CLI smoke test & benchmark)::

    python -m sim.camera
    python -m sim.camera --frames 30 --width 640 --height 360

Programmatic::

    from sim.camera import PinholeCamera, CameraPose
    cam = PinholeCamera()
    bgr, depth, seg = cam.render(CameraPose(x=2.25, y=2.25, z=0.8, yaw=90.0, pitch=-30.0))
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import NamedTuple

import numpy as np
import pybullet as p

from cv.config import load_config

DEFAULT_ARENA_CONFIG = Path("configs/arena.yaml")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class CameraPose:
    """Camera mounting pose in world coordinates."""

    x: float
    """World x position in metres (East / rightward)."""

    y: float
    """World y position in metres (North / forward)."""

    z: float
    """World z altitude in metres above ground."""

    yaw: float = 0.0
    """Heading angle in degrees (0 = East, 90 = North)."""

    pitch: float = -30.0
    """Pitch angle in degrees. Negative is tilted downwards (e.g. -30 deg)."""

    roll: float = 0.0
    """Roll / bank angle in degrees."""

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        """Return (x, y, z, yaw, pitch, roll)."""
        return (self.x, self.y, self.z, self.yaw, self.pitch, self.roll)


class RenderOutput(NamedTuple):
    """Output tuple returned by camera render."""

    bgr: np.ndarray
    """OpenCV-compatible BGR image (uint8, shape HxWx3)."""

    depth: np.ndarray
    """Depth buffer values in [0.0, 1.0] (float32, shape HxW)."""

    seg_mask: np.ndarray
    """PyBullet segmentation mask (int32, shape HxW)."""


# --------------------------------------------------------------------------- #
# Pinhole Camera
# --------------------------------------------------------------------------- #
class PinholeCamera:
    """
    Simulated pinhole camera for rendering PyBullet scenes.

    Parameters
    ----------
    arena_config_path : str or Path
        Path to arena YAML configuration.
    width : int, optional
        Override capture width in pixels.
    height : int, optional
        Override capture height in pixels.
    vertical_fov_deg : float, optional
        Override vertical field of view in degrees.
    near_clip : float, optional
        Override near clipping plane distance in metres.
    far_clip : float, optional
        Override far clipping plane distance in metres.
    renderer : int
        PyBullet renderer flag (default: ``p.ER_TINY_RENDERER``).
    """

    def __init__(
        self,
        arena_config_path: str | Path = DEFAULT_ARENA_CONFIG,
        width: int | None = None,
        height: int | None = None,
        vertical_fov_deg: float | None = None,
        near_clip: float | None = None,
        far_clip: float | None = None,
        renderer: int = p.ER_TINY_RENDERER,
    ) -> None:
        cfg = load_config(arena_config_path)
        cam_cfg = cfg.camera

        self.width: int = width if width is not None else int(cam_cfg.resolution.width)
        self.height: int = height if height is not None else int(cam_cfg.resolution.height)
        self.vertical_fov_deg: float = (
            vertical_fov_deg
            if vertical_fov_deg is not None
            else float(cam_cfg.fov.vertical_deg)
        )
        self.near_clip: float = (
            near_clip if near_clip is not None else float(cam_cfg.near_clip)
        )
        self.far_clip: float = (
            far_clip if far_clip is not None else float(cam_cfg.far_clip)
        )
        self.default_tilt_deg: float = float(cam_cfg.tilt_deg)
        self.default_height: float = float(cam_cfg.height)
        self.renderer: int = renderer

        # Lighting parameters loaded from configs/arena.yaml
        light_cfg = getattr(cfg, "lighting", None)
        if light_cfg is not None:
            self.ambient_coeff: float = float(np.mean(light_cfg.ambient_rgb))
            self.directional_color: list[float] = [float(c) for c in light_cfg.directional_color_rgb]
            self.directional_direction: list[float] = [float(d) for d in light_cfg.directional_direction]
            self.cast_shadows: bool = bool(light_cfg.cast_shadows)
        else:
            self.ambient_coeff = 0.35
            self.directional_color = [0.90, 0.90, 0.85]
            self.directional_direction = [0.1, 0.2, -1.0]
            self.cast_shadows = True

        # Precompute projection matrix
        self.aspect_ratio: float = self.width / float(self.height)
        self.projection_matrix: list[float] = list(
            p.computeProjectionMatrixFOV(
                fov=self.vertical_fov_deg,
                aspect=self.aspect_ratio,
                nearVal=self.near_clip,
                farVal=self.far_clip,
            )
        )

    # -- Pose & Matrix Computation --

    def compute_view_matrix(
        self,
        pose: CameraPose | tuple[float, float, float, float, float, float],
    ) -> list[float]:
        """
        Compute PyBullet 4x4 view matrix from camera pose.

        Parameters
        ----------
        pose : CameraPose or 6-tuple
            (x, y, z, yaw, pitch, roll) where angles are in degrees.
        """
        if isinstance(pose, tuple):
            x, y, z, yaw, pitch, roll = pose
        else:
            x, y, z, yaw, pitch, roll = pose.as_tuple()

        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)

        # Forward unit vector in world space:
        # yaw=0 -> +x, yaw=90 -> +y.
        # pitch < 0 tilts downward towards ground (-z).
        cp = math.cos(pitch_rad)
        sp = math.sin(pitch_rad)
        cy = math.cos(yaw_rad)
        sy = math.sin(yaw_rad)

        fwd = np.array([cp * cy, cp * sy, sp], dtype=np.float64)
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm > 1e-8:
            fwd /= fwd_norm

        # Nominal up vector before roll
        # If camera points horizontally (pitch=0), up is [0, 0, 1].
        # If camera pitches down (pitch=-theta), up tilts forward [sp*cy, sp*sy, -cp].
        up_nominal = np.array([-sp * cy, -sp * sy, cp], dtype=np.float64)
        up_norm = np.linalg.norm(up_nominal)
        if up_norm > 1e-8:
            up_nominal /= up_norm

        # Camera right vector = fwd x up
        right = np.cross(fwd, up_nominal)
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-8:
            right /= right_norm

        # Apply roll rotation around optical axis (fwd)
        cr = math.cos(roll_rad)
        sr = math.sin(roll_rad)
        up = cr * up_nominal + sr * right

        eye = [float(x), float(y), float(z)]
        target = [float(x + fwd[0]), float(y + fwd[1]), float(z + fwd[2])]
        up_list = [float(up[0]), float(up[1]), float(up[2])]

        view_mat = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=up_list,
        )
        return list(view_mat)

    # -- Rendering --

    def render(
        self,
        pose: CameraPose | tuple[float, float, float, float, float, float],
        physics_client: int | None = None,
        light_direction: list[float] | None = None,
        light_color: list[float] | None = None,
        light_distance: float | None = None,
        ambient_coeff: float | None = None,
        diffuse_coeff: float | None = None,
        shadow: int | bool | None = None,
    ) -> RenderOutput:
        """
        Capture a frame at the specified camera pose.

        Parameters
        ----------
        pose : CameraPose or 6-tuple
            (x, y, z, yaw, pitch, roll).
        physics_client : int, optional
            Physics client ID. If None, default active PyBullet client is used.
        light_direction : list[float], optional
            Override directional light vector [x, y, z].
        light_color : list[float], optional
            Override light color [r, g, b].
        light_distance : float, optional
            Override light distance from scene center.
        ambient_coeff : float, optional
            Override ambient light intensity.
        diffuse_coeff : float, optional
            Override diffuse light intensity.
        shadow : int or bool, optional
            Override shadow casting flag (0/False or 1/True).

        Returns
        -------
        RenderOutput
            NamedTuple with:
            - bgr: np.ndarray (H, W, 3) uint8 (OpenCV BGR)
            - depth: np.ndarray (H, W) float32
            - seg_mask: np.ndarray (H, W) int32
        """
        view_mat = self.compute_view_matrix(pose)

        use_shadow = (
            int(bool(shadow))
            if shadow is not None
            else (1 if self.cast_shadows else 0)
        )

        kwargs = {
            "width": self.width,
            "height": self.height,
            "viewMatrix": view_mat,
            "projectionMatrix": self.projection_matrix,
            "renderer": self.renderer,
            "flags": p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            "lightDirection": light_direction if light_direction is not None else self.directional_direction,
            "lightColor": light_color if light_color is not None else self.directional_color,
            "lightAmbientCoeff": ambient_coeff if ambient_coeff is not None else self.ambient_coeff,
            "lightDiffuseCoeff": diffuse_coeff if diffuse_coeff is not None else 0.65,
            "shadow": use_shadow,
        }
        if light_distance is not None:
            kwargs["lightDistance"] = float(light_distance)
        if physics_client is not None:
            kwargs["physicsClientId"] = physics_client

        w, h, rgba, depth_buf, seg_buf = p.getCameraImage(**kwargs)

        # Convert RGBA -> BGR uint8
        rgba_arr = np.reshape(rgba, (h, w, 4)).astype(np.uint8)
        bgr_arr = rgba_arr[:, :, [2, 1, 0]]  # RGBA to BGR

        # Depth buffer
        depth_arr = np.reshape(depth_buf, (h, w)).astype(np.float32)

        # Segmentation mask
        seg_arr = np.reshape(seg_buf, (h, w)).astype(np.int32)

        return RenderOutput(bgr=bgr_arr, depth=depth_arr, seg_mask=seg_arr)

    # -- Benchmarking --

    def benchmark(
        self,
        num_frames: int = 50,
        pose: CameraPose | None = None,
        physics_client: int | None = None,
    ) -> float:
        """
        Benchmark camera rendering and print frames per second (FPS).

        Parameters
        ----------
        num_frames : int
            Number of iterations to render.
        pose : CameraPose, optional
            Test pose. Defaults to nominal arena cruise pose.
        physics_client : int, optional
            PyBullet client ID.

        Returns
        -------
        float
            Rendered frames per second (FPS).
        """
        if pose is None:
            pose = CameraPose(
                x=1.5,
                y=1.5,
                z=self.default_height,
                yaw=0.0,
                pitch=-self.default_tilt_deg,
                roll=0.0,
            )

        # Warm up
        self.render(pose, physics_client=physics_client)

        t_start = time.perf_counter()
        for _ in range(num_frames):
            self.render(pose, physics_client=physics_client)
        t_elapsed = time.perf_counter() - t_start

        fps = num_frames / max(t_elapsed, 1e-6)
        ms_per_frame = (t_elapsed / num_frames) * 1000.0

        print(f"  Resolution  : {self.width} × {self.height}")
        print(f"  Renderer    : {'CPU TinyRenderer' if self.renderer == p.ER_TINY_RENDERER else 'OpenGL'}")
        print(f"  Frames      : {num_frames}")
        print(f"  Total time  : {t_elapsed:.3f} s")
        print(f"  Throughput  : {fps:.2f} FPS ({ms_per_frame:.2f} ms/frame)")

        return fps


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="NIDAR Camera Benchmark & Test")
    parser.add_argument("--frames", type=int, default=30, help="Frames to benchmark")
    parser.add_argument("--width", type=int, default=None, help="Image width (default: from arena.yaml)")
    parser.add_argument("--height", type=int, default=None, help="Image height (default: from arena.yaml)")
    parser.add_argument(
        "--arena",
        type=str,
        default=str(DEFAULT_ARENA_CONFIG),
        help="Path to arena.yaml",
    )
    args = parser.parse_args()

    cid = p.connect(p.DIRECT)
    try:
        # Create a simple floor for realistic rendering
        floor_col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid)
        p.createMultiBody(0, floor_col, physicsClientId=cid)

        camera = PinholeCamera(
            arena_config_path=args.arena,
            width=args.width,
            height=args.height,
        )

        print("=" * 60)
        print("  NIDAR Pinhole Camera Simulator Benchmark")
        print("=" * 60)
        camera.benchmark(num_frames=args.frames, physics_client=cid)
        print("=" * 60)

        # Test single render output properties
        test_pose = CameraPose(x=2.0, y=2.0, z=0.8, yaw=45.0, pitch=-30.0, roll=0.0)
        out = camera.render(test_pose, physics_client=cid)
        print(f"  Render verification:")
        print(f"    BGR shape     : {out.bgr.shape}, dtype={out.bgr.dtype}")
        print(f"    Depth shape   : {out.depth.shape}, dtype={out.depth.dtype}")
        print(f"    Seg mask shape: {out.seg_mask.shape}, dtype={out.seg_mask.dtype}")
        print("Camera verification passed.")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
