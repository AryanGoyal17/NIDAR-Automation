"""
sim/randomize.py — Simulation domain randomization and post-processing augmentor
================================================================================

Implements seeded domain randomization for synthetic visual data generation:
1. Per-sequence environmental randomization:
   - Procedural textures for walls and floors (solid, noise, checker, brick-like).
   - Directional light orientation (azimuth & elevation) and RGB color.
   - Ambient, diffuse, and specular reflection coefficients.
   - Dynamic cast shadow toggling.
   - Texture loading and application into active PyBullet simulation bodies.
2. Per-frame post-processing augmentations (applied after rendering):
   - Brightness shifts.
   - Contrast scaling.
   - Gamma power-law correction.
   - Sensor Gaussian noise.
   - Camera/drone directional motion blur.
   - Lossy JPEG compression artifacts.
   - Lens optical vignette.
   - Optional degraded low-light / night mission mode.

All parameter ranges, distributions, and probabilities are strictly governed
by ``configs/randomization.yaml``. Every operation is 100% deterministic when
provided with a random seed.

Usage (CLI test & preview generation)::

    python -m sim.randomize --seed 42 --output-dir outputs/randomize_preview

Programmatic::

    from sim.randomize import SequenceRandomizer, FrameAugmentor

    # 1. Per-sequence randomization
    seq_rand = SequenceRandomizer()
    seq_params = seq_rand.sample_sequence_params(seed=123)
    seq_rand.apply_to_sim(maze_res, seq_params, physics_client=cid)

    # 2. Per-frame post-processing
    augmentor = FrameAugmentor()
    aug_bgr = augmentor.augment(raw_bgr, is_low_light=seq_params.is_low_light, seed=456)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
import tempfile
from typing import Any

import cv2
import numpy as np
import pybullet as p
import yaml

from sim.maze import MazeResult

DEFAULT_CONFIG_PATH = Path("configs/randomization.yaml")


# --------------------------------------------------------------------------- #
# Configuration Loader
# --------------------------------------------------------------------------- #
def load_randomization_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and parse domain randomization configuration YAML."""
    p_path = Path(path)
    if not p_path.exists():
        raise FileNotFoundError(f"Randomization config not found: {p_path.resolve()}")
    with open(p_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config structure in {p_path}")
    return cfg


# --------------------------------------------------------------------------- #
# Sequence Parameters Dataclass
# --------------------------------------------------------------------------- #
@dataclass
class SequenceParams:
    """Holds sampled environmental and lighting parameters for a video sequence."""

    seed: int | None
    wall_texture_type: str
    floor_texture_type: str
    wall_texture: np.ndarray
    """BGR uint8 image of shape (H, W, 3)."""
    floor_texture: np.ndarray
    """BGR uint8 image of shape (H, W, 3)."""
    light_direction: list[float]
    """Normalized Cartesian light vector [dx, dy, dz]."""
    light_color: list[float]
    """Light RGB intensities [r, g, b]."""
    light_distance: float
    ambient_coeff: float
    diffuse_coeff: float
    specular_coeff: float
    shadow: bool
    is_low_light: bool


# --------------------------------------------------------------------------- #
# Procedural Texture Generator
# --------------------------------------------------------------------------- #
class ProceduralTextureGenerator:
    """
    Procedurally synthesizes 2D surface textures (solid, noise, checker, brick)
    for maze wall and floor surfaces.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_randomization_config()
        self.tex_cfg = config.get("sequence", {}).get("textures", {})
        res = self.tex_cfg.get("resolution", [256, 256])
        self.tex_height = int(res[0])
        self.tex_width = int(res[1])

    # -- Primitive Texture Synthesizers --

    @staticmethod
    def generate_solid(height: int, width: int, color_bgr: tuple[int, int, int]) -> np.ndarray:
        """Create a uniform solid color texture."""
        return np.full((height, width, 3), color_bgr, dtype=np.uint8)

    @staticmethod
    def generate_noise(
        height: int,
        width: int,
        base_bgr: tuple[int, int, int],
        std: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Create a textured surface with zero-mean Gaussian noise."""
        base = np.full((height, width, 3), base_bgr, dtype=np.float32)
        noise = rng.normal(0.0, std, size=(height, width, 3))
        return np.clip(base + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def generate_checker(
        height: int,
        width: int,
        grid_size: int,
        color1_bgr: tuple[int, int, int],
        color2_bgr: tuple[int, int, int],
    ) -> np.ndarray:
        """Create a 2D alternating checkerboard pattern."""
        grid_size = max(2, int(grid_size))
        tex = np.zeros((height, width, 3), dtype=np.uint8)
        cell_h = height / grid_size
        cell_w = width / grid_size

        for r in range(grid_size):
            y0 = int(round(r * cell_h))
            y1 = int(round((r + 1) * cell_h))
            for c in range(grid_size):
                x0 = int(round(c * cell_w))
                x1 = int(round((c + 1) * cell_w))
                col = color1_bgr if ((r + c) % 2 == 0) else color2_bgr
                tex[y0:y1, x0:x1] = col

        return tex

    @staticmethod
    def generate_brick(
        height: int,
        width: int,
        rows: int,
        cols: int,
        mortar_px: int,
        brick_bgr: tuple[int, int, int],
        mortar_bgr: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Create a staggered running-bond brick wall texture."""
        tex = np.full((height, width, 3), mortar_bgr, dtype=np.uint8)
        row_h = height / float(rows)
        col_w = width / float(cols)

        for r in range(rows):
            y0 = int(round(r * row_h + mortar_px / 2.0))
            y1 = int(round((r + 1) * row_h - mortar_px / 2.0))
            if y1 <= y0:
                continue

            # Stagger alternate rows by half brick width
            offset = (col_w / 2.0) if (r % 2 == 1) else 0.0

            for c in range(-1, cols + 1):
                x0 = int(round(c * col_w + offset + mortar_px / 2.0))
                x1 = int(round((c + 1) * col_w + offset - mortar_px / 2.0))
                x0_c = max(0, min(width, x0))
                x1_c = max(0, min(width, x1))

                if x1_c > x0_c:
                    if rng is not None:
                        # Slight per-brick color variation
                        jitter = rng.uniform(-10.0, 10.0)
                        col = [int(np.clip(brick_bgr[i] + jitter, 0, 255)) for i in range(3)]
                    else:
                        col = brick_bgr
                    tex[y0:y1, x0_c:x1_c] = col

        return tex

    # -- Random Texture Sampler --

    def sample_surface_texture(
        self,
        surface_type: str,  # "wall" or "floor"
        forced_tex_type: str | None = None,
        rng_py: random.Random | None = None,
        rng_np: np.random.Generator | None = None,
    ) -> tuple[str, np.ndarray]:
        """
        Sample a procedural texture based on surface type and YAML ranges.

        Returns
        -------
        tuple[str, np.ndarray]
            (texture_type_name, bgr_image)
        """
        if rng_py is None:
            rng_py = random.Random()
        if rng_np is None:
            rng_np = np.random.default_rng()

        enabled_types = self.tex_cfg.get(
            "enabled_types", ["solid", "noise", "checker", "brick"]
        )
        chosen_type = forced_tex_type if forced_tex_type else rng_py.choice(enabled_types)

        h, w = self.tex_height, self.tex_width

        def _rgb_to_bgr(rgb_floats: list[float]) -> tuple[int, int, int]:
            r = int(np.clip(rgb_floats[0] * 255.0, 0, 255))
            g = int(np.clip(rgb_floats[1] * 255.0, 0, 255))
            b = int(np.clip(rgb_floats[2] * 255.0, 0, 255))
            return (b, g, r)

        def _sample_color(c_min: list[float], c_max: list[float]) -> tuple[int, int, int]:
            rgb = [rng_py.uniform(c_min[i], c_max[i]) for i in range(3)]
            return _rgb_to_bgr(rgb)

        if chosen_type == "solid":
            cfg_s = self.tex_cfg.get("solid", {})
            c_min = cfg_s.get(f"{surface_type}_color_min", [0.4, 0.4, 0.4])
            c_max = cfg_s.get(f"{surface_type}_color_max", [0.8, 0.8, 0.8])
            bgr = _sample_color(c_min, c_max)
            img = self.generate_solid(h, w, bgr)

        elif chosen_type == "noise":
            cfg_n = self.tex_cfg.get("noise", {})
            c_min = cfg_n.get(f"{surface_type}_base_min", [0.4, 0.4, 0.4])
            c_max = cfg_n.get(f"{surface_type}_base_max", [0.8, 0.8, 0.8])
            std_range = cfg_n.get("noise_std_range", [10.0, 30.0])
            std = rng_py.uniform(std_range[0], std_range[1])
            base_bgr = _sample_color(c_min, c_max)
            img = self.generate_noise(h, w, base_bgr, std, rng_np)

        elif chosen_type == "checker":
            cfg_c = self.tex_cfg.get("checker", {})
            g_range = cfg_c.get("grid_size_range", [4, 16])
            grid_size = rng_py.randint(g_range[0], g_range[1])
            c1_min = cfg_c.get("color1_min", [0.3, 0.3, 0.3])
            c1_max = cfg_c.get("color1_max", [0.5, 0.5, 0.5])
            c2_min = cfg_c.get("color2_min", [0.6, 0.6, 0.6])
            c2_max = cfg_c.get("color2_max", [0.9, 0.9, 0.9])
            col1 = _sample_color(c1_min, c1_max)
            col2 = _sample_color(c2_min, c2_max)
            img = self.generate_checker(h, w, grid_size, col1, col2)

        elif chosen_type == "brick":
            cfg_b = self.tex_cfg.get("brick", {})
            r_range = cfg_b.get("brick_rows_range", [8, 16])
            c_range = cfg_b.get("brick_cols_range", [4, 8])
            m_range = cfg_b.get("mortar_thickness_px_range", [2, 5])
            rows = rng_py.randint(r_range[0], r_range[1])
            cols = rng_py.randint(c_range[0], c_range[1])
            mortar = rng_py.randint(m_range[0], m_range[1])
            b_min = cfg_b.get("brick_color_min", [0.5, 0.3, 0.25])
            b_max = cfg_b.get("brick_color_max", [0.8, 0.5, 0.4])
            m_min = cfg_b.get("mortar_color_min", [0.65, 0.65, 0.65])
            m_max = cfg_b.get("mortar_color_max", [0.85, 0.85, 0.85])
            brick_col = _sample_color(b_min, b_max)
            mortar_col = _sample_color(m_min, m_max)
            img = self.generate_brick(h, w, rows, cols, mortar, brick_col, mortar_col, rng_np)

        else:
            # Fallback to grey solid
            img = self.generate_solid(h, w, (128, 128, 128))
            chosen_type = "solid"

        return chosen_type, img


# --------------------------------------------------------------------------- #
# Sequence Randomizer
# --------------------------------------------------------------------------- #
class SequenceRandomizer:
    """
    Samples environmental parameters (textures, lighting, shadows, low-light mode)
    for a sequence and applies them to an active PyBullet simulation.
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.config = load_randomization_config(self.config_path)
        self.tex_gen = ProceduralTextureGenerator(self.config)
        self.seq_cfg = self.config.get("sequence", {})
        self.light_cfg = self.seq_cfg.get("lighting", {})
        self.low_light_cfg = self.config.get("frame", {}).get("low_light", {})

    def sample_sequence_params(self, seed: int | None = None) -> SequenceParams:
        """
        Sample a complete set of environmental and lighting parameters.

        Parameters
        ----------
        seed : int, optional
            Seed for deterministic sampling.
        """
        rng_py = random.Random(seed)
        rng_np = np.random.default_rng(seed)

        # 1. Sample procedural textures
        w_type, w_img = self.tex_gen.sample_surface_texture("wall", rng_py=rng_py, rng_np=rng_np)
        f_type, f_img = self.tex_gen.sample_surface_texture("floor", rng_py=rng_py, rng_np=rng_np)

        # 2. Lighting direction (Azimuth & Elevation)
        az_range = self.light_cfg.get("azimuth_deg_range", [0.0, 360.0])
        el_range = self.light_cfg.get("elevation_deg_range", [-75.0, -25.0])
        az_deg = rng_py.uniform(az_range[0], az_range[1])
        el_deg = rng_py.uniform(el_range[0], el_range[1])

        az_rad = math.radians(az_deg)
        el_rad = math.radians(el_deg)
        # Cartesian light direction pointing from light source towards arena center:
        # Elevation is negative (pointing down)
        dx = math.cos(el_rad) * math.cos(az_rad)
        dy = math.cos(el_rad) * math.sin(az_rad)
        dz = math.sin(el_rad)

        dist_range = self.light_cfg.get("distance_range", [10.0, 20.0])
        dist = rng_py.uniform(dist_range[0], dist_range[1])

        # Light color
        r_range = self.light_cfg.get("color_r_range", [0.85, 1.0])
        g_range = self.light_cfg.get("color_g_range", [0.80, 1.0])
        b_range = self.light_cfg.get("color_b_range", [0.75, 1.0])
        light_color = [
            rng_py.uniform(r_range[0], r_range[1]),
            rng_py.uniform(g_range[0], g_range[1]),
            rng_py.uniform(b_range[0], b_range[1]),
        ]

        # Coefficients
        amb_range = self.light_cfg.get("ambient_coeff_range", [0.25, 0.45])
        diff_range = self.light_cfg.get("diffuse_coeff_range", [0.50, 0.80])
        spec_range = self.light_cfg.get("specular_coeff_range", [0.05, 0.20])
        amb = rng_py.uniform(amb_range[0], amb_range[1])
        diff = rng_py.uniform(diff_range[0], diff_range[1])
        spec = rng_py.uniform(spec_range[0], spec_range[1])

        # Shadows on/off
        shadow_prob = float(self.light_cfg.get("shadow_prob", 0.85))
        shadow = rng_py.random() < shadow_prob

        # Optional low-light mode trigger
        low_light_enabled = bool(self.low_light_cfg.get("enabled", True))
        low_light_prob = float(self.low_light_cfg.get("prob", 0.15))
        is_low_light = low_light_enabled and (rng_py.random() < low_light_prob)

        if is_low_light:
            # Scale down ambient and diffuse illumination for low-light mode
            amb *= 0.35
            diff *= 0.35
            light_color = [c * 0.45 for c in light_color]

        return SequenceParams(
            seed=seed,
            wall_texture_type=w_type,
            floor_texture_type=f_type,
            wall_texture=w_img,
            floor_texture=f_img,
            light_direction=[dx, dy, dz],
            light_color=light_color,
            light_distance=dist,
            ambient_coeff=amb,
            diffuse_coeff=diff,
            specular_coeff=spec,
            shadow=shadow,
            is_low_light=is_low_light,
        )

    def apply_to_sim(
        self,
        maze_res: MazeResult,
        params: SequenceParams,
        physics_client: int,
    ) -> tuple[int, int]:
        """
        Write procedural textures to temporary files and bind them to the
        PyBullet simulation bodies (walls, floor).

        Parameters
        ----------
        maze_res : MazeResult
            Built maze containing wall and floor body IDs.
        params : SequenceParams
            Sampled sequence parameters with texture arrays.
        physics_client : int
            Active PyBullet client ID.

        Returns
        -------
        tuple[int, int]
            (wall_texture_unique_id, floor_texture_unique_id)
        """
        # Save textures as temporary PNGs for PyBullet loader
        with tempfile.NamedTemporaryFile(suffix="_wall.png", delete=False) as f_w:
            cv2.imwrite(f_w.name, params.wall_texture)
            w_path = Path(f_w.name)

        with tempfile.NamedTemporaryFile(suffix="_floor.png", delete=False) as f_f:
            cv2.imwrite(f_f.name, params.floor_texture)
            f_path = Path(f_f.name)

        try:
            wall_tex_id = p.loadTexture(str(w_path), physicsClientId=physics_client)
            floor_tex_id = p.loadTexture(str(f_path), physicsClientId=physics_client)

            # Apply floor texture
            if maze_res.floor_id >= 0:
                p.changeVisualShape(
                    maze_res.floor_id,
                    -1,
                    textureUniqueId=floor_tex_id,
                    physicsClientId=physics_client,
                )

            # Apply wall textures to all wall bodies
            for w_id in maze_res.wall_body_ids:
                p.changeVisualShape(
                    w_id,
                    -1,
                    textureUniqueId=wall_tex_id,
                    physicsClientId=physics_client,
                )

            return (wall_tex_id, floor_tex_id)
        finally:
            w_path.unlink(missing_ok=True)
            f_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Frame Post-Processing Augmentor
# --------------------------------------------------------------------------- #
class FrameAugmentor:
    """
    Applies realistic post-processing augmentations to rendered camera frames:
    brightness, contrast, gamma, sensor Gaussian noise, motion blur,
    lossy JPEG artifacts, optical vignette, and low-light mode.
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.config = load_randomization_config(self.config_path)
        self.frame_cfg = self.config.get("frame", {})

    # -- Transformation Primitives --

    @staticmethod
    def apply_brightness(img: np.ndarray, shift: float) -> np.ndarray:
        """Additive brightness adjustment."""
        return np.clip(img.astype(np.float32) + shift, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
        """Multiplicative contrast adjustment centered around 128."""
        return np.clip(128.0 + factor * (img.astype(np.float32) - 128.0), 0, 255).astype(
            np.uint8
        )

    @staticmethod
    def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
        """Power-law gamma correction curve."""
        inv_gamma = 1.0 / max(gamma, 1e-4)
        lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)]).astype(
            np.uint8
        )
        return cv2.LUT(img, lut)

    @staticmethod
    def apply_gaussian_noise(
        img: np.ndarray, std: float, rng_np: np.random.Generator
    ) -> np.ndarray:
        """Add zero-mean Gaussian sensor noise."""
        noise = rng_np.normal(0.0, std, size=img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_motion_blur(img: np.ndarray, kernel_size: int, angle_deg: float) -> np.ndarray:
        """Directional linear motion blur filter."""
        size = int(kernel_size)
        if size <= 1:
            return img
        if size % 2 == 0:
            size += 1

        kernel = np.zeros((size, size), dtype=np.float32)
        center = (size - 1) / 2.0
        theta = math.radians(angle_deg)
        dx = math.cos(theta)
        dy = math.sin(theta)

        for i in range(size):
            offset = i - center
            x = int(round(center + offset * dx))
            y = int(round(center + offset * dy))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0

        k_sum = kernel.sum()
        if k_sum > 0:
            kernel /= k_sum
        else:
            kernel[int(center), int(center)] = 1.0

        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
        """Simulate lossy digital/analog video link compression artifacts."""
        quality = int(np.clip(quality, 5, 100))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc = cv2.imencode(".jpg", img, encode_param)
        if not success:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec if dec is not None else img

    @staticmethod
    def apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
        """Radial optical lens vignette falloff towards corners."""
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return img

        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        max_dist = math.hypot(cx, cy)
        dist_norm = np.hypot(X - cx, Y - cy) / max_dist

        # Quadratic falloff
        mask = 1.0 - strength * (dist_norm**2)
        mask = np.clip(mask, 0.0, 1.0)[:, :, None]

        return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_low_light(
        img: np.ndarray,
        brightness_shift: float,
        contrast_factor: float,
        gamma: float,
        noise_std: float,
        saturation_factor: float,
        rng_np: np.random.Generator,
    ) -> np.ndarray:
        """Low-light / degraded camera sensor emulation."""
        # 1. Color desaturation in dark conditions
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 2. Exposure & contrast reduction
        out = FrameAugmentor.apply_contrast(out, contrast_factor)
        out = FrameAugmentor.apply_brightness(out, brightness_shift)
        out = FrameAugmentor.apply_gamma(out, gamma)

        # 3. High ISO noise
        out = FrameAugmentor.apply_gaussian_noise(out, noise_std, rng_np)
        return out

    # -- End-to-End Augmentation Pipeline --

    def augment(
        self,
        bgr: np.ndarray,
        is_low_light: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Execute full post-processing pipeline on rendered frame.

        Parameters
        ----------
        bgr : np.ndarray
            Input BGR frame of shape (H, W, 3) and uint8 dtype.
        is_low_light : bool
            Whether low-light mode is active for this frame/sequence.
        seed : int, optional
            Random seed for repeatable augmentation.

        Returns
        -------
        np.ndarray
            Augmented BGR frame (H, W, 3, uint8).
        """
        rng_py = random.Random(seed)
        rng_np = np.random.default_rng(seed)

        img = bgr.copy()

        # 1. Low-light mode
        if is_low_light:
            cfg_ll = self.frame_cfg.get("low_light", {})
            b_shift_range = cfg_ll.get("brightness_shift_range", [-65.0, -35.0])
            c_factor_range = cfg_ll.get("contrast_factor_range", [0.60, 0.85])
            gamma_range = cfg_ll.get("gamma_range", [1.30, 1.80])
            noise_std_range = cfg_ll.get("noise_std_range", [16.0, 36.0])
            sat_range = cfg_ll.get("saturation_factor_range", [0.40, 0.70])

            img = self.apply_low_light(
                img=img,
                brightness_shift=rng_py.uniform(b_shift_range[0], b_shift_range[1]),
                contrast_factor=rng_py.uniform(c_factor_range[0], c_factor_range[1]),
                gamma=rng_py.uniform(gamma_range[0], gamma_range[1]),
                noise_std=rng_py.uniform(noise_std_range[0], noise_std_range[1]),
                saturation_factor=rng_py.uniform(sat_range[0], sat_range[1]),
                rng_np=rng_np,
            )
        else:
            # 2. Standard Brightness
            cfg_b = self.frame_cfg.get("brightness", {})
            if cfg_b.get("enabled", True) and rng_py.random() < cfg_b.get("prob", 0.85):
                s_range = cfg_b.get("shift_range", [-25.0, 25.0])
                shift = rng_py.uniform(s_range[0], s_range[1])
                img = self.apply_brightness(img, shift)

            # 3. Standard Contrast
            cfg_c = self.frame_cfg.get("contrast", {})
            if cfg_c.get("enabled", True) and rng_py.random() < cfg_c.get("prob", 0.85):
                f_range = cfg_c.get("factor_range", [0.80, 1.25])
                factor = rng_py.uniform(f_range[0], f_range[1])
                img = self.apply_contrast(img, factor)

            # 4. Standard Gamma
            cfg_g = self.frame_cfg.get("gamma", {})
            if cfg_g.get("enabled", True) and rng_py.random() < cfg_g.get("prob", 0.80):
                g_range = cfg_g.get("gamma_range", [0.75, 1.35])
                gamma_val = rng_py.uniform(g_range[0], g_range[1])
                img = self.apply_gamma(img, gamma_val)

            # 5. Standard Gaussian Noise
            cfg_n = self.frame_cfg.get("gaussian_noise", {})
            if cfg_n.get("enabled", True) and rng_py.random() < cfg_n.get("prob", 0.75):
                std_range = cfg_n.get("std_range", [2.0, 14.0])
                std = rng_py.uniform(std_range[0], std_range[1])
                img = self.apply_gaussian_noise(img, std, rng_np)

        # 6. Motion Blur (independent of lighting)
        cfg_mb = self.frame_cfg.get("motion_blur", {})
        if cfg_mb.get("enabled", True) and rng_py.random() < cfg_mb.get("prob", 0.65):
            k_range = cfg_mb.get("kernel_size_range", [3, 9])
            ang_range = cfg_mb.get("angle_deg_range", [0.0, 360.0])
            k_size = rng_py.randint(k_range[0], k_range[1])
            angle = rng_py.uniform(ang_range[0], ang_range[1])
            img = self.apply_motion_blur(img, k_size, angle)

        # 7. JPEG Compression
        cfg_jpeg = self.frame_cfg.get("jpeg_compression", {})
        if cfg_jpeg.get("enabled", True) and rng_py.random() < cfg_jpeg.get("prob", 0.80):
            q_range = cfg_jpeg.get("quality_range", [40, 95])
            quality = rng_py.randint(q_range[0], q_range[1])
            img = self.apply_jpeg_compression(img, quality)

        # 8. Lens Vignette
        cfg_vig = self.frame_cfg.get("vignette", {})
        if cfg_vig.get("enabled", True) and rng_py.random() < cfg_vig.get("prob", 0.90):
            st_range = cfg_vig.get("strength_range", [0.15, 0.45])
            strength = rng_py.uniform(st_range[0], st_range[1])
            img = self.apply_vignette(img, strength)

        return img


# --------------------------------------------------------------------------- #
# CLI Demonstration
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="NIDAR Domain Randomization & Augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible generation")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to randomization.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/randomization_preview",
        help="Directory to save preview images",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  NIDAR Domain Randomization & Augmentation Pipeline")
    print("=" * 65)
    print(f"  Configuration : {args.config}")
    print(f"  Random Seed   : {args.seed}")
    print(f"  Output Dir    : {out_dir.resolve()}")
    print()

    # 1. Sample Sequence Parameters
    seq_rand = SequenceRandomizer(config_path=args.config)
    params = seq_rand.sample_sequence_params(seed=args.seed)

    print("Sampled Sequence Environmental Parameters:")
    print(f"  Wall texture  : {params.wall_texture_type}")
    print(f"  Floor texture : {params.floor_texture_type}")
    print(f"  Light vector  : [{params.light_direction[0]:.2f}, {params.light_direction[1]:.2f}, {params.light_direction[2]:.2f}]")
    print(f"  Light color   : [{params.light_color[0]:.2f}, {params.light_color[1]:.2f}, {params.light_color[2]:.2f}]")
    print(f"  Ambient coeff : {params.ambient_coeff:.2f}")
    print(f"  Diffuse coeff : {params.diffuse_coeff:.2f}")
    print(f"  Cast shadows  : {params.shadow}")
    print(f"  Low-light mode: {params.is_low_light}")
    print()

    # Save sampled texture previews
    cv2.imwrite(str(out_dir / "sampled_wall_texture.png"), params.wall_texture)
    cv2.imwrite(str(out_dir / "sampled_floor_texture.png"), params.floor_texture)

    # 2. Test Procedural Textures of all 4 types
    tex_gen = ProceduralTextureGenerator()
    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    for t_type in ["solid", "noise", "checker", "brick"]:
        _, img = tex_gen.sample_surface_texture("wall", forced_tex_type=t_type, rng_py=rng_py, rng_np=rng_np)
        cv2.imwrite(str(out_dir / f"texture_{t_type}.png"), img)
    print(f"Saved all 4 texture types to {out_dir}/texture_*.png")

    # 3. Test Frame Augmentations on a baseline image
    test_img = np.full((270, 480, 3), 128, dtype=np.uint8)
    cv2.putText(test_img, "NIDAR TEST", (140, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 240, 240), 2)

    augmentor = FrameAugmentor(config_path=args.config)
    aug_day = augmentor.augment(test_img, is_low_light=False, seed=args.seed)
    aug_night = augmentor.augment(test_img, is_low_light=True, seed=args.seed)

    cv2.imwrite(str(out_dir / "aug_standard.png"), aug_day)
    cv2.imwrite(str(out_dir / "aug_low_light.png"), aug_night)

    print(f"Saved sample augmented frames to {out_dir}/aug_*.png")
    print("=" * 65)
    print("Domain randomization initialization OK.")


if __name__ == "__main__":
    main()
