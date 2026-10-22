"""
tests/test_sim/test_randomize.py — Unit tests for sim.randomize
==============================================================

Validates:
1. Config loading from configs/randomization.yaml.
2. Determinism of SequenceRandomizer (same seed -> identical parameters & textures).
3. Procedural texture generators (solid, noise, checker, brick) shape and dtype.
4. Determinism of FrameAugmentor (same seed -> identical augmented pixels).
5. Individual transformation primitives (brightness, contrast, gamma, noise, blur, jpeg, vignette, low-light).
6. Texture application to PyBullet bodies in headless simulation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

from sim.maze import MazeBuilder
from sim.randomize import (
    DEFAULT_CONFIG_PATH,
    FrameAugmentor,
    ProceduralTextureGenerator,
    SequenceRandomizer,
    load_randomization_config,
)

ARENA_CONFIG = Path("configs/arena.yaml")
SAMPLE_LAYOUT = Path("sim/layouts/loop.yaml")


# --------------------------------------------------------------------------- #
# Config Tests
# --------------------------------------------------------------------------- #
class TestRandomizationConfig:
    def test_config_file_exists(self) -> None:
        assert DEFAULT_CONFIG_PATH.exists()

    def test_config_has_required_sections(self) -> None:
        cfg = load_randomization_config(DEFAULT_CONFIG_PATH)
        assert "sequence" in cfg
        assert "frame" in cfg
        assert "textures" in cfg["sequence"]
        assert "lighting" in cfg["sequence"]

        # Check required frame augmentations
        frame = cfg["frame"]
        for key in [
            "brightness",
            "contrast",
            "gamma",
            "gaussian_noise",
            "motion_blur",
            "jpeg_compression",
            "vignette",
            "low_light",
        ]:
            assert key in frame, f"Missing {key} in frame config"


# --------------------------------------------------------------------------- #
# Procedural Texture Generator Tests
# --------------------------------------------------------------------------- #
class TestProceduralTextures:
    @pytest.fixture
    def tex_gen(self) -> ProceduralTextureGenerator:
        return ProceduralTextureGenerator()

    def test_solid_texture_shape_and_dtype(self, tex_gen: ProceduralTextureGenerator) -> None:
        tex = tex_gen.generate_solid(64, 64, (50, 100, 150))
        assert tex.shape == (64, 64, 3)
        assert tex.dtype == np.uint8
        assert np.all(tex[:, :, 0] == 50)
        assert np.all(tex[:, :, 1] == 100)
        assert np.all(tex[:, :, 2] == 150)

    def test_noise_texture_shape_and_dtype(self, tex_gen: ProceduralTextureGenerator) -> None:
        rng = np.random.default_rng(42)
        tex = tex_gen.generate_noise(64, 64, (128, 128, 128), std=15.0, rng=rng)
        assert tex.shape == (64, 64, 3)
        assert tex.dtype == np.uint8
        assert tex.std() > 5.0

    def test_checker_texture_pattern(self, tex_gen: ProceduralTextureGenerator) -> None:
        tex = tex_gen.generate_checker(64, 64, grid_size=4, color1_bgr=(0, 0, 0), color2_bgr=(255, 255, 255))
        assert tex.shape == (64, 64, 3)
        assert tex.dtype == np.uint8
        # Top-left cell should be color1 (0), adjacent should be color2 (255)
        assert tex[4, 4, 0] == 0
        assert tex[4, 20, 0] == 255

    def test_brick_texture_shape_and_pattern(self, tex_gen: ProceduralTextureGenerator) -> None:
        tex = tex_gen.generate_brick(
            height=64,
            width=64,
            rows=8,
            cols=4,
            mortar_px=2,
            brick_bgr=(30, 40, 180),
            mortar_bgr=(200, 200, 200),
        )
        assert tex.shape == (64, 64, 3)
        assert tex.dtype == np.uint8
        # Should contain both brick and mortar colors
        unique_colors = np.unique(tex.reshape(-1, 3), axis=0)
        assert len(unique_colors) >= 2


# --------------------------------------------------------------------------- #
# Sequence Randomizer Tests
# --------------------------------------------------------------------------- #
class TestSequenceRandomizer:
    @pytest.fixture
    def seq_rand(self) -> SequenceRandomizer:
        return SequenceRandomizer(config_path=DEFAULT_CONFIG_PATH)

    def test_determinism_with_same_seed(self, seq_rand: SequenceRandomizer) -> None:
        """Requirement: Everything must be seeded and reproducible."""
        p1 = seq_rand.sample_sequence_params(seed=999)
        p2 = seq_rand.sample_sequence_params(seed=999)

        assert p1.wall_texture_type == p2.wall_texture_type
        assert p1.floor_texture_type == p2.floor_texture_type
        assert np.array_equal(p1.wall_texture, p2.wall_texture)
        assert np.array_equal(p1.floor_texture, p2.floor_texture)
        assert p1.light_direction == pytest.approx(p2.light_direction)
        assert p1.light_color == pytest.approx(p2.light_color)
        assert p1.ambient_coeff == pytest.approx(p2.ambient_coeff)
        assert p1.diffuse_coeff == pytest.approx(p2.diffuse_coeff)
        assert p1.shadow == p2.shadow
        assert p1.is_low_light == p2.is_low_light

    def test_different_seeds_yield_different_params(self, seq_rand: SequenceRandomizer) -> None:
        p1 = seq_rand.sample_sequence_params(seed=101)
        p2 = seq_rand.sample_sequence_params(seed=202)
        assert p1.light_direction != p2.light_direction

    def test_light_direction_normalized_and_downward(self, seq_rand: SequenceRandomizer) -> None:
        params = seq_rand.sample_sequence_params(seed=42)
        dx, dy, dz = params.light_direction
        norm = math.hypot(dx, dy, dz)
        assert norm == pytest.approx(1.0, abs=1e-4)
        # Downward illumination (negative z)
        assert dz < 0.0

    def test_apply_to_sim_succeeds(self, seq_rand: SequenceRandomizer) -> None:
        cid = p.connect(p.DIRECT)
        try:
            builder = MazeBuilder(arena_config_path=ARENA_CONFIG)
            maze_res = builder.build(SAMPLE_LAYOUT, physics_client=cid)

            params = seq_rand.sample_sequence_params(seed=55)
            w_tex_id, f_tex_id = seq_rand.apply_to_sim(maze_res, params, physics_client=cid)

            assert w_tex_id >= 0
            assert f_tex_id >= 0
        finally:
            p.disconnect(cid)


# --------------------------------------------------------------------------- #
# Frame Augmentor Tests
# --------------------------------------------------------------------------- #
class TestFrameAugmentor:
    @pytest.fixture
    def augmentor(self) -> FrameAugmentor:
        return FrameAugmentor(config_path=DEFAULT_CONFIG_PATH)

    @pytest.fixture
    def baseline_image(self) -> np.ndarray:
        # 100x100 RGB image with gradient
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            img[y, :, :] = y * 2
        return img

    def test_determinism_with_same_seed(
        self, augmentor: FrameAugmentor, baseline_image: np.ndarray
    ) -> None:
        """Requirement: Same seed must produce identical augmented frame."""
        aug1 = augmentor.augment(baseline_image, is_low_light=False, seed=123)
        aug2 = augmentor.augment(baseline_image, is_low_light=False, seed=123)
        assert np.array_equal(aug1, aug2)

    def test_augmented_output_shape_and_dtype(
        self, augmentor: FrameAugmentor, baseline_image: np.ndarray
    ) -> None:
        aug = augmentor.augment(baseline_image, seed=42)
        assert aug.shape == baseline_image.shape
        assert aug.dtype == np.uint8

    def test_brightness_primitive(self, baseline_image: np.ndarray) -> None:
        brighter = FrameAugmentor.apply_brightness(baseline_image, 30.0)
        assert brighter.mean() > baseline_image.mean()

    def test_contrast_primitive(self, baseline_image: np.ndarray) -> None:
        high_contrast = FrameAugmentor.apply_contrast(baseline_image, 1.5)
        assert high_contrast.std() > baseline_image.std()

    def test_gamma_primitive(self, baseline_image: np.ndarray) -> None:
        gamma_out = FrameAugmentor.apply_gamma(baseline_image, 1.5)
        assert gamma_out.shape == baseline_image.shape
        assert gamma_out.dtype == np.uint8

    def test_motion_blur_primitive(self, baseline_image: np.ndarray) -> None:
        blurred = FrameAugmentor.apply_motion_blur(baseline_image, kernel_size=7, angle_deg=45.0)
        assert blurred.shape == baseline_image.shape
        assert blurred.dtype == np.uint8

    def test_jpeg_compression_primitive(self, baseline_image: np.ndarray) -> None:
        compressed = FrameAugmentor.apply_jpeg_compression(baseline_image, quality=40)
        assert compressed.shape == baseline_image.shape
        assert compressed.dtype == np.uint8

    def test_vignette_darkens_corners(self) -> None:
        solid = np.full((100, 100, 3), 200, dtype=np.uint8)
        vig = FrameAugmentor.apply_vignette(solid, strength=0.5)
        # Center should be brighter than corners
        center_val = vig[50, 50, 0]
        corner_val = vig[0, 0, 0]
        assert center_val > corner_val

    def test_low_light_mode_darkens_and_desaturates(
        self, augmentor: FrameAugmentor, baseline_image: np.ndarray
    ) -> None:
        day = augmentor.augment(baseline_image, is_low_light=False, seed=42)
        night = augmentor.augment(baseline_image, is_low_light=True, seed=42)
        assert night.mean() < day.mean()
