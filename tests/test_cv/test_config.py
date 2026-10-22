"""
tests/test_cv/test_config.py - Smoke & Unit Tests for NIDAR Configuration Loader

Validates that:
1. configs/default.yaml exists and parses cleanly.
2. The ConfigDict object enables dot-notation access to nested parameters.
3. Missing config paths raise a clear FileNotFoundError.
4. Serializing ConfigDict back to a dictionary preserves all nested structures.
"""

from pathlib import Path

import pytest

from cv.config import ConfigDict, load_config


def test_default_config_file_exists():
    """Verify that configs/default.yaml exists in the repository."""
    config_path = Path("configs/default.yaml")
    assert config_path.exists(), f"Missing default configuration at {config_path.resolve()}"


def test_load_default_config():
    """Verify that loading the default config succeeds and returns a ConfigDict."""
    cfg = load_config()
    assert isinstance(cfg, ConfigDict), "load_config() must return an instance of ConfigDict"


def test_default_config_expected_fields():
    """Smoke test: verify key configuration sections and fields are populated."""
    cfg = load_config()

    # Model parameters
    assert hasattr(cfg, "model"), "Config missing 'model' section"
    assert isinstance(cfg.model.name, str)
    assert 0.0 <= cfg.model.confidence_threshold <= 1.0
    assert cfg.model.image_size > 0
    assert isinstance(cfg.model.device, str)
    assert isinstance(cfg.model.target_classes, list)

    # Input/Output paths
    assert hasattr(cfg, "input"), "Config missing 'input' section"
    assert hasattr(cfg, "output"), "Config missing 'output' section"
    assert isinstance(cfg.output.results_dir, str)

    # Logging settings
    assert hasattr(cfg, "logging"), "Config missing 'logging' section"
    assert cfg.logging.level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def test_missing_config_raises_file_not_found():
    """Verify that a non-existent configuration path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent_config_12345.yaml")


def test_config_to_dict_roundtrip():
    """Verify that ConfigDict converts back into a standard Python dictionary."""
    data = {"subsystem": "cv", "parameters": {"fps": 30, "enabled": True}}
    cfg = ConfigDict(data)

    assert cfg.subsystem == "cv"
    assert cfg.parameters.fps == 30
    assert cfg.parameters.enabled is True

    exported = cfg.to_dict()
    assert exported == data


def test_arena_config_file_exists():
    """Verify that configs/arena.yaml exists in the repository."""
    config_path = Path("configs/arena.yaml")
    assert config_path.exists(), f"Missing arena configuration at {config_path.resolve()}"


def test_load_arena_config():
    """Verify that loading configs/arena.yaml succeeds and exposes all required simulation parameters."""
    cfg = load_config("configs/arena.yaml")
    assert isinstance(cfg, ConfigDict)

    # Arena geometry & materials
    assert hasattr(cfg, "arena"), "Config missing 'arena' section"
    assert cfg.arena.corridor_width > 0
    assert cfg.arena.wall_height > 0
    assert isinstance(cfg.arena.walls.color_rgb, list)
    assert isinstance(cfg.arena.floor.color_rgb, list)
    assert isinstance(cfg.arena.ceiling.present, bool)
    assert cfg.arena.ceiling.height > 0

    # Lighting
    assert hasattr(cfg, "lighting"), "Config missing 'lighting' section"
    assert isinstance(cfg.lighting.ambient_rgb, list)
    assert len(cfg.lighting.ambient_rgb) == 3

    # Camera sensor
    assert hasattr(cfg, "camera"), "Config missing 'camera' section"
    assert cfg.camera.height > 0
    assert cfg.camera.tilt_deg >= 0
    assert cfg.camera.fov.horizontal_deg > 0
    assert cfg.camera.fov.vertical_deg > 0
    assert cfg.camera.resolution.width > 0
    assert cfg.camera.resolution.height > 0

    # Survivor parameters
    assert hasattr(cfg, "survivor"), "Config missing 'survivor' section"
    assert cfg.survivor.max_count == 6
    assert isinstance(cfg.survivor.target_type, str)
    assert cfg.survivor.dimensions.height_m > 0
    assert isinstance(cfg.survivor.postures, list)
    assert len(cfg.survivor.postures) > 0

