"""
cv/config.py - YAML Configuration Loader for NIDAR CV Pipeline

Loads a YAML configuration file and provides typed, dot-notation access
to all settings. Falls back to configs/default.yaml if no path is given.

Usage:
    from cv.config import load_config

    cfg = load_config()                          # loads configs/default.yaml
    cfg = load_config("configs/custom.yaml")     # loads a custom override

    print(cfg.model.name)                  # "yolov8n.pt"
    print(cfg.model.confidence_threshold)  # 0.5
    print(cfg.model.image_size)            # 640
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Default config path (relative to project root)
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class ConfigDict:
    """
    A simple wrapper that turns a nested dictionary into an object with
    dot-notation access.

    Example:
        d = {"model": {"name": "yolov8n.pt", "confidence_threshold": 0.5}}
        cfg = ConfigDict(d)
        cfg.model.name  # "yolov8n.pt"

    Why not just use a plain dict?
    Because cfg.model.name is clearer and less error-prone than
    cfg["model"]["name"] when you are reading code at 2 AM before
    a competition run.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively wrap nested dicts so dot-access works at all levels
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a plain dictionary (useful for serialisation)."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"ConfigDict({self.to_dict()})"


def load_config(config_path: str | Path | None = None) -> ConfigDict:
    """
    Load a YAML configuration file and return a ConfigDict.

    Args:
        config_path: Path to a YAML file. If None, loads configs/default.yaml.

    Returns:
        ConfigDict with all settings accessible via dot-notation.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML syntax is invalid.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path.resolve()}\n"
            f"Expected at: {DEFAULT_CONFIG_PATH.resolve()}\n"
            f"Run from the project root directory (d:\\NIDAR)."
        )

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got: {type(raw)}")

    return ConfigDict(raw)


# ---------------------------------------------------------------------------
# Quick self-test: run this file directly to verify the config loads.
# Usage: .venv/Scripts/python.exe -m cv.config
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded successfully from:", DEFAULT_CONFIG_PATH.resolve())
    print()
    print(f"  Model name            : {cfg.model.name}")
    print(f"  Confidence threshold  : {cfg.model.confidence_threshold}")
    print(f"  Image size            : {cfg.model.image_size}")
    print(f"  Device                : {cfg.model.device}")
    print(f"  Target classes        : {cfg.model.target_classes}")
    print(f"  Video path            : {cfg.input.video_path}")
    print(f"  Results dir           : {cfg.output.results_dir}")
    print(f"  Log level             : {cfg.logging.level}")
    print()
    print("Full config dict:")
    print(cfg.to_dict())
