"""Centralized configuration manager for Aerial Reconnaissance Workstation.

Provides strongly typed configuration schema, atomic persistence to disk,
corrupted file recovery, and runtime synchronization for UI and detection components.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Determine repository root directory
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "settings.json"

DEFAULT_MODEL_URLS: Dict[str, str] = {
    "yolo_320.onnx": "https://github.com/volynetsdv/MD/releases/download/v1.0.0-weights/yolo_320.onnx",
    "yolo_416.onnx": "https://github.com/volynetsdv/MD/releases/download/v1.0.0-weights/yolo_416.onnx",
    "yolo_512.onnx": "https://github.com/volynetsdv/MD/releases/download/v1.0.0-weights/yolo_512.onnx",
    "yolo_640.onnx": "https://github.com/volynetsdv/MD/releases/download/v1.0.0-weights/yolo_640.onnx",
}


@dataclass
class AppConfig:
    """Application configuration schema with persistence and validation."""

    # 1. Detection parameters
    default_confidence: float = 0.20
    default_iou: float = 0.45
    default_altitude: float = 150.0
    default_vram_limit_mb: int = 2048

    # 2. Visualization & Overlay parameters
    show_class_ids_only: bool = False
    scale_overlay_text_with_zoom: bool = True
    base_font_size: int = 12
    box_border_width: int = 2

    # 3. Window & Layout geometry
    window_geometry: str = ""
    splitter_sizes: List[int] = field(default_factory=lambda: [300, 900, 300])
    splitter_state: str = ""

    # 4. Storage & Filesystem paths
    models_dir: str = "models"
    recent_dir: str = "data/sliced_dota/images/val"
    model_urls: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_URLS))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Atomically persist configuration to JSON file.

        Uses temporary file and atomic replace to prevent corrupted state on crash.
        """
        target_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = target_path.with_suffix(".tmp")

            data = self.to_dict()
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            # Atomic file replace
            tmp_path.replace(target_path)
            logger.debug("Configuration successfully saved to %s", target_path)
            return True
        except Exception as err:
            logger.error("Failed to save configuration to %s: %s", target_path, err)
            return False

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> AppConfig:
        """Load configuration from JSON file with validation and fallback to defaults.

        If file is missing or corrupted, returns default configuration and regenerates file.
        """
        target_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

        if not target_path.exists():
            logger.info("Configuration file not found at %s. Creating defaults.", target_path)
            config = cls()
            config.save(target_path)
            return config

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if not isinstance(raw_data, dict):
                raise ValueError("Config root must be a JSON object dictionary")

            validated_data: Dict[str, Any] = {}

            # Detection
            if "default_confidence" in raw_data:
                val = float(raw_data["default_confidence"])
                validated_data["default_confidence"] = max(0.01, min(1.0, val))

            if "default_iou" in raw_data:
                val = float(raw_data["default_iou"])
                validated_data["default_iou"] = max(0.01, min(1.0, val))

            if "default_altitude" in raw_data:
                val = float(raw_data["default_altitude"])
                validated_data["default_altitude"] = max(10.0, min(2000.0, val))

            if "default_vram_limit_mb" in raw_data:
                validated_data["default_vram_limit_mb"] = int(raw_data["default_vram_limit_mb"])

            # Visualization
            if "show_class_ids_only" in raw_data:
                validated_data["show_class_ids_only"] = bool(raw_data["show_class_ids_only"])

            if "scale_overlay_text_with_zoom" in raw_data:
                validated_data["scale_overlay_text_with_zoom"] = bool(
                    raw_data["scale_overlay_text_with_zoom"]
                )

            if "base_font_size" in raw_data:
                val = int(raw_data["base_font_size"])
                validated_data["base_font_size"] = max(6, min(36, val))

            if "box_border_width" in raw_data:
                val = int(raw_data["box_border_width"])
                validated_data["box_border_width"] = max(1, min(10, val))

            # Geometry
            if "window_geometry" in raw_data:
                validated_data["window_geometry"] = str(raw_data["window_geometry"])

            if "splitter_sizes" in raw_data and isinstance(raw_data["splitter_sizes"], list):
                validated_data["splitter_sizes"] = [int(x) for x in raw_data["splitter_sizes"]]

            if "splitter_state" in raw_data:
                validated_data["splitter_state"] = str(raw_data["splitter_state"])

            # Paths & Models
            if "models_dir" in raw_data:
                validated_data["models_dir"] = str(raw_data["models_dir"])

            if "recent_dir" in raw_data:
                validated_data["recent_dir"] = str(raw_data["recent_dir"])

            if "model_urls" in raw_data and isinstance(raw_data["model_urls"], dict):
                validated_data["model_urls"] = {
                    str(k): str(v) for k, v in raw_data["model_urls"].items()
                }

            return cls(**validated_data)

        except Exception as err:
            logger.warning(
                "Corrupted or invalid configuration file at %s (%s). Generating clean defaults.",
                target_path,
                err,
            )
            config = cls()
            config.save(target_path)
            return config

    def reset_to_defaults(self) -> None:
        """Reset all configuration values to their factory defaults."""
        defaults = AppConfig()
        for field_name in asdict(defaults).keys():
            setattr(self, field_name, getattr(defaults, field_name))


_GLOBAL_CONFIG: Optional[AppConfig] = None


def get_config(reload: bool = False, path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Retrieve global singleton AppConfig instance."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None or reload:
        _GLOBAL_CONFIG = AppConfig.load(path)
    return _GLOBAL_CONFIG
