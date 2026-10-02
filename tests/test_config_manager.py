"""Unit tests for centralized configuration management (core/config_manager.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest

from core.config_manager import AppConfig, get_config


class TestAppConfig:
    """Comprehensive test suite for AppConfig persistence and validation."""

    def test_default_values(self):
        """Verify factory default values for all configuration domains."""
        cfg = AppConfig()

        # Detection
        assert cfg.default_confidence == 0.20
        assert cfg.default_iou == 0.45
        assert cfg.default_altitude == 150.0
        assert cfg.default_vram_limit_mb == 2048

        # Visualization
        assert cfg.show_class_ids_only is False
        assert cfg.scale_overlay_text_with_zoom is True
        assert cfg.base_font_size == 12
        assert cfg.box_border_width == 2

        # Geometry
        assert cfg.splitter_sizes == [300, 900, 300]
        assert cfg.window_geometry == ""
        assert cfg.splitter_state == ""

        # Paths
        assert cfg.models_dir == "models"
        assert cfg.recent_dir == "data/sliced_dota/images/val"

    def test_save_and_load(self, tmp_path: Path):
        """Verify atomic JSON serialization and clean round-trip deserialization."""
        config_path = tmp_path / "sub" / "settings.json"
        cfg = AppConfig(
            default_confidence=0.35,
            default_altitude=220.0,
            show_class_ids_only=True,
            base_font_size=14,
            splitter_sizes=[280, 800, 320],
        )

        assert cfg.save(config_path) is True
        assert config_path.exists()

        loaded_cfg = AppConfig.load(config_path)
        assert loaded_cfg.default_confidence == 0.35
        assert loaded_cfg.default_altitude == 220.0
        assert loaded_cfg.show_class_ids_only is True
        assert loaded_cfg.base_font_size == 14
        assert loaded_cfg.splitter_sizes == [280, 800, 320]

    def test_reset_to_defaults(self, tmp_path: Path):
        """Verify that reset_to_defaults restores all fields to initial factory values."""
        cfg = AppConfig(
            default_confidence=0.85,
            default_altitude=500.0,
            show_class_ids_only=True,
            scale_overlay_text_with_zoom=False,
            base_font_size=24,
        )

        cfg.reset_to_defaults()

        assert cfg.default_confidence == 0.20
        assert cfg.default_altitude == 150.0
        assert cfg.show_class_ids_only is False
        assert cfg.scale_overlay_text_with_zoom is True
        assert cfg.base_font_size == 12

    def test_corrupted_json_recovery(self, tmp_path: Path):
        """Verify that corrupted or unparseable JSON files gracefully regenerate valid defaults."""
        config_path = tmp_path / "corrupt_settings.json"
        config_path.write_text("{ broken json: [1, 2, 3, truncated...", encoding="utf-8")

        loaded = AppConfig.load(config_path)
        assert loaded.default_confidence == 0.20
        assert loaded.default_altitude == 150.0
        assert loaded.show_class_ids_only is False

        # Verify that valid JSON was rewritten to disk
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data.get("default_confidence") == 0.20

    def test_non_dict_json_recovery(self, tmp_path: Path):
        """Verify handling of JSON file containing an array instead of an object."""
        config_path = tmp_path / "array_settings.json"
        config_path.write_text('["item1", "item2"]', encoding="utf-8")

        loaded = AppConfig.load(config_path)
        assert loaded.default_confidence == 0.20
        assert loaded.models_dir == "models"

    def test_missing_file_creation(self, tmp_path: Path):
        """Verify that loading a non-existent path creates the file with default values."""
        config_path = tmp_path / "non_existent" / "settings.json"
        assert not config_path.exists()

        cfg = AppConfig.load(config_path)
        assert config_path.exists()
        assert cfg.default_confidence == 0.20

    def test_singleton_get_config(self, tmp_path: Path):
        """Verify get_config singleton access and reload capability."""
        cfg_file = tmp_path / "singleton_settings.json"
        c1 = get_config(reload=True, path=cfg_file)
        c2 = get_config()
        assert c1 is c2

        c1.default_confidence = 0.40
        c1.save(cfg_file)

        c3 = get_config(reload=True, path=cfg_file)
        assert c3.default_confidence == 0.40
