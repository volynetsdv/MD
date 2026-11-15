"""Headless tests for CanvasViewer and MainWindow UI enhancements.

Validates:
- Non-destructive vector bounding box and badge creation.
- Adaptive label toggling: full names vs compact ID tags (#ID).
- Fixed screen-space font scaling via ItemIgnoresTransformations flag.
- QSplitter 3-panel ergonomics, QScrollArea class panel, and SettingsDialog.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QScrollArea,
    QSplitter,
)

from core.config_manager import AppConfig, get_config
from gui.canvas_viewer import CanvasViewer
from gui.main_window import MainWindow
from gui.settings_dialog import SettingsDialog


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance exists for offscreen Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestCanvasViewerFeatures:
    """Test suite for adaptive labeling, font scaling, and visual overlays."""

    def test_bounding_box_and_primitive_creation(self, qapp):
        """Verify vector primitives creation in headless mode."""
        viewer = CanvasViewer()
        pixmap = QPixmap(1920, 1080)
        pixmap.fill(Qt.GlobalColor.black)
        viewer.set_image(pixmap)

        sample_detections = [
            {"id": 1, "x": 100.0, "y": 150.0, "w": 60.0, "h": 40.0, "conf": 0.94, "class_id": 0},
            {"id": 2, "x": 500.0, "y": 600.0, "w": 80.0, "h": 50.0, "conf": 0.88, "class_id": 9},
        ]
        viewer.update_detections(sample_detections)

        scene_items = viewer.scene().items()
        rect_items = [it for it in scene_items if isinstance(it, QGraphicsRectItem)]
        # 2 bounding boxes + 2 contrast badges = 4 rect items
        assert len(rect_items) == 4

        text_items = [it for it in scene_items if isinstance(it, QGraphicsSimpleTextItem)]
        assert len(text_items) == 2

    def test_show_class_ids_only_toggle(self, qapp):
        """Verify toggling compact ID mode switches text from 'plane' to '#00'."""
        viewer = CanvasViewer()
        viewer.set_show_class_ids_only(False)

        sample_detections = [
            {"id": 1, "x": 200.0, "y": 200.0, "w": 70.0, "h": 50.0, "conf": 0.92, "class_id": 0}
        ]
        viewer.update_detections(sample_detections)

        text_items = [it for it in viewer.scene().items() if isinstance(it, QGraphicsSimpleTextItem)]
        assert len(text_items) == 1
        initial_text = text_items[0].text()

        # By default, full name contains class name ('plane')
        assert "plane" in initial_text

        # Toggle compact ID mode
        viewer.set_show_class_ids_only(True)
        assert viewer.is_show_class_ids_only() is True

        compact_text = text_items[0].text()
        assert "#00" in compact_text
        assert "plane" not in compact_text
        assert "%" in compact_text

        # Toggle back to full names
        viewer.set_show_class_ids_only(False)
        assert viewer.is_show_class_ids_only() is False
        restored_text = text_items[0].text()
        assert "plane" in restored_text

    def test_scale_overlay_text_with_zoom_toggle(self, qapp):
        """Verify ItemIgnoresTransformations flag is toggled on text and badge items."""
        viewer = CanvasViewer()
        sample_detections = [
            {"id": 1, "x": 300.0, "y": 300.0, "w": 50.0, "h": 50.0, "conf": 0.85, "class_id": 1}
        ]
        viewer.update_detections(sample_detections)

        text_items = [it for it in viewer.scene().items() if isinstance(it, QGraphicsSimpleTextItem)]
        assert len(text_items) == 1
        txt_item = text_items[0]

        # 1. When scale_text_with_zoom is True (default), ItemIgnoresTransformations is NOT set
        viewer.set_scale_text_with_zoom(True)
        assert viewer.is_scale_text_with_zoom() is True
        flag = QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        assert not bool(txt_item.flags() & flag)

        # 2. When scale_text_with_zoom is False, ItemIgnoresTransformations MUST be set
        viewer.set_scale_text_with_zoom(False)
        assert viewer.is_scale_text_with_zoom() is False
        assert bool(txt_item.flags() & flag)

        # 3. Switching back clears the flag
        viewer.set_scale_text_with_zoom(True)
        assert not bool(txt_item.flags() & flag)

    def test_font_size_and_box_border_updates(self, qapp):
        """Verify setting base font size and border width updates items dynamically."""
        viewer = CanvasViewer()
        sample_detections = [
            {"id": 1, "x": 100.0, "y": 100.0, "w": 40.0, "h": 40.0, "conf": 0.80, "class_id": 2}
        ]
        viewer.update_detections(sample_detections)

        viewer.set_base_font_size(16)
        assert viewer.get_base_font_size() == 16

        viewer.set_box_border_width(4)
        assert viewer.get_box_border_width() == 4

        boxes = [it for it in viewer.scene().items() if isinstance(it, QGraphicsRectItem) and it.zValue() == 10.0]
        assert len(boxes) == 1
        assert boxes[0].pen().width() == 4


class TestMainWindowRefactoring:
    """Test suite for MainWindow 3-panel QSplitter, QScrollArea, and settings integration."""

    def test_splitter_three_panels(self, qapp):
        """Verify MainWindow has horizontal QSplitter with 3 panels: left, canvas, right."""
        win = MainWindow()
        assert hasattr(win, "splitter")
        assert isinstance(win.splitter, QSplitter)
        assert win.splitter.orientation() == Qt.Orientation.Horizontal
        assert win.splitter.count() == 3

        # Central widget is the canvas
        assert win.splitter.widget(1) is win.canvas

    def test_scroll_area_for_classes_with_numbering(self, qapp):
        """Verify 16 DOTA classes are inside QScrollArea and have [00] numbering."""
        win = MainWindow()

        # Find QScrollArea in window
        scroll_areas = win.findChildren(QScrollArea)
        assert len(scroll_areas) >= 1
        classes_scroll = scroll_areas[0]
        assert classes_scroll.widgetResizable() is True

        # Check checkboxes
        assert len(win._class_checkboxes) == 16

        # Check numbering format: [00], [01], ..., [15]
        chk_0 = win._class_checkboxes[0]
        assert chk_0.text().startswith("[00]")
        assert "plane" in chk_0.text()

        chk_9 = win._class_checkboxes[9]
        assert chk_9.text().startswith("[09]")

    def test_select_all_and_deselect_all_classes(self, qapp):
        """Verify helper methods to toggle all class checkboxes simultaneously."""
        win = MainWindow()

        # Deselect all
        win._deselect_all_classes()
        for chk in win._class_checkboxes.values():
            assert chk.isChecked() is False

        # Select all
        win._select_all_classes()
        for chk in win._class_checkboxes.values():
            assert chk.isChecked() is True

    def test_settings_dialog_fields_and_defaults_reset(self, qapp, tmp_path: Path):
        """Verify SettingsDialog edits parameters and resets to defaults cleanly."""
        cfg_file = tmp_path / "dialog_test_settings.json"
        cfg = get_config(reload=True, path=cfg_file)
        cfg.default_confidence = 0.50
        cfg.save(cfg_file)

        dialog = SettingsDialog()
        assert dialog.spin_conf.value() == 0.50

        # Click reset to defaults
        dialog._on_reset_to_defaults()
        assert dialog.spin_conf.value() == 0.20
        assert dialog.spin_iou.value() == 0.45
        assert dialog.spin_alt.value() == 150.0

    def test_close_event_persists_geometry(self, qapp, tmp_path: Path):
        """Verify closeEvent serializes window layout and splitter sizes to AppConfig."""
        cfg_file = tmp_path / "layout_settings.json"
        cfg = get_config(reload=True, path=cfg_file)

        win = MainWindow()
        win.splitter.setSizes([290, 850, 300])

        event = QCloseEvent()
        win.closeEvent(event)

        reloaded = AppConfig.load(cfg_file)
        assert len(reloaded.splitter_sizes) == 3

    def test_toolbar_canvas_tools_only_no_input_duplication(self, qapp):
        """Verify top toolbar contains only canvas tools and no duplicate input buttons."""
        win = MainWindow()

        # Left panel controls
        assert win.btn_open is not None
        assert win.btn_open_folder is not None
        assert win.btn_open_file is win.btn_open
        assert not hasattr(win, "btn_test_8k") or win.btn_test_8k is None
        assert not hasattr(win, "btn_load_mock_8k") or win.btn_load_mock_8k is None

        # Top toolbar action texts
        tb_actions = win.toolbar.actions()
        action_texts = [a.text() for a in tb_actions if not a.isSeparator()]

        # Must contain canvas tools
        assert "Масштабувати шрифт із зумом" in action_texts
        assert any("ID" in t for t in action_texts)
        assert "Вписати зображення" in action_texts
        assert "Масштаб 100%" in action_texts

        # Must NOT contain duplicate file opening actions
        assert not any("Фото" in t or "Папка" in t or "8K Тест" in t for t in action_texts)

    def test_dark_tactical_styles_applied(self, qapp):
        """Verify unified dark tactical styles are applied to window, scroll area, and dialog."""
        from gui.styles import DARK_TACTICAL_STYLE

        assert "#0b1326" in DARK_TACTICAL_STYLE
        assert "#0f1d3a" in DARK_TACTICAL_STYLE
        assert "#080f1e" in DARK_TACTICAL_STYLE
        assert "#1e3563" in DARK_TACTICAL_STYLE
        assert "QFileDialog" in DARK_TACTICAL_STYLE

        win = MainWindow()
        scroll_areas = win.findChildren(QScrollArea)
        assert len(scroll_areas) >= 1
        classes_scroll = scroll_areas[0]

        # Verify dark background on class scroll area
        assert "#080f1e" in classes_scroll.styleSheet()
