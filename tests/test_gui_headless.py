"""Headless automated tests for Qt/PySide6 operator workstation GUI.

Runs with QT_QPA_PLATFORM=offscreen and pytest-qt to validate:
- Creation and initialization of CanvasViewer and MainWindow.
- Safe loading of 8K (7680x4320) aerial mock images.
- Pristine preservation of background QGraphicsPixmapItem (non-destructive architecture).
- Correct creation of QGraphicsRectItem and QGraphicsSimpleTextItem vector primitives.
- Dynamic filtering by confidence threshold and class visibility.
- Smooth cursor-centered zoom and pan transformations.
- Full UI workflow: loading image, running detection simulation, populating target table,
  and focusing on targets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Ensure offscreen platform is set before any Qt import
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from typing import List

import pytest
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QMouseEvent,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)

from gui.canvas_viewer import CanvasViewer
from gui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance exists for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestCanvasViewerHeadless:
    """Test suite for the CanvasViewer widget."""

    def test_viewer_initialization(self, qapp):
        """Test clean initialization of CanvasViewer and its graphics scene."""
        viewer = CanvasViewer()
        assert viewer.scene() is not None
        assert viewer.has_image() is False
        assert viewer.get_image_size() == (0, 0)
        assert len(viewer.get_detections()) == 0
        assert viewer.get_zoom_level() == 1.0

    def test_load_mock_8k_image(self, qapp):
        """Test loading a mock 8K (7680x4320) pixmap into CanvasViewer."""
        viewer = CanvasViewer()
        w, h = 7680, 4320

        # Create 8K mock raster
        mock_pixmap = QPixmap(w, h)
        mock_pixmap.fill(Qt.GlobalColor.darkGray)

        viewer.set_image(mock_pixmap)

        assert viewer.has_image() is True
        assert viewer.get_image_size() == (w, h)

        # Scene bounds must strictly match 8K image
        scene_rect = viewer.scene().sceneRect()
        assert scene_rect.width() == float(w)
        assert scene_rect.height() == float(h)

        # Pristine background layer verification
        items = viewer.scene().items()
        pixmap_items = [it for it in items if isinstance(it, QGraphicsPixmapItem)]
        assert len(pixmap_items) == 1
        bg_item = pixmap_items[0]
        assert bg_item.zValue() == 0.0
        assert bg_item.pixmap().width() == w
        assert bg_item.pixmap().height() == h

    def test_non_destructive_vector_overlay(self, qapp):
        """Test vector overlay creation with absolute 8K pixel coordinates.

        Verifies that:
        - QGraphicsRectItem and QGraphicsSimpleTextItem are added.
        - Background QGraphicsPixmapItem remains intact and unchanged.
        - Primitives have correct Z-values (overlay on top).
        """
        viewer = CanvasViewer()
        w, h = 7680, 4320
        pixmap = QPixmap(w, h)
        pixmap.fill(Qt.GlobalColor.black)
        viewer.set_image(pixmap)

        # Sample detections in absolute 8K pixel coordinates
        sample_detections = [
            {"id": 1, "x": 4200.0, "y": 1800.0, "w": 64.0, "h": 48.0, "conf": 0.92, "class_id": 1},
            {"id": 2, "x": 1200.0, "y": 3000.0, "w": 120.0, "h": 90.0, "conf": 0.85, "class_id": 2},
            {"id": 3, "x": 6000.0, "y": 800.0, "w": 50.0, "h": 40.0, "conf": 0.42, "class_id": 0},
        ]

        viewer.update_detections(sample_detections)

        scene_items = viewer.scene().items()

        # Background raster layer must still be present and solitary
        bg_items = [it for it in scene_items if isinstance(it, QGraphicsPixmapItem)]
        assert len(bg_items) == 1
        assert bg_items[0].zValue() == 0.0

        # Verify QGraphicsRectItem primitives (bounding boxes and badge backgrounds)
        rect_items = [it for it in scene_items if isinstance(it, QGraphicsRectItem)]
        # 3 bounding boxes + 3 badge backgrounds = 6 rect items
        assert len(rect_items) == 6

        # Verify bounding boxes have cosmetic pens (for crispness across zoom levels)
        box_rect_items = [it for it in rect_items if it.zValue() == 10.0]
        assert len(box_rect_items) == 3
        for box in box_rect_items:
            assert box.pen().isCosmetic() is True

        # Verify QGraphicsSimpleTextItem primitives
        text_items = [it for it in scene_items if isinstance(it, QGraphicsSimpleTextItem)]
        assert len(text_items) == 3
        for txt in text_items:
            assert txt.zValue() >= 10.0
            # Must contain confidence percentage
            assert "%" in txt.text()

        assert len(viewer.get_detections()) == 3

    def test_filtering_by_confidence(self, qapp):
        """Test filtering detection primitives by confidence threshold."""
        viewer = CanvasViewer()
        pixmap = QPixmap(1000, 1000)
        viewer.set_image(pixmap)

        detections = [
            {"id": 1, "x": 100, "y": 100, "w": 50, "h": 50, "conf": 0.90, "class_id": 0},
            {"id": 2, "x": 200, "y": 200, "w": 50, "h": 50, "conf": 0.70, "class_id": 1},
            {"id": 3, "x": 300, "y": 300, "w": 50, "h": 50, "conf": 0.35, "class_id": 2},
        ]
        viewer.update_detections(detections)

        # Baseline: with min_conf = 0.0, all 3 are visible
        viewer.set_min_confidence(0.0)
        assert len(viewer.get_visible_detections()) == 3

        # Filter with min_conf = 0.50 -> item 3 should be hidden
        viewer.set_min_confidence(0.50)
        visible_50 = viewer.get_visible_detections()
        assert len(visible_50) == 2
        assert {d["id"] for d in visible_50} == {1, 2}

        # Filter with min_conf = 0.80 -> only item 1 should be visible
        viewer.set_min_confidence(0.80)
        visible_80 = viewer.get_visible_detections()
        assert len(visible_80) == 1
        assert visible_80[0]["id"] == 1

    def test_filtering_by_class_visibility(self, qapp):
        """Test toggling visibility of individual detection classes."""
        viewer = CanvasViewer()
        pixmap = QPixmap(1000, 1000)
        viewer.set_image(pixmap)

        detections = [
            {"id": 1, "x": 100, "y": 100, "w": 50, "h": 50, "conf": 0.9, "class_id": 0},
            {"id": 2, "x": 200, "y": 200, "w": 50, "h": 50, "conf": 0.9, "class_id": 1},
            {"id": 3, "x": 300, "y": 300, "w": 50, "h": 50, "conf": 0.9, "class_id": 2},
        ]
        viewer.update_detections(detections)

        # Hide class 1
        viewer.set_class_visibility(class_id=1, visible=False)
        visible = viewer.get_visible_detections()
        assert len(visible) == 2
        assert {d["class_id"] for d in visible} == {0, 2}

        # Re-enable class 1
        viewer.set_class_visibility(class_id=1, visible=True)
        assert len(viewer.get_visible_detections()) == 3

        # Hide all classes
        viewer.set_all_classes_visibility(False)
        assert len(viewer.get_visible_detections()) == 0

    def test_zoom_and_pan(self, qapp):
        """Test smooth zoom scaling and pan operations."""
        viewer = CanvasViewer()
        viewer.resize(800, 600)
        pixmap = QPixmap(4000, 3000)
        viewer.set_image(pixmap)

        initial_zoom = viewer.get_zoom_level()

        # Simulate wheel event (zoom in)
        wheel_event_in = QWheelEvent(
            QPointF(400, 300),
            QPointF(400, 300),
            QPoint(0, 0),
            QPoint(0, 120),  # positive delta: zoom in
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        viewer.wheelEvent(wheel_event_in)
        assert viewer.get_zoom_level() > initial_zoom

        # Reset zoom to 100%
        viewer.reset_zoom()
        assert viewer.get_zoom_level() == 1.0

        # Fit in view
        viewer.fit_to_view()
        assert viewer.get_zoom_level() < 1.0  # 4000x3000 image in 800x600 view must scale down

    def test_focus_on_detection(self, qapp):
        """Test centering the viewport on a specific detection target."""
        viewer = CanvasViewer()
        viewer.resize(800, 600)
        pixmap = QPixmap(7680, 4320)
        viewer.set_image(pixmap)

        detections = [
            {"id": 1, "x": 5000.0, "y": 3000.0, "w": 100.0, "h": 80.0, "conf": 0.95, "class_id": 2},
        ]
        viewer.update_detections(detections)

        success = viewer.focus_on_detection(0, target_zoom=2.0)
        assert success is True
        assert viewer.get_zoom_level() == 2.0

        # Invalid index returns False
        assert viewer.focus_on_detection(99) is False


class TestMainWindowHeadless:
    """Test suite for the MainWindow workstation interface."""

    def test_main_window_initialization(self, qapp):
        """Verify MainWindow initializes all required controls and panels."""
        win = MainWindow()

        # Central canvas
        assert isinstance(win.canvas, CanvasViewer)

        # Altitude selector requirements: QSpinBox, 10-500m
        assert win.spin_altitude.minimum() == 10
        assert win.spin_altitude.maximum() == 500
        assert win.spin_altitude.value() == 120
        assert "м" in win.spin_altitude.suffix()

        # VRAM options
        assert win.combo_vram.count() >= 5
        assert "Авто" in win.combo_vram.itemText(0)

        # Buttons and controls
        assert win.btn_open is not None
        assert win.btn_load_mock_8k is not None
        assert win.btn_detect is not None
        assert win.btn_detect.isEnabled() is False  # disabled until image loaded

        # Target list table
        assert win.table_targets.columnCount() == 4

        # Status and progress bar
        assert win.progress_bar is not None
        assert win.status_bar is not None

    def test_main_window_load_mock_8k(self, qapp):
        """Verify loading mock 8K pattern updates UI state."""
        win = MainWindow()

        win.load_mock_8k_image()

        assert win.canvas.has_image() is True
        assert win.canvas.get_image_size() == (7680, 4320)
        assert win.btn_detect.isEnabled() is True
        assert "7680 × 4320" in win.lbl_image_info.text()

    def test_main_window_detection_workflow(self, qapp, qtbot):
        """Verify full detection simulation workflow.

        - Triggers detection on mock 8K image.
        - Verifies worker emits results.
        - Verifies detections overlay in canvas.
        - Verifies target list table is populated.
        - Verifies status bar shows processing time and FPS.
        """
        win = MainWindow()
        win.load_mock_8k_image()

        # Execute detection
        win.start_detection()
        assert win._worker is not None

        # Wait for worker thread to complete using qtbot
        with qtbot.waitSignal(win._worker.detection_finished, timeout=5000):
            pass

        # Verify detections were added to canvas
        detections = win.canvas.get_detections()
        assert len(detections) >= 3
        assert len(detections) <= 5

        # Verify table populated with same number of rows
        assert win.table_targets.rowCount() == len(detections)

        # Verify status bar contains processing metrics
        assert "мс" in win.lbl_status_proc.text()
        assert "FPS" in win.lbl_status_proc.text()

    def test_main_window_target_table_selection(self, qapp, qtbot):
        """Verify selecting a row in target table focuses canvas on detection."""
        win = MainWindow()
        win.load_mock_8k_image()

        # Provide deterministic mock detections directly
        mock_dets = [
            {"id": 1, "x": 2000.0, "y": 1500.0, "w": 80.0, "h": 60.0, "conf": 0.94, "class_id": 1},
            {"id": 2, "x": 5500.0, "y": 3200.0, "w": 110.0, "h": 85.0, "conf": 0.88, "class_id": 2},
        ]
        win.canvas.update_detections(mock_dets)
        win._populate_target_table(mock_dets)

        assert win.table_targets.rowCount() == 2

        # Select row 1
        win.table_targets.selectRow(1)
        # Verify canvas zoom was updated by focus_on_detection
        assert win.canvas.get_zoom_level() == 1.5

    def test_confidence_slider_interaction(self, qapp):
        """Verify slider changes update CanvasViewer confidence threshold."""
        win = MainWindow()
        win.slider_conf.setValue(60)

        assert win.lbl_conf_val.text() == "60%"
        assert win.canvas._min_confidence == 0.60
