"""Stress, geometry, filtering, and C++ ownership tests for CanvasViewer.

Validates:
- Zero crashes (none_dealloc / SIGABRT) across 30 consecutive cycles x 1,500 detections (45,000 items).
- Fast batch insertion with BspTreeIndex restored afterward (no quadratic degradation).
- Hierarchical parent-child QGraphicsItem ownership (rect_item -> badge_bg -> text_item).
- Accurate child-local coordinates (text_item.pos() == QPointF(4.0, 2.0)) across toggles.
- Confidence and class filtering propagating cleanly to child badge and text items.
- Clean removal via clear_detections() leaving strictly 1 background QGraphicsPixmapItem.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsScene

from gui.canvas_viewer import CanvasViewer


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance exists for offscreen Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def canvas_viewer(qapp):
    """Create a CanvasViewer instance with deterministic Qt teardown in offscreen mode."""
    viewer = CanvasViewer()
    yield viewer
    viewer.clear_detections()
    viewer.close()
    viewer.deleteLater()
    QApplication.processEvents()
    gc.collect()


def _generate_dense_detections(count: int = 1500, seed_offset: int = 0) -> List[Dict[str, Any]]:
    """Generate deterministic dense detections across an 8K canvas."""
    dets: List[Dict[str, Any]] = []
    for i in range(count):
        cls_id = (i + seed_offset) % 16
        conf = 0.20 + ((i + seed_offset) % 80) * 0.01  # spans [0.20, 0.99]
        x = float(((i * 97 + seed_offset * 13) % 7500) + 10)
        y = float(((i * 53 + seed_offset * 17) % 4200) + 40)
        dets.append(
            {
                "id": i + 1,
                "x": x,
                "y": y,
                "w": 48.0,
                "h": 36.0,
                "conf": round(conf, 2),
                "class_id": cls_id,
            }
        )
    return dets


class TestCanvasViewerStressAndHierarchy:
    """Stress and parent-child geometry tests for dense aerial scenes."""

    def test_dense_30_cycles_1500_detections_stress_and_performance(self, canvas_viewer):
        """Run 30 consecutive update_detections cycles with 1,500 targets (45,000 total creations/deletions)."""
        viewer = canvas_viewer
        pixmap = QPixmap(7680, 4320)
        pixmap.fill(Qt.GlobalColor.black)
        viewer.set_image(pixmap)

        num_cycles = 30
        num_dets = 1500
        timings_ms: List[float] = []

        # Warmup cycle to populate font metrics and class style cache
        viewer.update_detections(_generate_dense_detections(num_dets, seed_offset=0))
        QApplication.processEvents()

        for cycle in range(1, num_cycles + 1):
            dets = _generate_dense_detections(num_dets, seed_offset=cycle)
            t0 = time.perf_counter()
            viewer.update_detections(dets)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings_ms.append(elapsed_ms)

            # Verify no leaks: QGraphicsScene.items() returns both top-level and child items
            all_scene_items = viewer._scene.items()
            top_level_items = [it for it in all_scene_items if it.parentItem() is None]
            assert len(top_level_items) == num_dets + 1
            assert len(all_scene_items) == 1 + num_dets * 3
            assert len(viewer._detection_items) == num_dets
            # BspTreeIndex must be active after update completes
            assert viewer._scene.itemIndexMethod() == QGraphicsScene.ItemIndexMethod.BspTreeIndex

            # Simulate real Qt event loop between frames so deferred scene/view events flush cleanly
            QApplication.processEvents()
            if cycle % 5 == 0:
                gc.collect()

        # Adaptive performance threshold: allow extra headroom for software offscreen CPU runners in CI
        is_ci = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))
        max_median_ms = 250.0 if is_ci else 100.0
        median_ms = sorted(timings_ms)[len(timings_ms) // 2]
        assert median_ms < max_median_ms, (
            f"Median update_detections time for {num_dets} targets was {median_ms:.2f} ms "
            f"(limit: < {max_median_ms:.1f} ms; min={min(timings_ms):.2f} ms, max={max(timings_ms):.2f} ms)"
        )

        # Clean up and verify strictly 1 item (_pixmap_item) remains
        viewer.clear_detections()
        QApplication.processEvents()
        gc.collect()
        remaining_items = viewer._scene.items()
        assert len(remaining_items) == 1
        assert remaining_items[0] is viewer._pixmap_item

    def test_hierarchy_filtering_and_visual_toggles_geometry(self, canvas_viewer):
        """Verify parent-child ownership, confidence/class filtering, zoom flags, and local coordinates."""
        viewer = canvas_viewer
        pixmap = QPixmap(4000, 3000)
        pixmap.fill(Qt.GlobalColor.black)
        viewer.set_image(pixmap)

        dets = [
            {"id": 1, "x": 100.0, "y": 200.0, "w": 60.0, "h": 40.0, "conf": 0.90, "class_id": 0},
            {"id": 2, "x": 400.0, "y": 500.0, "w": 80.0, "h": 50.0, "conf": 0.80, "class_id": 1},
            {"id": 3, "x": 700.0, "y": 800.0, "w": 50.0, "h": 50.0, "conf": 0.30, "class_id": 1},
        ]
        viewer.update_detections(dets)

        # 1. Verify parent-child hierarchy & cached resources
        assert len(viewer._detection_items) == 3
        for item in viewer._detection_items:
            assert item.rect_item.parentItem() is None
            assert item.badge_bg.parentItem() is item.rect_item
            assert item.text_item.parentItem() is item.badge_bg
            assert item.text_item.pos() == QPointF(4.0, 2.0)

        # 2. Confidence filter (0.5) hides det #3 and its children (badge_bg, text_item)
        viewer.set_min_confidence(0.5)
        item0, item1, item2 = viewer._detection_items
        assert item0.rect_item.isVisible() is True
        assert item0.badge_bg.isVisible() is True
        assert item0.text_item.isVisible() is True

        assert item1.rect_item.isVisible() is True
        assert item1.badge_bg.isVisible() is True
        assert item1.text_item.isVisible() is True

        assert item2.rect_item.isVisible() is False
        assert item2.badge_bg.isVisible() is False
        assert item2.text_item.isVisible() is False

        # 3. Class filter hides class 0 (det #1) along with its children
        viewer.set_class_visibility(0, False)
        assert item0.rect_item.isVisible() is False
        assert item0.badge_bg.isVisible() is False
        assert item0.text_item.isVisible() is False
        assert item1.rect_item.isVisible() is True

        # 4. Toggle scale_text_with_zoom=False & show_class_ids_only=True
        viewer.set_scale_text_with_zoom(False)
        viewer.set_show_class_ids_only(True)
        flag = QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        for item in viewer._detection_items:
            assert bool(item.badge_bg.flags() & flag) is True
            assert bool(item.text_item.flags() & flag) is False
            assert item.text_item.pos() == QPointF(4.0, 2.0)
            assert item.text_item.text().startswith(f"#{item.class_id:02d}")

        # 5. clear_detections() leaves strictly 1 item (_pixmap_item)
        viewer.clear_detections()
        QApplication.processEvents()
        assert len(viewer._scene.items()) == 1
        assert viewer._scene.items()[0] is viewer._pixmap_item
