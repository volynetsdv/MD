"""Unit Tests for Dynamic Class Discovery & Fallback Mechanism."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from gui.async_worker import InferenceWorker, format_detections_for_gui
from gui.batch_triage_worker import BatchTriageWorker
from gui.canvas_viewer import CanvasViewer
from gui.main_window import MainWindow
from src.detector_dispatcher import (
    DEFAULT_CLASS_NAMES,
    UnifiedDetector,
    _parse_names_metadata,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_default_class_names_structure():
    assert len(DEFAULT_CLASS_NAMES) == 16
    assert DEFAULT_CLASS_NAMES[0] == "Літак (plane)"
    assert DEFAULT_CLASS_NAMES[1] == "Судно / Корабель (ship)"


def test_parse_names_metadata():
    # Dict
    d = {0: "plane", 1: "ship"}
    assert _parse_names_metadata(d) == {0: "plane", 1: "ship"}

    # List
    lst = ["plane", "ship", "tank"]
    assert _parse_names_metadata(lst) == {0: "plane", 1: "ship", 2: "tank"}

    # JSON string
    j_str = json.dumps({0: "plane", 1: "ship"})
    assert _parse_names_metadata(j_str) == {0: "plane", 1: "ship"}

    # Python dict string
    py_str = "{0: 'plane', 1: 'ship'}"
    assert _parse_names_metadata(py_str) == {0: "plane", 1: "ship"}

    # None / Empty
    assert _parse_names_metadata(None) is None
    assert _parse_names_metadata("") is None


def test_unified_detector_get_class_names():
    det = UnifiedDetector()
    names = det.get_class_names()
    assert isinstance(names, dict)
    assert len(names) >= 15
    assert 0 in names

    # Override with custom class names
    custom = {0: "custom_plane", 1: "custom_tank"}
    det.set_class_names(custom)
    assert det.get_class_names() == custom

    # Reset
    det.set_class_names(None)
    assert det.get_class_names() == DEFAULT_CLASS_NAMES


def test_format_detections_with_custom_classes():
    custom = {0: "custom_uav", 1: "custom_vessel"}
    dets = [
        {"class_id": 0, "conf": 0.85, "bbox": [10, 20, 30, 40]},
        {"class_id": 1, "conf": 0.90, "bbox": [50, 60, 70, 80]},
    ]
    formatted = format_detections_for_gui(dets, class_names=custom)
    assert formatted[0]["class_name"] == "custom_uav"
    assert formatted[1]["class_name"] == "custom_vessel"


def test_canvas_viewer_set_class_names(qapp):
    viewer = CanvasViewer()
    custom = {0: "custom_plane", 1: "custom_ship"}
    viewer.set_class_names(custom)
    assert viewer._class_names[0] == "custom_plane"
    assert viewer._class_names[1] == "custom_ship"


def test_main_window_update_class_names(qapp):
    win = MainWindow()
    custom = {0: "UAV", 1: "Submarine"}
    win.update_class_names(custom)
    assert len(win._class_checkboxes) == 2
    assert win._class_checkboxes[0].text() == "[00] UAV"
    assert win._class_checkboxes[1].text() == "[01] Submarine"

    # Reset to default
    win.update_class_names(None)
    assert len(win._class_checkboxes) == 16


def test_inference_and_batch_workers_class_names(tmp_path: Path):
    custom = {0: "UAV", 1: "Drone"}
    inf_worker = InferenceWorker(
        image_source="dummy.png",
        class_names=custom,
    )
    assert inf_worker.get_class_names() == custom

    batch_worker = BatchTriageWorker(
        input_folder=tmp_path,
        class_names=custom,
    )
    assert batch_worker.get_class_names() == custom
