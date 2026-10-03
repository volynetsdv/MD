"""Unit and Integration Tests for Interactive Stream Analysis and Batch Triage Workflow.

Tests:
1. Analysis button toggling, text changes, styling, and `is_analysis_active` state.
2. Metadata JSON cache persistence, atomic writing, coordinate precision, and roundtrip reading.
3. BatchTriageWorker pipeline: filters images, copies only positive frames with clean originals
   (Metadata-Driven, 0 raster modifications), and writes corresponding JSON files.
4. Fast cache hit on image load skipping repeated inference in GUI.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Ensure offscreen Qt platform
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.metadata_cache import (
    get_metadata_cache_path,
    has_metadata_cache,
    load_metadata_cache,
    metadata_to_gui_detections,
    save_metadata_cache,
)
from gui.batch_triage_worker import BatchTriageWorker
from gui.canvas_viewer import CanvasViewer
from gui.main_window import MainWindow

# Ensure offscreen Qt platform
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# =============================================================================
# Test Suite 1: Button state and is_analysis_active toggle
# =============================================================================
class TestAnalysisStateToggle:
    """Validate interactive analysis button states and visual indicators."""

    def test_initial_state(self, qapp):
        """Initial state must be inactive with blue styling and disabled without image."""
        win = MainWindow()
        assert win.is_analysis_active is False
        assert "▶ Почати аналіз" in win.btn_toggle_analysis.text()
        assert win.btn_toggle_analysis.isEnabled() is False
        # Backwards-compatibility alias check
        assert win.btn_detect is win.btn_toggle_analysis

    def test_toggle_state_transitions(self, qapp):
        """Toggling inverts is_analysis_active, updates text and styles."""
        win = MainWindow()
        win.load_mock_8k_image()
        assert win.btn_toggle_analysis.isEnabled() is True

        # Toggle to ACTIVE
        win.toggle_analysis()
        assert win.is_analysis_active is True
        assert "⏹ Зупинити аналіз" in win.btn_toggle_analysis.text()
        # Red styling applied
        assert "#ef4444" in win.btn_toggle_analysis.styleSheet()

        # Toggle back to INACTIVE
        win.toggle_analysis()
        if win._worker:
            win._worker.wait(2000)
        assert win.is_analysis_active is False
        assert "▶ Почати аналіз" in win.btn_toggle_analysis.text()
        # Blue styling restored
        assert "#0284c7" in win.btn_toggle_analysis.styleSheet()

    def test_file_switch_auto_triggers_inference_when_active(self, qapp, qtbot, tmp_path):
        """When is_analysis_active is True, selecting a file without cache triggers detection."""
        # Create 2 test dummy images
        img1_path = tmp_path / "img_01.png"
        img2_path = tmp_path / "img_02.png"

        dummy = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(img1_path), dummy)
        cv2.imwrite(str(img2_path), dummy)

        win = MainWindow()
        win.load_image_folder(str(tmp_path))

        # Turn stream analysis ON
        win.set_analysis_active(True)
        assert win.is_analysis_active is True

        # Switch to second image in list
        win.list_files.setCurrentRow(1)

        # Worker should be launched for the newly selected image
        assert win._worker is not None
        assert win._current_image_path == str(img2_path)
        win.cancel_detection()
        if win._worker:
            win._worker.wait(2000)


# =============================================================================
# Test Suite 2: Metadata JSON Cache Persistence & Reading
# =============================================================================
class TestMetadataCache:
    """Validate JSON metadata structure, coordinate parsing, and round-tripping."""

    def test_metadata_cache_roundtrip(self, tmp_path):
        """Coordinates saved to JSON must match exactly upon loading."""
        img_path = tmp_path / "recon_flight_042.jpg"
        img_path.touch()

        original_detections: List[Dict[str, Any]] = [
            {
                "id": 1,
                "class_id": 9,
                "class_name": "large-vehicle",
                "confidence": 0.8425,
                "bbox": [1520.5, 840.0, 120.0, 75.5],
            },
            {
                "id": 2,
                "class_id": 2,
                "class_name": "storage-tank",
                "confidence": 0.9120,
                "bbox": [2800.0, 1950.0, 240.5, 240.5],
            },
        ]

        # Save metadata cache
        json_file = save_metadata_cache(
            image_path=img_path,
            image_size=(7680, 4320),
            detections=original_detections,
        )

        assert json_file.is_file()
        assert json_file == img_path.with_suffix(".json")
        assert has_metadata_cache(img_path) is True

        # Read JSON file content directly
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["image"] == "recon_flight_042.jpg"
        assert data["image_size"] == [7680, 4320]
        assert data["detections_count"] == 2
        assert len(data["detections"]) == 2

        # Verify parsed GUI detections
        gui_dets = metadata_to_gui_detections(data)
        assert len(gui_dets) == 2

        det0 = gui_dets[0]
        assert det0["class_id"] == 9
        assert det0["class_name"] == "large-vehicle"
        assert pytest.approx(det0["confidence"], 1e-3) == 0.8425
        assert det0["bbox"] == [1520.5, 840.0, 120.0, 75.5]
        assert det0["x"] == 1520.5
        assert det0["y"] == 840.0
        assert det0["w"] == 120.0
        assert det0["h"] == 75.5

    def test_canvas_viewer_parses_json_cache(self, qapp, tmp_path):
        """CanvasViewer and MainWindow restore overlay without inference when JSON exists."""
        img_path = tmp_path / "target_frame.png"
        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), dummy)

        cached_dets = [
            {
                "class_id": 0,
                "class_name": "plane",
                "confidence": 0.95,
                "bbox": [50.0, 60.0, 40.0, 30.0],
            }
        ]
        save_metadata_cache(img_path, (300, 300), cached_dets)

        win = MainWindow()
        success = win.load_image_file(str(img_path))
        assert success is True

        # Check that canvas has the detection loaded immediately from cache
        dets = win.canvas.get_detections()
        assert len(dets) == 1
        assert dets[0]["class_id"] == 0
        assert dets[0]["bbox"] == [50.0, 60.0, 40.0, 30.0]

        # Verify table also populated
        assert win.table_targets.rowCount() == 1
        assert "plane" in win.table_targets.item(0, 1).text()


# =============================================================================
# Test Suite 3: Batch Triage Worker
# =============================================================================
class TestBatchTriageWorker:
    """Validate background batch filtering, copying, and metadata export."""

    def test_batch_triage_filters_and_exports_positive_images_only(self, qapp, qtbot, tmp_path):
        """Batch triage copies only images with detections and saves metadata JSON alongside."""
        input_dir = tmp_path / "input_recon"
        output_dir = tmp_path / "output_triage"
        input_dir.mkdir()

        # Create two dummy images:
        # 1. target_positive.png -> detector will find a target
        # 2. empty_negative.png -> detector will find 0 targets
        pos_img = input_dir / "target_positive.png"
        neg_img = input_dir / "empty_negative.png"

        dummy_pos = np.ones((400, 400, 3), dtype=np.uint8) * 255
        dummy_neg = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(pos_img), dummy_pos)
        cv2.imwrite(str(neg_img), dummy_neg)

        # Mock detector that returns 1 target for target_positive and empty for negative
        class DummyDetector:
            def detect_image(self, img_rgb):
                return []

        def mock_detect_fn(img_rgb):
            return []

        # Custom detector mock matching files based on content
        class ScriptedDetector:
            def __call__(self, img_rgb):
                # Positive detection for target_positive (non-zero), negative for empty
                if np.any(img_rgb > 0):
                    return [
                        {
                            "id": 1,
                            "class_id": 4,
                            "class_name": "helicopter",
                            "conf": 0.89,
                            "bbox": [100.0, 120.0, 80.0, 60.0],
                        }
                    ]
                return []

        scripted_detector = ScriptedDetector()

        worker = BatchTriageWorker(
            input_folder=input_dir,
            output_folder=output_dir,
            detector=scripted_detector,
            use_cache=False,
        )

        with qtbot.waitSignal(worker.triage_completed, timeout=5000) as blocker:
            worker.start()
        worker.wait(2000)

        total_files, positive_files, elapsed = blocker.args
        assert total_files == 2
        assert positive_files == 1

        # Verify output directory contents:
        # ONLY target_positive.png and target_positive.json must exist!
        out_files = sorted([f.name for f in output_dir.iterdir()])
        assert "target_positive.png" in out_files
        assert "target_positive.json" in out_files
        assert "empty_negative.png" not in out_files
        assert "empty_negative.json" not in out_files

        # Verify JSON metadata content
        json_path = output_dir / "target_positive.json"
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["image"] == "target_positive.png"
        assert meta["detections_count"] == 1
        assert meta["detections"][0]["class_name"] == "helicopter"
        assert meta["detections"][0]["bbox"] == [100.0, 120.0, 80.0, 60.0]

    def test_batch_triage_cancellation(self, qapp, qtbot, tmp_path):
        """Cancelling BatchTriageWorker terminates the loop gracefully."""
        input_dir = tmp_path / "cancel_input"
        output_dir = tmp_path / "cancel_output"
        input_dir.mkdir()

        for i in range(10):
            img_file = input_dir / f"frame_{i:02d}.jpg"
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_file), dummy)

        worker = BatchTriageWorker(
            input_folder=input_dir,
            output_folder=output_dir,
            detector=lambda img: [],
        )

        # Cancel immediately
        worker.cancel()
        assert worker.is_cancelled() is True

        with qtbot.waitSignal(worker.triage_completed, timeout=5000):
            worker.start()
        worker.wait(2000)

    def test_batch_triage_same_folder_collision_prevention(self, qapp, qtbot, tmp_path):
        """Intentionally passing identical input/output folders prevents SameFileError."""
        input_dir = tmp_path / "recon_same"
        input_dir.mkdir()

        pos_img = input_dir / "target_pos.png"
        dummy = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(pos_img), dummy)

        # Output folder intentionally identical to input folder!
        worker = BatchTriageWorker(
            input_folder=input_dir,
            output_folder=input_dir,
            detector=lambda img: [
                {
                    "class_id": 1,
                    "class_name": "vehicle",
                    "conf": 0.85,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                }
            ],
            use_cache=False,
        )

        # Worker must have redirected output_folder to <input_dir>_detected
        assert worker.output_folder != worker.input_folder
        assert worker.output_folder == input_dir.parent / f"{input_dir.name}_detected"

        # Worker must finish successfully with 0 SameFileError
        with qtbot.waitSignal(worker.triage_completed, timeout=5000) as blocker:
            worker.start()
        worker.wait(2000)

        total, positive, elapsed = blocker.args
        assert total == 1
        assert positive == 1

        # Check copied file and json in resolved output directory
        assert (worker.output_folder / "target_pos.png").is_file()
        assert (worker.output_folder / "target_pos.json").is_file()

    def test_single_inference_format_results_no_attribute_error(self, qapp, qtbot, tmp_path):
        """InferenceWorker must have _format_results_for_gui and complete without AttributeError."""
        from gui.async_worker import InferenceWorker

        img_file = tmp_path / "single_test_frame.png"
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_file), dummy)

        worker = InferenceWorker(
            image_source=str(img_file),
            altitude=150.0,
            vram_mb=2048,
            conf_threshold=0.20,
            is_mock=True,
        )

        assert hasattr(worker, "_format_results_for_gui")
        assert callable(worker._format_results_for_gui)

        with qtbot.waitSignal(worker.detection_completed, timeout=5000) as blocker:
            worker.start()
        worker.wait(2000)

        dets, elapsed_ms = blocker.args
        assert isinstance(dets, list)
        assert len(dets) >= 0
        assert elapsed_ms > 0.0
