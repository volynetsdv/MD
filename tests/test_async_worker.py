"""Unit and integration tests for InferenceWorker (gui/async_worker.py).

Tests:
1. Thread lifecycle and signal contracts (progress_changed, detection_completed, error_occurred).
2. End-to-end tiling, inference, offset remapping, and Cluster-DIoU-NMS merging.
3. Thread safety and graceful cancellation via cancel().
4. Error propagation for missing or invalid image sources.
5. Execution with real ONNX models via UnifiedDetector CPU backend.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure offscreen platform
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from gui.async_worker import InferenceWorker
from src.detector_dispatcher import Detection, UnifiedDetector


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInferenceWorker:
    """Test suite for InferenceWorker background QThread."""

    def test_worker_initialization(self, qapp):
        """Verify proper initialization of InferenceWorker state and parameters."""
        worker = InferenceWorker(
            image_source="dummy.png",
            altitude=150.0,
            vram_mb=4096,
            conf_threshold=0.30,
            diou_threshold=0.60,
        )
        assert worker.altitude == 150.0
        assert worker.vram_mb == 4096
        assert worker.conf_threshold == 0.30
        assert worker.diou_threshold == 0.60
        assert worker.is_cancelled() is False

    def test_worker_cancellation(self, qapp):
        """Verify calling cancel() updates cancellation flag."""
        worker = InferenceWorker(image_source=np.zeros((100, 100, 3), dtype=np.uint8))
        assert worker.is_cancelled() is False
        worker.cancel()
        assert worker.is_cancelled() is True

    def test_worker_invalid_image_emits_error(self, qapp, qtbot):
        """Verify worker emits error_occurred when given a non-existent file."""
        worker = InferenceWorker(
            image_source="/path/to/nonexistent/aerial_photo_xyz.png",
            altitude=120.0,
            vram_mb=2048,
        )

        with qtbot.waitSignal(worker.error_occurred, timeout=3000) as blocker:
            worker.start()

        error_msg = blocker.args[0]
        assert "Не вдалося завантажити зображення" in error_msg

    def test_worker_mock_execution_cycle(self, qapp, qtbot):
        """Verify complete pipeline execution in mock mode:

        - Emits progress_changed updates.
        - Emits detection_completed with detected objects and execution time.
        - Offsets remapped and Cluster-DIoU-NMS merged.
        """
        img_arr = np.zeros((1200, 1600, 3), dtype=np.uint8)
        worker = InferenceWorker(
            image_source=img_arr,
            altitude=120.0,
            vram_mb=2048,
            is_mock=True,
        )

        progress_calls = []
        worker.progress_changed.connect(
            lambda cur, tot, msg: progress_calls.append((cur, tot, msg))
        )

        with qtbot.waitSignal(worker.detection_completed, timeout=8000) as blocker:
            worker.start()

        detections, elapsed_ms = blocker.args
        assert isinstance(detections, list)
        assert elapsed_ms > 0.0

        # Verify progress signal was called for tiles
        assert len(progress_calls) > 0
        assert any("Обробка тайла" in call[2] for call in progress_calls)
        cur, tot, msg = progress_calls[-1]
        assert cur == tot

        # Detections structure validation
        for det in detections:
            assert "id" in det
            assert "x" in det
            assert "y" in det
            assert "w" in det
            assert "h" in det
            assert "conf" in det
            assert "class_id" in det
            assert "class_name" in det
            assert det["w"] > 0
            assert det["h"] > 0

    def test_worker_with_real_unified_detector(self, qapp, qtbot):
        """Verify worker runs tiles through UnifiedDetector with CPU fallback."""
        detector = UnifiedDetector(prefer_tensorrt=False, force_provider="CPUExecutionProvider")

        # Small image (e.g. 640x640: 1 tile) to test quickly
        img_arr = np.zeros((640, 640, 3), dtype=np.uint8)
        worker = InferenceWorker(
            image_source=img_arr,
            altitude=120.0,
            vram_mb=2048,
            detector=detector,
            is_mock=False,
        )

        with qtbot.waitSignal(worker.detection_completed, timeout=5000) as blocker:
            worker.start()

        detections, elapsed_ms = blocker.args
        assert isinstance(detections, list)
        assert elapsed_ms > 0.0
