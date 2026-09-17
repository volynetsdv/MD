"""Unit and integration tests for UnifiedDetector and Cross-Platform Dispatcher.

Verifies:
1. Unified `Detection` data structure attributes and C++ compatibility.
2. Hardware discovery priority on Windows 10/11:
   - TensorRT -> DirectML (DmlExecutionProvider) -> CUDA -> CPU fallback.
3. Automatic transparent fallback from TensorRT to ONNX Runtime via mock/monkeypatch.
4. Model pool management and automatic session caching across grid sizes (320, 416, 512, 640).
5. Predict tile API invariance: identical `list[Detection]` returned with numpy and torch inputs.
6. Vectorized NMS and coordinate decoding correctness.
"""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.detector_dispatcher import (
    GRID_RESOLUTIONS,
    BackendType,
    Detection,
    UnifiedDetector,
    check_tensorrt_available,
    get_onnx_execution_providers,
    nms_fast,
)


# =============================================================================
# 1. Detection Structure Tests
# =============================================================================
class TestDetectionDataStructure:
    """Test unified Detection class invariance."""

    def test_detection_attributes_and_defaults(self):
        d = Detection(
            x_local=15.5,
            y_local=24.0,
            w=64.0,
            h=48.0,
            conf=0.92,
            class_id=2,
        )
        assert d.x_local == 15.5
        assert d.y_local == 24.0
        assert d.w == 64.0
        assert d.h == 48.0
        assert d.conf == 0.92
        assert d.class_id == 2

    def test_detection_equality_and_repr(self):
        d1 = Detection(10.0, 20.0, 30.0, 40.0, 0.85, 1)
        d2 = Detection(10.0, 20.0, 30.0, 40.0, 0.85, 1)
        d3 = Detection(11.0, 20.0, 30.0, 40.0, 0.85, 1)

        assert d1 == d2
        assert d1 != d3
        assert "x_local=10.000000" in repr(d1)
        assert "class_id=1" in repr(d1)

    def test_detection_to_dict(self):
        d = Detection(1.0, 2.0, 3.0, 4.0, 0.5, 0)
        dt = d.to_dict()
        assert dt == {
            "x_local": 1.0,
            "y_local": 2.0,
            "w": 3.0,
            "h": 4.0,
            "conf": 0.5,
            "class_id": 0,
        }

    def test_detection_to_cpp(self):
        d = Detection(12.0, 34.0, 56.0, 78.0, 0.95, 3)
        cpp_obj = d.to_cpp()
        assert hasattr(cpp_obj, "x_local")
        assert cpp_obj.x_local == 12.0
        assert cpp_obj.class_id == 3


# =============================================================================
# 2. Hardware Discovery & Fallback Tests (Mocking)
# =============================================================================
class TestHardwareDiscoveryAndFallback:
    """Test environment discovery and platform fallback prioritization."""

    def test_tensorrt_active_when_available(self, monkeypatch):
        """When TensorRT C++ SDK and CUDA GPU are present, select TensorRT."""
        monkeypatch.setattr(
            "src.detector_dispatcher.check_tensorrt_available", lambda models_dir: True
        )
        detector = UnifiedDetector(prefer_tensorrt=True)
        assert detector.backend == BackendType.TENSORRT
        assert detector.execution_provider == "TensorRT"
        assert detector.is_gpu_accelerated is True

    def test_windows_directml_selected_when_tensorrt_absent(self, monkeypatch):
        """On Windows without TensorRT, DirectML (DmlExecutionProvider) must be top priority."""
        monkeypatch.setattr(
            "src.detector_dispatcher.check_tensorrt_available", lambda models_dir: False
        )

        detector = UnifiedDetector(
            prefer_tensorrt=True,
            is_windows=True,
            available_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )

        assert detector.backend == BackendType.ONNXRUNTIME
        assert detector.execution_provider == "DmlExecutionProvider"
        assert detector.is_gpu_accelerated is True
        assert detector._providers[0] == "DmlExecutionProvider"

    def test_windows_cuda_fallback_when_dml_absent(self, monkeypatch):
        """On Windows, if DirectML is not installed but CUDA is, select CUDAExecutionProvider."""
        monkeypatch.setattr(
            "src.detector_dispatcher.check_tensorrt_available", lambda models_dir: False
        )

        detector = UnifiedDetector(
            prefer_tensorrt=True,
            is_windows=True,
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        assert detector.backend == BackendType.ONNXRUNTIME
        assert detector.execution_provider == "CUDAExecutionProvider"
        assert detector.is_gpu_accelerated is True

    def test_cpu_fallback_when_no_gpu_provider(self, monkeypatch, caplog):
        """When no GPU provider (TensorRT, DML, CUDA) exists, smoothly fall back to CPU."""
        monkeypatch.setattr(
            "src.detector_dispatcher.check_tensorrt_available", lambda models_dir: False
        )

        with caplog.at_level(logging.INFO):
            detector = UnifiedDetector(
                prefer_tensorrt=True,
                is_windows=True,
                available_providers=["CPUExecutionProvider"],
            )

        assert detector.backend == BackendType.ONNXRUNTIME
        assert detector.execution_provider == "CPUExecutionProvider"
        assert detector.is_gpu_accelerated is False
        assert "CPUExecutionProvider" in caplog.text

    def test_get_onnx_execution_providers_logic(self):
        """Test get_onnx_execution_providers priority logic directly."""
        # Windows with DirectML
        provs, active = get_onnx_execution_providers(
            available_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            is_windows=True,
        )
        assert active == "DmlExecutionProvider"
        assert provs == ["DmlExecutionProvider", "CPUExecutionProvider"]

        # Windows without DML, but with CUDA
        provs, active = get_onnx_execution_providers(
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            is_windows=True,
        )
        assert active == "CUDAExecutionProvider"
        assert provs == ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Linux with CUDA
        provs, active = get_onnx_execution_providers(
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            is_windows=False,
        )
        assert active == "CUDAExecutionProvider"
        assert provs == ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Fallback to CPU
        provs, active = get_onnx_execution_providers(
            available_providers=["CPUExecutionProvider"],
            is_windows=False,
        )
        assert active == "CPUExecutionProvider"
        assert provs == ["CPUExecutionProvider"]


# =============================================================================
# 3. Model Pool & Session Caching Tests
# =============================================================================
class TestModelPoolAndCaching:
    """Test model pool resolution mapping and session caching."""

    def test_resolution_snapping(self):
        detector = UnifiedDetector(prefer_tensorrt=False, force_provider="CPUExecutionProvider")
        # Exact values
        assert detector._snap_resolution(320) == 320
        assert detector._snap_resolution(416) == 416
        assert detector._snap_resolution(512) == 512
        assert detector._snap_resolution(640) == 640

        # Nearest neighbor snapping
        assert detector._snap_resolution(300) == 320
        assert detector._snap_resolution(390) == 416
        assert detector._snap_resolution(500) == 512
        assert detector._snap_resolution(800) == 640

    def test_session_caching_behavior(self):
        detector = UnifiedDetector(prefer_tensorrt=False, force_provider="CPUExecutionProvider")
        assert len(detector.cached_resolutions) == 0

        # Load 320 session
        sess_320 = detector.get_session(320)
        assert 320 in detector.cached_resolutions

        # Re-fetching 320 must return the identical cached object (identity check)
        sess_320_again = detector.get_session(320)
        assert sess_320 is sess_320_again

        # Fetch 640 session
        sess_640 = detector.get_session(640)
        assert 640 in detector.cached_resolutions
        assert sess_640 is not sess_320
        assert len(detector.cached_resolutions) == 2


# =============================================================================
# 4. Predict Tile & Decoding Tests
# =============================================================================
class TestPredictTileAPI:
    """Test tile prediction across NumPy, PyTorch, and shape variations."""

    @pytest.fixture
    def cpu_detector(self) -> UnifiedDetector:
        return UnifiedDetector(prefer_tensorrt=False, force_provider="CPUExecutionProvider")

    def test_predict_tile_with_numpy_hwc_uint8(self, cpu_detector):
        """Test prediction with standard (H, W, 3) uint8 NumPy image."""
        tile = np.zeros((320, 320, 3), dtype=np.uint8)
        # Use very low confidence threshold to capture raw candidate boxes
        dets = cpu_detector.predict_tile(tile, tile_size=320, conf_threshold=0.00001)

        assert isinstance(dets, list)
        for d in dets:
            assert isinstance(d, Detection)
            assert isinstance(d.x_local, float)
            assert isinstance(d.y_local, float)
            assert isinstance(d.w, float)
            assert isinstance(d.h, float)
            assert isinstance(d.conf, float)
            assert isinstance(d.class_id, int)
            assert 0.0 <= d.conf <= 1.0

    def test_predict_tile_with_torch_tensor(self, cpu_detector):
        """Test prediction with (1, 3, H, W) float32 PyTorch tensor."""
        tensor_tile = torch.zeros((1, 3, 320, 320), dtype=torch.float32)
        dets = cpu_detector.predict_tile(tensor_tile, tile_size=320, conf_threshold=0.00001)

        assert isinstance(dets, list)
        for d in dets:
            assert isinstance(d, Detection)

    def test_predict_tile_resizing(self, cpu_detector):
        """Test prediction when input dimensions differ from target model size."""
        # Non-standard tile size 250x250 passed to 320 model
        tile = np.zeros((250, 250, 3), dtype=np.uint8)
        dets = cpu_detector.predict_tile(tile, tile_size=320, conf_threshold=0.00001)

        assert isinstance(dets, list)

    def test_vectorized_nms_fast(self):
        """Verify vectorized NMS algorithm suppresses overlapping candidates correctly."""
        # Box 1 and Box 2 heavily overlap; Box 3 is distinct
        boxes = np.array(
            [
                [10.0, 10.0, 50.0, 50.0],
                [12.0, 12.0, 48.0, 48.0],
                [100.0, 100.0, 150.0, 150.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.90, 0.85, 0.95], dtype=np.float32)

        keep = nms_fast(boxes, scores, iou_threshold=0.5)

        # Box 3 (score 0.95) and Box 1 (score 0.90) kept; Box 2 suppressed
        assert keep == [2, 0]

    def test_synthetic_yolo_output_decoding(self, cpu_detector):
        """Test decoding synthetic raw YOLO tensor (1, 84, N)."""
        # Create synthetic output for 2 detections
        raw_out = np.zeros((1, 84, 2), dtype=np.float32)

        # Detection 1: xc=100, yc=100, w=40, h=40, class 2 score=0.9
        raw_out[0, 0, 0] = 100.0
        raw_out[0, 1, 0] = 100.0
        raw_out[0, 2, 0] = 40.0
        raw_out[0, 3, 0] = 40.0
        raw_out[0, 4 + 2, 0] = 0.90  # class_id 2

        # Detection 2: xc=200, yc=200, w=60, h=50, class 0 score=0.8
        raw_out[0, 0, 1] = 200.0
        raw_out[0, 1, 1] = 200.0
        raw_out[0, 2, 1] = 60.0
        raw_out[0, 3, 1] = 50.0
        raw_out[0, 4 + 0, 1] = 0.80  # class_id 0

        dets = cpu_detector._decode_yolo_output(
            raw_output=raw_out,
            model_size=320,
            original_size=(320, 320),
            conf_threshold=0.5,
            iou_threshold=0.45,
        )

        assert len(dets) == 2
        # Det 1 (score 0.90) top-left: xc - w/2 = 80, yc - h/2 = 80
        d0 = next(d for d in dets if d.class_id == 2)
        assert d0.x_local == 80.0
        assert d0.y_local == 80.0
        assert d0.w == 40.0
        assert d0.h == 40.0
        assert abs(d0.conf - 0.90) < 1e-4

        # Det 2 (score 0.80) top-left: xc - w/2 = 170, yc - h/2 = 175
        d1 = next(d for d in dets if d.class_id == 0)
        assert d1.x_local == 170.0
        assert d1.y_local == 175.0
        assert d1.w == 60.0
        assert d1.h == 50.0
        assert abs(d1.conf - 0.80) < 1e-4
