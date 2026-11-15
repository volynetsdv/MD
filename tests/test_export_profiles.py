"""
Unit tests for train_pipeline.export_models.

Verifies:
1. Presence of all 4 resolution profiles (320, 416, 512, 640) for both .onnx and .engine artifacts.
2. Structural validity of each generated ONNX model via onnx.checker.check_model.
3. Inference with dummy tensor via ONNX Runtime to confirm detection output tensor dimensions:
   (batch=1, 4 + num_classes, num_anchors) where num_anchors is precisely:
     (size // 8)^2 + (size // 16)^2 + (size // 32)^2
"""

import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from train_pipeline.export_models import (
    GRID_RESOLUTIONS,
    HAS_TENSORRT,
    _is_cuda_gpu_available,
    build_trt_engine,
    export_all_profiles,
)


# =============================================================================
# Test Session Fixture: Export All 4 Profiles Once
# =============================================================================
@pytest.fixture(scope="module")
def exported_models_dir(tmp_path_factory) -> Path:
    """
    Generate all 4 resolution profiles [320, 416, 512, 640] in a temporary directory.
    Uses yolo11n architecture for fast and reproducible test runs.
    """
    output_dir = tmp_path_factory.mktemp("exported_yolo_models")

    export_all_profiles(
        weights="yolo11n.yaml",
        output_dir=output_dir,
        sizes=GRID_RESOLUTIONS,
        fp16=True,
        int8=False,
        workspace_mb=512,
        opset=17,
        device="cpu",
    )

    return output_dir


# =============================================================================
# Test 1: Presence of all 4 ONNX profiles (always) and TensorRT engines (GPU only)
# =============================================================================
def test_all_four_profiles_presence(exported_models_dir: Path):
    """Test 1a: Verify presence of all 4 ONNX grid profiles regardless of GPU availability."""
    for sz in GRID_RESOLUTIONS:
        onnx_file = exported_models_dir / f"yolo_{sz}.onnx"
        assert onnx_file.exists(), f"Missing ONNX file: {onnx_file}"
        assert onnx_file.stat().st_size > 1024, f"ONNX file is suspiciously small: {onnx_file}"


@pytest.mark.skipif(
    not (_is_cuda_gpu_available() and HAS_TENSORRT),
    reason="No physical NVIDIA GPU/driver available in CI environment",
)
def test_tensorrt_engines_presence(exported_models_dir: Path):
    """Test 1b: Verify TensorRT .engine artifacts are built when NVIDIA GPU + TRT are present."""
    for sz in GRID_RESOLUTIONS:
        engine_file = exported_models_dir / f"yolo_{sz}.engine"
        assert engine_file.exists(), f"Missing Engine file: {engine_file}"
        assert engine_file.stat().st_size > 0, f"Engine file is empty: {engine_file}"


def test_export_without_gpu_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog):
    """Verify build_trt_engine and export_all_profiles fall back cleanly without CUDA GPU/driver."""
    monkeypatch.setattr("train_pipeline.export_models._is_cuda_gpu_available", lambda: False)

    out_dir = tmp_path / "cpu_only_export"
    with caplog.at_level("WARNING"):
        results = export_all_profiles(
            weights="yolo11n.yaml",
            output_dir=out_dir,
            sizes=(320,),
            fp16=True,
            int8=False,
            workspace_mb=256,
            opset=17,
            device="cpu",
        )

    assert 320 in results
    assert results[320]["onnx"] is not None and results[320]["onnx"].exists()
    assert results[320]["engine"] is None
    assert "skipping TensorRT .engine build" in caplog.text

    if HAS_TENSORRT:
        import train_pipeline.export_models as em

        monkeypatch.setattr("train_pipeline.export_models._is_cuda_gpu_available", lambda: True)

        def _failing_builder(_logger):
            raise TypeError("pybind11::init(): factory function returned nullptr")

        monkeypatch.setattr(em.trt, "Builder", _failing_builder)
        res = build_trt_engine(
            onnx_path=results[320]["onnx"],
            engine_path=out_dir / "yolo_320.engine",
            img_size=320,
        )
        assert res is None


# =============================================================================
# Test 2: Structural validity of ONNX models via onnx.checker.check_model
# =============================================================================
@pytest.mark.parametrize("sz", GRID_RESOLUTIONS)
def test_onnx_structure_validity(exported_models_dir: Path, sz: int):
    """
    Test 2: Validate ONNX graph structure and opset integrity using onnx.checker.check_model.
    """
    onnx_file = exported_models_dir / f"yolo_{sz}.onnx"
    assert onnx_file.exists(), f"ONNX model for size {sz} not found."

    # Load and validate ONNX protobuf
    model_proto = onnx.load(str(onnx_file))
    onnx.checker.check_model(model_proto)

    # Validate graph inputs
    graph = model_proto.graph
    assert len(graph.input) == 1, f"Expected 1 input tensor, got {len(graph.input)}"
    input_tensor = graph.input[0]
    assert input_tensor.name == "images", f"Expected input name 'images', got '{input_tensor.name}'"

    # Validate input dimensions [1, 3, sz, sz] or dynamic batch ['batch', 3, sz, sz]
    dim_values = []
    for d in input_tensor.type.tensor_type.shape.dim:
        dim_values.append(
            d.dim_value
            if d.HasField("dim_value")
            else (d.dim_param if d.HasField("dim_param") else None)
        )

    assert dim_values in (
        [1, 3, sz, sz],
        ["batch", 3, sz, sz],
    ), f"Input shape mismatch: expected [1, 3, {sz}, {sz}] or dynamic batch, got {dim_values}"


# =============================================================================
# Test 3: Dummy-tensor inference via ONNX Runtime and output shape verification
# =============================================================================
@pytest.mark.parametrize("sz", GRID_RESOLUTIONS)
def test_onnxruntime_dummy_tensor_inference(exported_models_dir: Path, sz: int):
    """
    Test 3: Run dummy float32 tensor through ONNX Runtime to confirm detection output shapes.
    Expected output tensor shape: (1, 84, num_anchors)
    where:
      - 84 = 4 bounding box coordinates (xc, yc, w, h) + 80 class confidence scores.
      - num_anchors = (sz // 8)^2 + (sz // 16)^2 + (sz // 32)^2 (P3, P4, P5 multi-scale heads).
    """
    onnx_file = exported_models_dir / f"yolo_{sz}.onnx"
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

    input_meta = session.get_inputs()[0]
    assert input_meta.shape in (
        [1, 3, sz, sz],
        ["batch", 3, sz, sz],
    ), (
        f"Session input shape mismatch for {sz}: expected [1, 3, {sz}, {sz}] or dynamic batch, got {input_meta.shape}"
    )

    # Compute expected number of anchor detection candidates
    p3 = (sz // 8) ** 2
    p4 = (sz // 16) ** 2
    p5 = (sz // 32) ** 2
    expected_anchors = p3 + p4 + p5

    # Run inference with dummy float32 tensor
    dummy_input = np.zeros((1, 3, sz, sz), dtype=np.float32)
    outputs = session.run(None, {input_meta.name: dummy_input})

    assert len(outputs) >= 1, "ONNX Runtime returned empty output list"
    out_tensor = outputs[0]

    assert out_tensor.dtype == np.float32, f"Expected float32 output, got {out_tensor.dtype}"
    assert out_tensor.ndim == 3, f"Expected 3D output tensor [B, C, N], got shape {out_tensor.shape}"

    batch_size, num_channels, num_anchors = out_tensor.shape
    assert batch_size == 1, f"Expected batch size 1, got {batch_size}"
    assert num_channels == 84, f"Expected 84 output channels (4 coords + 80 classes), got {num_channels}"
    assert num_anchors == expected_anchors, (
        f"For resolution {sz}x{sz}: expected {expected_anchors} anchors, got {num_anchors}"
    )

