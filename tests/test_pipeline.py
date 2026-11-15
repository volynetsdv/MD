"""Unit and Integration Tests for Dynamic Batching, OOM Safety, and Detection Equivalence."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List
import cv2
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.config_manager import AppConfig, get_config
from gui.async_worker import run_tiled_inference
from scripts.benchmark_system import box_iou, run_benchmark_suite
from src.detector_dispatcher import UnifiedDetector, calculate_safe_batch_size


def test_calculate_safe_batch_size_standard_mode():
    """is_turbo=False must return strictly 1 regardless of available memory."""
    assert calculate_safe_batch_size(target_tile_size=640, is_turbo=False, max_cap=4) == 1
    assert calculate_safe_batch_size(target_tile_size=320, is_turbo=False, max_cap=8) == 1
    assert calculate_safe_batch_size(target_tile_size=736, is_turbo=False, max_cap=4) == 1


def test_calculate_safe_batch_size_oom_guard(monkeypatch: pytest.MonkeyPatch):
    """When free VRAM is critically low (e.g. 100 MB), formula must return 1 (OOM Guard)."""
    monkeypatch.setattr(UnifiedDetector, "query_vram_mb", staticmethod(lambda: (100.0, 16384.0)))

    batch_sz = calculate_safe_batch_size(target_tile_size=640, is_turbo=True, max_cap=4)
    assert batch_sz == 1


def test_calculate_safe_batch_size_large_memory(monkeypatch: pytest.MonkeyPatch):
    """When free VRAM is abundant (e.g. 8000 MB), formula must return max_cap."""
    monkeypatch.setattr(UnifiedDetector, "query_vram_mb", staticmethod(lambda: (8000.0, 16384.0)))

    batch_sz = calculate_safe_batch_size(target_tile_size=640, is_turbo=True, max_cap=4)
    assert batch_sz == 4

    batch_sz_8 = calculate_safe_batch_size(target_tile_size=320, is_turbo=True, max_cap=8)
    assert batch_sz_8 == 8


def test_calculate_safe_batch_size_dynamic_scaling(monkeypatch: pytest.MonkeyPatch):
    """Tile size quadratic scaling: Delta(320) consumes 1/4 of Delta(640)."""
    # Suppose free memory is 250 MB.
    # safety_factor=0.80 -> 200 MB safe budget.
    # For 640: cost = 110.0 MB -> 200 // 110 = 1.
    # For 320: cost = 110.0 * (320/640)^2 = 27.5 MB -> 200 // 27.5 = 7.
    monkeypatch.setattr(UnifiedDetector, "query_vram_mb", staticmethod(lambda: (250.0, 16384.0)))

    b_640 = calculate_safe_batch_size(target_tile_size=640, is_turbo=True, max_cap=4)
    b_320 = calculate_safe_batch_size(target_tile_size=320, is_turbo=True, max_cap=8)

    assert b_640 == 1
    assert b_320 == 7


def test_calculate_safe_batch_size_unavailable_vram_fallbacks(monkeypatch: pytest.MonkeyPatch):
    """When VRAM cannot be queried, CPU falls back to 1 and DirectML falls back to 2."""
    monkeypatch.setattr(UnifiedDetector, "query_vram_mb", staticmethod(lambda: (0.0, 0.0)))

    assert calculate_safe_batch_size(target_tile_size=640, is_turbo=True) == 1
    assert (
        calculate_safe_batch_size(
            target_tile_size=640, is_turbo=True, execution_provider="DmlExecutionProvider"
        )
        == 2
    )


def test_predict_batch_structure():
    """UnifiedDetector.predict_batch must return a list of detections per slice."""
    detector = UnifiedDetector()
    tile1 = np.zeros((640, 640, 3), dtype=np.uint8)
    tile2 = np.zeros((640, 640, 3), dtype=np.uint8)

    batch_dets = detector.predict_batch([tile1, tile2], tile_size=640, conf_threshold=0.20)
    assert isinstance(batch_dets, list)
    assert len(batch_dets) == 2
    for dets in batch_dets:
        assert isinstance(dets, list)


def test_standard_turbo_detection_equivalence():
    """Verify that Standard (B=1) and Turbo (Dynamic B) yield equivalent detections."""
    test_img_path = repo_root / "data" / "dota_v1.5" / "images" / "val" / "P0003.jpg"
    if not test_img_path.exists():
        pytest.skip("Test image data/dota_v1.5/images/val/P0003.jpg not available")

    img_bgr = cv2.imread(str(test_img_path))
    assert img_bgr is not None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detector = UnifiedDetector()

    # Pass 1: Standard Mode (B = 1)
    dets_std = run_tiled_inference(
        img_rgb=img_rgb,
        altitude=150.0,
        vram_mb=2048,
        conf_threshold=0.20,
        diou_threshold=0.50,
        detector=detector,
        is_turbo=False,
    )

    # Pass 2: Turbo Mode (Dynamic Max Batch)
    dets_turbo = run_tiled_inference(
        img_rgb=img_rgb,
        altitude=150.0,
        vram_mb=2048,
        conf_threshold=0.20,
        diou_threshold=0.50,
        detector=detector,
        is_turbo=True,
        turbo_max_cap=4,
    )

    # Equivalence assertions: 0 dropped detections
    assert len(dets_std) == len(dets_turbo)
    assert len(dets_std) > 0

    # Pairwise comparison
    for d1, d2 in zip(dets_std, dets_turbo):
        assert d1["class_id"] == d2["class_id"]
        assert pytest.approx(d1["confidence"], abs=1e-3) == d2["confidence"]

        # Convert [x, y, w, h] to [x1, y1, x2, y2] for IoU verification
        b1 = np.array([[d1["x"], d1["y"], d1["x"] + d1["w"], d1["y"] + d1["h"]]])
        b2 = np.array([[d2["x"], d2["y"], d2["x"] + d2["w"], d2["y"] + d2["h"]]])
        iou = float(box_iou(b1, b2)[0, 0])
        assert iou >= 0.999


def test_benchmark_double_pass_and_report_generation(tmp_path: Path):
    """Test full double-pass benchmark execution and verify Markdown report contents."""
    out_dir = tmp_path / "bench_double_pass"
    results, summary_md = run_benchmark_suite(
        samples=1,
        output_dir=out_dir,
        altitude=150.0,
        vram_mb=2048,
        conf_thresh=0.20,
        diou_thresh=0.50,
        leak_check_iterations=0,
        warmup=0,
        custom_resolutions=["640x640"],
        mode="both",
        turbo_max_cap=4,
    )

    assert (out_dir / "benchmark_results.json").exists()
    assert (out_dir / "benchmark_summary.md").exists()

    assert "standard_mode" in results and results["standard_mode"] is not None
    assert "turbo_mode" in results and results["turbo_mode"] is not None
    assert "comparison" in results and results["comparison"] is not None

    comp = results["comparison"]
    assert "mean_batch_size" in comp
    assert "throughput_fps" in comp
    assert "peak_vram_mb" in comp

    # Verify report structure
    assert "## Порівняння режимів продуктивності (Standard vs Turbo Mode)" in summary_md
    assert "## Підсумковий аналіз ефективності системи" in summary_md
    assert "Середній розмір батчу (B)" in summary_md

    # Verify absence of dissertation/thesis terminology
    assert "дисертац" not in summary_md.lower()
    assert "розділ 3" not in summary_md.lower()
    assert "магістерськ" not in summary_md.lower()
