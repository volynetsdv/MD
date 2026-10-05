"""Unit and Integration Tests for Scientific Benchmark System."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.benchmark_system import (
    calculate_baseline_640_tiles_count,
    compute_pcie_metrics,
    compute_stats,
    generate_synthetic_aerial_frame,
    get_system_hardware_info,
    run_benchmark_suite,
)


def test_system_hardware_discovery():
    hw = get_system_hardware_info()
    assert isinstance(hw, dict)
    assert "os" in hw
    assert "cpu_count_logical" in hw
    assert hw["cpu_count_logical"] >= 1
    assert "total_ram_gb" in hw
    assert hw["total_ram_gb"] > 0


def test_compute_stats():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    st = compute_stats(vals)
    assert st["mean"] == 30.0
    assert st["min"] == 10.0
    assert st["max"] == 50.0
    assert st["p50"] == 30.0
    assert st["p90"] == 46.0
    assert st["p99"] == 49.6


def test_baseline_640_tiles_count():
    # 4K (3840x2160)
    cnt_4k = calculate_baseline_640_tiles_count(3840, 2160, 150.0)
    assert cnt_4k == 60

    # 8K (7680x4320)
    cnt_8k = calculate_baseline_640_tiles_count(7680, 4320, 150.0)
    assert cnt_8k == 240

    # 13K (13000x6500)
    cnt_13k = calculate_baseline_640_tiles_count(13000, 6500, 150.0)
    assert cnt_13k == 578


def test_pcie_metrics_calculation():
    # 8K (7680x4320)
    metrics = compute_pcie_metrics(7680, 4320, 50)
    assert metrics["frame_raster_mb"] > 90.0
    assert metrics["h2d_stream_mb"] == metrics["frame_raster_mb"]
    assert metrics["d2h_zero_readback_kb"] < 5.0
    assert metrics["readback_reduction_percent"] > 99.99
    assert metrics["total_bus_reduction_percent"] >= 49.9


def test_synthetic_frame_generator(tmp_path: Path):
    out_file = tmp_path / "test_synth.jpg"
    generate_synthetic_aerial_frame(640, 640, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_fast_benchmark_run(tmp_path: Path):
    out_dir = tmp_path / "bench_out"
    results, summary_md = run_benchmark_suite(
        samples=1,
        output_dir=out_dir,
        altitude=150.0,
        vram_mb=2048,
        conf_thresh=0.20,
        diou_thresh=0.50,
        leak_check_iterations=2,
        warmup=0,
        custom_resolutions=["640x640"],
    )
    assert (out_dir / "benchmark_results.json").exists()
    assert (out_dir / "benchmark_summary.md").exists()
    assert "benchmarks" in results
    assert "640x640" in results["benchmarks"]
    assert "memory_leak_audit" in results
    assert len(summary_md) > 100
