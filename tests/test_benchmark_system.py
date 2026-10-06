"""Unit and Integration Tests for Scientific Benchmark System."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.benchmark_system import (
    SystemResourceMonitor,
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


def test_system_resource_monitor_lifecycle():
    monitor = SystemResourceMonitor(interval_sec=0.05)
    monitor.record_baseline()
    assert monitor.baseline_rss_mb > 0.0

    monitor.start()
    time.sleep(0.15)
    monitor.stop()

    assert monitor.final_rss_mb > 0.0
    assert len(monitor.rss_ram_samples) >= 1

    summary = monitor.get_summary()
    assert "baseline" in summary
    assert "final" in summary
    assert "delta" in summary
    assert "memory_leak_check" in summary
    assert summary["memory_leak_check"]["status"] == "PASS [OK]"


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
    assert "system_resource_telemetry" in results
    assert len(summary_md) > 100


def test_batch_directory_benchmark(tmp_path: Path):
    in_dir = tmp_path / "test_images"
    detected_dir = tmp_path / "test_detected"
    out_dir = tmp_path / "bench_reports"

    in_dir.mkdir()
    generate_synthetic_aerial_frame(640, 640, in_dir / "frame1.jpg")
    generate_synthetic_aerial_frame(640, 640, in_dir / "frame2.jpg")

    results, summary_md = run_benchmark_suite(
        input_path=in_dir,
        output_detected_dir=detected_dir,
        output_dir=out_dir,
        altitude=150.0,
        vram_mb=2048,
        conf_thresh=0.20,
        diou_thresh=0.50,
        warmup=0,
    )

    assert (out_dir / "benchmark_results.json").exists()
    assert (out_dir / "benchmark_summary.md").exists()
    assert results.get("batch_summary") is not None
    assert results["batch_summary"]["total_images"] == 2
    assert "system_resource_telemetry" in results

    # Check if positive detections were replicated with companion JSON
    if results["batch_summary"]["positive_images"] > 0:
        assert detected_dir.exists()
        copied_files = list(detected_dir.glob("*.jpg"))
        json_files = list(detected_dir.glob("*.json"))
        assert len(copied_files) == results["batch_summary"]["positive_images"]
        assert len(json_files) == len(copied_files)
