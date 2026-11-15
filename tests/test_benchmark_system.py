"""Unit and Integration Tests for Scientific Benchmark System."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.benchmark_system import (
    SystemResourceMonitor,
    YOLO_TRAINING_BASELINE_METRICS,
    box_iou,
    calculate_baseline_640_tiles_count,
    compute_ap_101,
    compute_pcie_metrics,
    compute_stats,
    evaluate_detection_accuracy,
    find_default_gt_labels_dir,
    generate_synthetic_aerial_frame,
    get_system_hardware_info,
    load_ground_truth,
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


def test_box_iou():
    # Box 1: [0, 0, 10, 10] (area 100)
    # Box 2: [0, 0, 10, 10] (identical, IoU=1.0)
    # Box 3: [5, 0, 15, 10] (half overlap, inter=50, union=150, IoU=1/3)
    # Box 4: [20, 20, 30, 30] (disjoint, IoU=0.0)
    b1 = np.array([[0.0, 0.0, 10.0, 10.0]])
    b2 = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [5.0, 0.0, 15.0, 10.0],
        [20.0, 20.0, 30.0, 30.0],
    ])
    ious = box_iou(b1, b2)
    assert ious.shape == (1, 3)
    assert pytest.approx(ious[0, 0], 1e-4) == 1.0
    assert pytest.approx(ious[0, 1], 1e-4) == 50.0 / 150.0
    assert pytest.approx(ious[0, 2], 1e-4) == 0.0

    # Empty edge case
    empty_ious = box_iou(np.empty((0, 4)), b2)
    assert empty_ious.shape == (0, 3)


def test_compute_ap_101():
    # Perfect detection
    rec_perf = np.array([0.5, 1.0])
    prec_perf = np.array([1.0, 1.0])
    ap_perf = compute_ap_101(rec_perf, prec_perf)
    assert ap_perf > 0.99

    # Empty curve
    assert compute_ap_101(np.empty(0), np.empty(0)) == 0.0

    # Zero recall
    rec_zero = np.array([0.0])
    prec_zero = np.array([0.0])
    assert compute_ap_101(rec_zero, prec_zero) == 0.0


def test_find_default_gt_labels_dir():
    # Real dataset path
    val_images = repo_root / "data" / "dota_v1.5" / "images" / "val"
    if val_images.exists():
        found = find_default_gt_labels_dir(val_images)
        assert found is not None
        assert found.is_dir()
        assert found.name == "val"
        assert found.parent.name == "labels"

    # None path
    assert find_default_gt_labels_dir(None) is None


def test_load_ground_truth():
    gt_dir = repo_root / "data" / "dota_v1.5" / "labels" / "val"
    if gt_dir.exists():
        gts = load_ground_truth(gt_dir, "P0003.jpg", 1147, 1023)
        assert isinstance(gts, list)
        assert len(gts) == 55
        sample = gts[0]
        assert "class_id" in sample
        assert "bbox" in sample
        assert len(sample["bbox"]) == 4
        x1, y1, x2, y2 = sample["bbox"]
        assert 0.0 <= x1 < x2 <= 1147.0
        assert 0.0 <= y1 < y2 <= 1023.0


def test_evaluate_detection_accuracy():
    class_names = {0: "plane", 1: "ship"}
    # Synthetic ground truth
    gt_map = {
        "img1": [
            {"class_id": 0, "bbox": [10.0, 10.0, 50.0, 50.0]},
            {"class_id": 1, "bbox": [100.0, 100.0, 200.0, 200.0]},
        ],
        "img2": [
            {"class_id": 0, "bbox": [30.0, 30.0, 70.0, 70.0]},
        ],
    }
    # Synthetic predictions (perfect match for plane on img1 and img2, false positive for ship)
    pred_map = {
        "img1": [
            {"class_id": 0, "conf": 0.95, "bbox": [10.0, 10.0, 40.0, 40.0]},  # [10, 10, 50, 50]
            {"class_id": 1, "conf": 0.30, "bbox": [500.0, 500.0, 20.0, 20.0]},  # False positive
        ],
        "img2": [
            {"class_id": 0, "conf": 0.88, "bbox": [30.0, 30.0, 40.0, 40.0]},  # [30, 30, 70, 70]
        ],
    }

    acc = evaluate_detection_accuracy(pred_map, gt_map, class_names, primary_iou_thresh=0.50)
    assert acc["total_ground_truth"] == 3
    assert acc["total_predictions"] == 3
    assert acc["total_true_positives"] == 2
    assert acc["total_false_positives"] == 1
    assert pytest.approx(acc["precision"], 1e-2) == 2.0 / 3.0
    assert pytest.approx(acc["recall"], 1e-2) == 2.0 / 3.0
    assert "plane" in acc["per_class"]
    assert "ship" in acc["per_class"]
    assert acc["per_class"]["plane"]["ap50"] > 0.90
    assert acc["per_class"]["ship"]["recall"] == 0.0


def test_batch_accuracy_benchmark(tmp_path: Path):
    dota_images = repo_root / "data" / "dota_v1.5" / "images" / "val"
    dota_labels = repo_root / "data" / "dota_v1.5" / "labels" / "val"
    if not dota_images.exists() or not dota_labels.exists():
        pytest.skip("DOTA validation set not found")

    out_dir = tmp_path / "bench_acc_out"
    results, summary_md = run_benchmark_suite(
        input_path=dota_images,
        gt_labels_dir=dota_labels,
        eval_accuracy=True,
        max_images=2,
        output_dir=out_dir,
        warmup=0,
    )

    assert (out_dir / "benchmark_results.json").exists()
    assert (out_dir / "benchmark_summary.md").exists()
    assert "accuracy_metrics" in results
    acc = results["accuracy_metrics"]
    assert acc is not None
    assert "precision" in acc
    assert "recall" in acc
    assert "map50" in acc
    assert "map50_95" in acc
    assert acc["total_ground_truth"] > 0
    assert "End-to-End Accuracy" in summary_md

