#!/usr/bin/env python3
"""Autonomous Diagnostic Harness for DOTA Batch Triage Profiling.

Monitors every frame processing stage:
- t_read (cv2.imread, cv2.cvtColor)
- num_tiles & tile planning
- t_infer (raw tile forward passes & remapping)
- t_nms (boundary NMS merge)
- t_io_write (file copy & JSON metadata cache write)
- Process RSS RAM & GPU VRAM
- Critical Stall Alert triggered whenever t_total > 1.5s
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.metadata_cache import (
    has_metadata_cache,
    load_metadata_cache,
    metadata_to_gui_detections,
    save_metadata_cache,
)
from gui.async_worker import (
    calculate_tiles_for_image,
    format_detections_for_gui,
    merge_boundary_detections,
    remap_detection_to_global,
)
from src.detector_dispatcher import UnifiedDetector


def get_gpu_vram_mb() -> float:
    """Query current GPU VRAM utilization in MB via nvidia-smi or torch."""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip().split("\n")[0])
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        pass
    return 0.0


def profile_batch_pipeline(
    input_dir: Path,
    output_dir: Path,
    altitude: float = 150.0,
    vram_mb: int = 2048,
    conf_thresh: float = 0.20,
    diou_thresh: float = 0.50,
    stall_threshold_sec: float = 1.5,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Run full instrumented batch analysis across all images in input_dir."""
    print("=" * 80)
    print("      DOTA BATCH TRIAGE INSTRUMENTED PROFILING & TELEMETRY SUITE      ")
    print("=" * 80)
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Stall Threshold:  {stall_threshold_sec:.2f}s")

    # Step 1: Clean and recreate output directory
    if output_dir.exists():
        print(f"Cleaning existing output directory: {output_dir} ...")
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output directory prepared.")

    # Find valid images
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    all_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
    if max_images is not None:
        all_files = all_files[:max_images]
    total_files = len(all_files)
    print(f"Discovered {total_files} images to process.\n")

    # Initialize autonomous detector
    print("Initializing UnifiedDetector ...")
    detector = UnifiedDetector()
    print(f"Detector initialized: Backend={detector.backend.value}, Provider={detector.execution_provider}")

    # Warm up 640 session so cold JIT compile doesn't distort runtime telemetry
    print("Pre-warming detector session (tile size 640) ...")
    t_warm = time.perf_counter()
    detector.get_session(640)
    # Warmup forward pass
    dummy_tile = np.zeros((640, 640, 3), dtype=np.uint8)
    detector.predict_tile(dummy_tile, 640)
    print(f"Warmup complete in {time.perf_counter() - t_warm:.2f}s.\n")

    process = psutil.Process()
    stalls: List[Dict[str, Any]] = []
    frame_metrics: List[Dict[str, Any]] = []
    positive_count = 0

    batch_start_time = time.perf_counter()

    for idx, img_path in enumerate(all_files):
        t_frame_start = time.perf_counter()

        # Phase 1: I/O Read & Color conversion
        t0 = time.perf_counter()
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[{idx+1}/{total_files}] Failed to read {img_path.name}")
            continue
        img_h, img_w = bgr.shape[:2]
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        del bgr
        t_read = time.perf_counter() - t0

        # Phase 2: Tiling calculation
        t0 = time.perf_counter()
        tile_size, tiles = calculate_tiles_for_image(img_w, img_h, altitude, vram_mb)
        num_tiles = len(tiles)
        t_tile_plan = time.perf_counter() - t0

        # Phase 3: Raw Tiled Inference
        t0 = time.perf_counter()
        all_global_detections: List[Any] = []
        for tile_idx, tile in enumerate(tiles):
            tx, ty, tw, th = tile.x, tile.y, tile.w, tile.h
            tile_slice = img_rgb[ty : ty + th, tx : tx + tw]
            tile_dets = detector.predict_tile(
                tile_slice,
                tile_size=tile_size,
                conf_threshold=conf_thresh,
            )
            del tile_slice
            for d in tile_dets:
                all_global_detections.append(remap_detection_to_global(d, tile, tile_size, tile_idx))
        t_infer = time.perf_counter() - t0
        del img_rgb

        # Phase 4: NMS Boundary Merging
        t0 = time.perf_counter()
        merged_detections = merge_boundary_detections(
            all_global_detections,
            tiles,
            diou_threshold=diou_thresh,
            conf_threshold=conf_thresh,
        )
        final_detections = format_detections_for_gui(merged_detections)
        t_nms = time.perf_counter() - t0

        # Phase 5: Triage Decision & Output I/O
        t0 = time.perf_counter()
        num_dets = len(final_detections)
        if num_dets > 0:
            positive_count += 1
            dst_file = output_dir / img_path.name
            shutil.copy2(img_path.resolve(), dst_file.resolve())
            save_metadata_cache(dst_file, (img_w, img_h), final_detections)
        t_io_write = time.perf_counter() - t0

        del final_detections, all_global_detections

        t_total = time.perf_counter() - t_frame_start
        rss_mb = process.memory_info().rss / 1e6
        vram_mb_used = get_gpu_vram_mb()

        metric = {
            "idx": idx + 1,
            "filename": img_path.name,
            "width": img_w,
            "height": img_h,
            "num_tiles": num_tiles,
            "num_dets": num_dets,
            "t_read": t_read,
            "t_tile_plan": t_tile_plan,
            "t_infer": t_infer,
            "t_nms": t_nms,
            "t_io_write": t_io_write,
            "t_total": t_total,
            "rss_mb": rss_mb,
            "vram_mb": vram_mb_used,
        }
        frame_metrics.append(metric)

        # Critical Stall Detector (threshold: 1.5s)
        if t_total >= stall_threshold_sec:
            stalls.append(metric)
            print("\n" + "!" * 80)
            print(f"[CRITICAL STALL DETECTED at image #{idx+1}: {img_path.name}]")
            print(f"Dimensions: {img_w}x{img_h} | Tiles: {num_tiles} | Dets: {num_dets}")
            print(
                f"Timing: Read={t_read*1000:.1f}ms | Plan={t_tile_plan*1000:.1f}ms | "
                f"Infer={t_infer*1000:.1f}ms ({t_infer/max(1,num_tiles)*1000:.1f}ms/tile) | "
                f"NMS={t_nms*1000:.1f}ms | Write={t_io_write*1000:.1f}ms | Total={t_total:.3f}s"
            )
            print(f"Process RSS RAM: {rss_mb:.1f} MB | VRAM allocated/used: {vram_mb_used:.1f} MB")
            print("!" * 80 + "\n", flush=True)
        elif (idx + 1) % 25 == 0 or idx == total_files - 1:
            elapsed = time.perf_counter() - batch_start_time
            curr_fps = (idx + 1) / elapsed
            print(
                f"[{idx+1:03d}/{total_files:03d}] {img_path.name} ({img_w}x{img_h}, {num_tiles:3d} tiles) | "
                f"T={t_total*1000:6.1f}ms (Inf:{t_infer*1000:5.1f}ms, NMS:{t_nms*1000:4.1f}ms) | "
                f"RSS={rss_mb:6.1f}MB | Avg FPS={curr_fps:5.2f}",
                flush=True,
            )

    total_batch_time = time.perf_counter() - batch_start_time
    avg_fps = total_files / total_batch_time if total_batch_time > 0 else 0.0

    print("\n" + "=" * 80)
    print("                    PROFILING RUN COMPLETED                    ")
    print("=" * 80)
    print(f"Total Processed:    {total_files} images")
    print(f"Positive Images:    {positive_count} images ({positive_count/max(1,total_files)*100:.1f}%)")
    print(f"Total Elapsed Time: {total_batch_time:.2f} s ({total_batch_time/60:.2f} min)")
    print(f"Overall Average FPS: {avg_fps:.2f} images/sec ({total_batch_time/max(1,total_files)*1000:.1f} ms/img)")
    print(f"Total Stalls (> {stall_threshold_sec}s): {len(stalls)}")

    # Analysis of stalls
    if stalls:
        print("\n--- DETAILED SUMMARY OF ALL STALLED FRAMES ---")
        print(f"{'#':<4} {'Filename':<12} {'Dimensions':<12} {'Tiles':<6} {'Dets':<6} {'Read(ms)':<9} {'Infer(ms)':<10} {'NMS(ms)':<8} {'Total(s)':<8} {'RSS(MB)':<8}")
        print("-" * 90)
        for s in stalls:
            dim_str = f"{s['width']}x{s['height']}"
            print(
                f"{s['idx']:<4} {s['filename']:<12} {dim_str:<12} {s['num_tiles']:<6} {s['num_dets']:<6} "
                f"{s['t_read']*1000:<9.1f} {s['t_infer']*1000:<10.1f} {s['t_nms']*1000:<8.1f} "
                f"{s['t_total']:<8.3f} {s['rss_mb']:<8.1f}"
            )
        print("-" * 90)

    # Save summary report to JSON
    report = {
        "total_files": total_files,
        "positive_count": positive_count,
        "total_batch_time_sec": total_batch_time,
        "average_fps": avg_fps,
        "num_stalls": len(stalls),
        "stalls": stalls,
        "metrics": frame_metrics,
    }

    report_path = repo_root / "scripts" / "profile_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed telemetry data saved to: {report_path}")

    return report


if __name__ == "__main__":
    val_in = Path("/workspaces/MD/data/dota_v1.5/images/val")
    val_out = Path("/workspaces/MD/data/dota_v1.5/images/val_detected")
    profile_batch_pipeline(val_in, val_out)
