"""
Unit tests for train_pipeline.dataset_slicer using pytest.

Tests:
1. Correct clipping and splitting of a bounding box located directly on a tile seam.
2. Filtering out shards/fragments whose remaining area is below the 25% threshold (and configurable threshold).
3. Invariant check: all output YOLO coordinates are strictly clamped within [0.0, 1.0].
4. End-to-end dataset slicing workflow with directory layout verification.
"""

import math
import sys
from pathlib import Path
from typing import List

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import numpy as np
import pytest

from train_pipeline.dataset_slicer import (
    AerialDatasetSlicer,
    YOLOBox,
    clip_and_normalize_bbox,
)



class MockTileRect:
    """Mock tile rectangle satisfying the pytiling_core.Rect interface."""

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


# =============================================================================
# Test 1: Bounding box located strictly on the seam between two adjacent tiles
# =============================================================================
def test_seam_box_splitting_across_two_tiles():
    """
    Test 1:
    A bounding box is centered strictly on the vertical boundary between Tile 0 and Tile 1.
    - Full image size: 2000 x 1000
    - Tile 0 covers x in [0, 1000], y in [0, 1000]
    - Tile 1 covers x in [1000, 2000], y in [0, 1000]
    - Target object: centered at x=1000, y=500 with width=100 px and height=200 px.
      (x in [950, 1050], y in [400, 600]).
    - 50% of the object falls in Tile 0 (x in [950, 1000]), 50% in Tile 1 (x in [1000, 1050]).
    - Both pieces should be preserved because 50% >= 25%.
    - Coordinates in each tile must be correctly normalized.
    """
    img_w = 2000
    img_h = 1000

    # Object center: (1000/2000, 500/1000) = (0.5, 0.5)
    # Width: 100/2000 = 0.05, Height: 200/1000 = 0.20
    orig_box = YOLOBox(class_id=3, x_center=0.5, y_center=0.5, width=0.05, height=0.20)

    tile_left = MockTileRect(0, 0, 1000, 1000)
    tile_right = MockTileRect(1000, 0, 1000, 1000)

    # 1. Slice in Tile Left
    left_box = clip_and_normalize_bbox(
        box=orig_box,
        img_w=img_w,
        img_h=img_h,
        tile_rect=tile_left,
        min_area_ratio=0.25,
    )
    assert left_box is not None, "Tile Left should retain its 50% slice of the object"
    assert left_box.class_id == 3

    # Expected absolute x in left tile: [950, 1000] -> local [950, 1000] / 1000 = [0.95, 1.0]
    # Center = 0.975, width = 0.05
    # Expected absolute y in left tile: [400, 600] -> local [400, 600] / 1000 = [0.4, 0.6]
    # Center = 0.500, height = 0.20
    assert pytest.approx(left_box.x_center, abs=1e-5) == 0.975
    assert pytest.approx(left_box.y_center, abs=1e-5) == 0.500
    assert pytest.approx(left_box.width, abs=1e-5) == 0.05
    assert pytest.approx(left_box.height, abs=1e-5) == 0.20

    # 2. Slice in Tile Right
    right_box = clip_and_normalize_bbox(
        box=orig_box,
        img_w=img_w,
        img_h=img_h,
        tile_rect=tile_right,
        min_area_ratio=0.25,
    )
    assert right_box is not None, "Tile Right should retain its 50% slice of the object"
    assert right_box.class_id == 3

    # Expected absolute x in right tile: [1000, 1050] -> local [0, 50] / 1000 = [0.0, 0.05]
    # Center = 0.025, width = 0.05
    # Expected absolute y in right tile: [400, 600] -> local [400, 600] / 1000 = [0.4, 0.6]
    # Center = 0.500, height = 0.20
    assert pytest.approx(right_box.x_center, abs=1e-5) == 0.025
    assert pytest.approx(right_box.y_center, abs=1e-5) == 0.500
    assert pytest.approx(right_box.width, abs=1e-5) == 0.05
    assert pytest.approx(right_box.height, abs=1e-5) == 0.20

    # Verify that the two pieces perfectly add up to the original physical object width:
    # left width in px: 0.05 * 1000 = 50 px. Right width in px: 0.05 * 1000 = 50 px. Total = 100 px.
    assert (left_box.width * tile_left.w + right_box.width * tile_right.w) == pytest.approx(100.0)


# =============================================================================
# Test 2: Filtering of small fragments below threshold (< 25%)
# =============================================================================
def test_fragment_filtering_below_threshold():
    """
    Test 2:
    An object is divided unevenly by a tile boundary:
    - Total width = 100 px, height = 100 px (area = 10,000 px^2).
    - Boundary is at x = 1000.
    - Object occupies x in [980, 1080], y in [400, 500].
    - In Tile Left (x <= 1000): object slice has width = 20 px, area = 2,000 px^2 (20% of original).
      Since 20% < 25%, this shard MUST BE FILTERED OUT (returns None).
    - In Tile Right (x >= 1000): object slice has width = 80 px, area = 8,000 px^2 (80% of original).
      Since 80% >= 25%, this shard MUST BE PRESERVED.
    """
    img_w = 2000
    img_h = 1000

    # Object x in [980, 1080] -> center = 1030, width = 100
    # Object y in [400, 500]  -> center = 450,  height = 100
    xc_norm = 1030.0 / img_w
    yc_norm = 450.0 / img_h
    w_norm = 100.0 / img_w
    h_norm = 100.0 / img_h

    box = YOLOBox(class_id=1, x_center=xc_norm, y_center=yc_norm, width=w_norm, height=h_norm)

    tile_left = MockTileRect(0, 0, 1000, 1000)
    tile_right = MockTileRect(1000, 0, 1000, 1000)

    # 1. Left Tile: 20% fraction should be discarded
    left_result = clip_and_normalize_bbox(
        box=box,
        img_w=img_w,
        img_h=img_h,
        tile_rect=tile_left,
        min_area_ratio=0.25,
    )
    assert left_result is None, "A fragment retaining only 20% area must be discarded (< 25%)"

    # 2. Left Tile with relaxed threshold (e.g. 0.15 = 15%): should now be retained
    left_result_relaxed = clip_and_normalize_bbox(
        box=box,
        img_w=img_w,
        img_h=img_h,
        tile_rect=tile_left,
        min_area_ratio=0.15,
    )
    assert left_result_relaxed is not None, "Should be retained when min_area_ratio=0.15"

    # 3. Right Tile: 80% fraction should be preserved
    right_result = clip_and_normalize_bbox(
        box=box,
        img_w=img_w,
        img_h=img_h,
        tile_rect=tile_right,
        min_area_ratio=0.25,
    )
    assert right_result is not None, "An 80% area fragment must be kept (>= 25%)"
    assert right_result.class_id == 1

    # In right tile: local x in [0, 80] / 1000 -> center = 0.040, width = 0.080
    assert pytest.approx(right_result.x_center, abs=1e-5) == 0.040
    assert pytest.approx(right_result.width, abs=1e-5) == 0.080

    # 4. Completely non-overlapping tile
    tile_distant = MockTileRect(1500, 500, 500, 500)
    distant_result = clip_and_normalize_bbox(
        box=box,
        img_w=img_w,
        img_h=img_h,
        tile_rect=tile_distant,
        min_area_ratio=0.25,
    )
    assert distant_result is None, "Tile outside the bounding box must return None"


# =============================================================================
# Test 3: Strict boundary clamp invariant: all coordinates strictly in [0.0, 1.0]
# =============================================================================
def test_coordinates_invariant_strictly_within_0_to_1():
    """
    Test 3:
    Verifies that for any corner cases (boxes extending beyond image boundaries,
    boxes on tile perimeters, irregular scales, extreme aspect ratios):
    Every resulting YOLO coordinate satisfies:
        0.0 <= x_center <= 1.0
        0.0 <= y_center <= 1.0
        0.0 <  width    <= 1.0
        0.0 <  height   <= 1.0
        0.0 <= x_center - width/2
        x_center + width/2 <= 1.0
        0.0 <= y_center - height/2
        y_center + height/2 <= 1.0
    """
    img_w = 7680
    img_h = 4320

    # Challenging test cases including edge touches, negative overflows, and corner cuts
    test_boxes = [
        # Box touching top-left corner
        YOLOBox(class_id=0, x_center=0.01, y_center=0.01, width=0.02, height=0.02),
        # Box crossing top-left image boundary (noisy annotation)
        YOLOBox(class_id=0, x_center=0.005, y_center=0.005, width=0.03, height=0.03),
        # Box touching bottom-right corner
        YOLOBox(class_id=1, x_center=0.99, y_center=0.99, width=0.02, height=0.02),
        # Box partially exceeding bottom-right boundary
        YOLOBox(class_id=1, x_center=0.995, y_center=0.995, width=0.03, height=0.03),
        # Large central box spanning multiple tiles
        YOLOBox(class_id=2, x_center=0.50, y_center=0.50, width=0.40, height=0.30),
        # Very thin horizontal box
        YOLOBox(class_id=3, x_center=0.25, y_center=0.30, width=0.20, height=0.005),
        # Very thin vertical box
        YOLOBox(class_id=4, x_center=0.75, y_center=0.60, width=0.005, height=0.20),
    ]

    # Grid of arbitrary tiles covering various positions
    test_tiles = [
        MockTileRect(0, 0, 640, 640),
        MockTileRect(600, 0, 640, 640),
        MockTileRect(3500, 2000, 640, 640),
        MockTileRect(7040, 3680, 640, 640),
        MockTileRect(1920, 1080, 512, 512),
        MockTileRect(7200, 4000, 480, 320),  # Edge partial tile
    ]

    tested_count = 0
    for tile in test_tiles:
        for box in test_boxes:
            clipped = clip_and_normalize_bbox(
                box=box,
                img_w=img_w,
                img_h=img_h,
                tile_rect=tile,
                min_area_ratio=0.10,  # low ratio to stress-test more fragments
            )
            if clipped is not None:
                tested_count += 1
                # Check centers
                assert 0.0 <= clipped.x_center <= 1.0, f"x_center out of bounds: {clipped.x_center}"
                assert 0.0 <= clipped.y_center <= 1.0, f"y_center out of bounds: {clipped.y_center}"
                # Check dimensions
                assert 0.0 < clipped.width <= 1.0, f"width out of bounds: {clipped.width}"
                assert 0.0 < clipped.height <= 1.0, f"height out of bounds: {clipped.height}"

                # Check box extent invariants with floating point margin
                half_w = clipped.width / 2.0
                half_h = clipped.height / 2.0
                assert clipped.x_center - half_w >= -1e-7, (
                    f"x_min < 0: {clipped.x_center - half_w}"
                )
                assert clipped.x_center + half_w <= 1.0 + 1e-7, (
                    f"x_max > 1: {clipped.x_center + half_w}"
                )
                assert clipped.y_center - half_h >= -1e-7, (
                    f"y_min < 0: {clipped.y_center - half_h}"
                )
                assert clipped.y_center + half_h <= 1.0 + 1e-7, (
                    f"y_max > 1: {clipped.y_center + half_h}"
                )

    assert tested_count > 0, "At least several boxes should have valid intersections"


# =============================================================================
# Test 4: Full End-to-End Dataset Slicing with Multi-Processing
# =============================================================================
def test_end_to_end_dataset_slicer_pipeline(tmp_path: Path):
    """
    Test 4:
    Verifies full execution of AerialDatasetSlicer:
    - Creates synthetic aerial test images and corresponding YOLO annotations.
    - Slices dataset using ProcessPoolExecutor.
    - Verifies directory structure (images/train, labels/train).
    - Verifies output tile image readability and label formatting.
    """
    input_img_dir = tmp_path / "raw_images"
    input_lbl_dir = tmp_path / "raw_labels"
    output_dir = tmp_path / "sliced_dataset"

    input_img_dir.mkdir(parents=True)
    input_lbl_dir.mkdir(parents=True)

    # 1. Create two synthetic test images (e.g. 1920x1080)
    w, h = 1920, 1080
    for idx in (1, 2):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Add diagonal gradient and distinct color block
        img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
        img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1)
        img[300:500, 400:600] = (0, 0, 255)
        cv2.imwrite(str(input_img_dir / f"uav_frame_{idx:03d}.jpg"), img)

        # Create YOLO label with two objects
        # Box 1: (400..600, 300..500) -> xc=500/1920, yc=400/1080, w=200/1920, h=200/1080
        # Box 2: (1200..1400, 700..900)
        lbl_content = (
            f"0 {500/w:.6f} {400/h:.6f} {200/w:.6f} {200/h:.6f}\n"
            f"2 {1300/w:.6f} {800/h:.6f} {200/w:.6f} {200/h:.6f}\n"
        )
        with open(input_lbl_dir / f"uav_frame_{idx:03d}.txt", "w", encoding="utf-8") as f:
            f.write(lbl_content)

    # 2. Run AerialDatasetSlicer with 2 worker processes
    slicer = AerialDatasetSlicer(
        image_dir=input_img_dir,
        annotation_dir=input_lbl_dir,
        output_dir=output_dir,
        altitude=100.0,
        vram_mb=2048,
        min_area_ratio=0.25,
        split="train",
        save_empty_tiles=False,
        num_workers=2,
    )

    summary = slicer.process_dataset()

    # 3. Assert execution results
    assert summary["total_images"] == 2
    assert summary["processed_successfully"] == 2
    assert summary["failed_count"] == 0
    assert summary["total_tiles_generated"] > 0
    assert summary["total_annotations_generated"] > 0

    # 4. Check directory structure
    out_images = output_dir / "images" / "train"
    out_labels = output_dir / "labels" / "train"
    assert out_images.exists()
    assert out_labels.exists()

    generated_images = list(out_images.glob("*.jpg"))
    generated_labels = list(out_labels.glob("*.txt"))
    assert len(generated_images) == summary["total_tiles_generated"]
    assert len(generated_labels) == summary["total_tiles_generated"]

    # 5. Verify image content & label invariants on generated files
    for img_file in generated_images:
        loaded = cv2.imread(str(img_file))
        assert loaded is not None, f"Generated tile image {img_file} is unreadable"

        # Matching label file
        lbl_file = out_labels / f"{img_file.stem}.txt"
        assert lbl_file.exists(), f"Missing corresponding label file for {img_file}"

        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    box = YOLOBox.from_yolo_line(line_str)
                    assert 0.0 <= box.x_center <= 1.0
                    assert 0.0 <= box.y_center <= 1.0
                    assert 0.0 < box.width <= 1.0
                    assert 0.0 < box.height <= 1.0
