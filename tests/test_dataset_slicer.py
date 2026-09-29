"""
Unit tests for train_pipeline.dataset_slicer using pytest.

Tests:
1. Correct clipping and splitting of a bounding box located directly on a tile seam.
2. Filtering out shards/fragments whose remaining area is below the 25% threshold.
3. Invariant check: all output YOLO coordinates are strictly clamped within [0.0, 1.0].
4. End-to-end dataset slicing workflow with directory layout verification.
5. DotaOBBAdapter: conversion of 8-point polygons into normalized YOLO HBB.
6. DotaOBBAdapter: skipping headers (imagesource:, gsd:) and handling difficult flag.
7. detect_format: auto-detection of YOLO vs DOTA annotation files.
8. Multi-extension image discovery (.png, .tif, .jpg, .bmp) with case-insensitivity.
9. FileNotFoundError raised on empty image directory or zero matching annotations.
10. End-to-end DOTA slicing with format='auto' and mixed image extensions.
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
    DOTA_V15_CLASSES,
    AerialDatasetSlicer,
    BaseAnnotationAdapter,
    DotaOBBAdapter,
    YoloHBBAdapter,
    YoloOBBAdapter,
    YOLOBox,
    clip_and_normalize_bbox,
    detect_format,
    get_adapter,
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
    """
    img_w = 7680
    img_h = 4320

    test_boxes = [
        YOLOBox(class_id=0, x_center=0.01, y_center=0.01, width=0.02, height=0.02),
        YOLOBox(class_id=0, x_center=0.005, y_center=0.005, width=0.03, height=0.03),
        YOLOBox(class_id=1, x_center=0.99, y_center=0.99, width=0.02, height=0.02),
        YOLOBox(class_id=1, x_center=0.995, y_center=0.995, width=0.03, height=0.03),
        YOLOBox(class_id=2, x_center=0.50, y_center=0.50, width=0.40, height=0.30),
        YOLOBox(class_id=3, x_center=0.25, y_center=0.30, width=0.20, height=0.005),
        YOLOBox(class_id=4, x_center=0.75, y_center=0.60, width=0.005, height=0.20),
    ]

    test_tiles = [
        MockTileRect(0, 0, 640, 640),
        MockTileRect(600, 0, 640, 640),
        MockTileRect(3500, 2000, 640, 640),
        MockTileRect(7040, 3680, 640, 640),
        MockTileRect(1920, 1080, 512, 512),
        MockTileRect(7200, 4000, 480, 320),
    ]

    tested_count = 0
    for tile in test_tiles:
        for box in test_boxes:
            clipped = clip_and_normalize_bbox(
                box=box,
                img_w=img_w,
                img_h=img_h,
                tile_rect=tile,
                min_area_ratio=0.10,
            )
            if clipped is not None:
                tested_count += 1
                assert 0.0 <= clipped.x_center <= 1.0, f"x_center out of bounds: {clipped.x_center}"
                assert 0.0 <= clipped.y_center <= 1.0, f"y_center out of bounds: {clipped.y_center}"
                assert 0.0 < clipped.width <= 1.0, f"width out of bounds: {clipped.width}"
                assert 0.0 < clipped.height <= 1.0, f"height out of bounds: {clipped.height}"

                half_w = clipped.width / 2.0
                half_h = clipped.height / 2.0
                assert clipped.x_center - half_w >= -1e-7
                assert clipped.x_center + half_w <= 1.0 + 1e-7
                assert clipped.y_center - half_h >= -1e-7
                assert clipped.y_center + half_h <= 1.0 + 1e-7

    assert tested_count > 0, "At least several boxes should have valid intersections"


# =============================================================================
# Test 4: Full End-to-End Dataset Slicing with Multi-Processing
# =============================================================================
def test_end_to_end_dataset_slicer_pipeline(tmp_path: Path):
    """
    Test 4:
    Verifies full execution of AerialDatasetSlicer with YOLO format.
    """
    input_img_dir = tmp_path / "raw_images"
    input_lbl_dir = tmp_path / "raw_labels"
    output_dir = tmp_path / "sliced_dataset"

    input_img_dir.mkdir(parents=True)
    input_lbl_dir.mkdir(parents=True)

    w, h = 1920, 1080
    for idx in (1, 2):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
        img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1)
        img[300:500, 400:600] = (0, 0, 255)
        cv2.imwrite(str(input_img_dir / f"uav_frame_{idx:03d}.jpg"), img)

        lbl_content = (
            f"0 {500/w:.6f} {400/h:.6f} {200/w:.6f} {200/h:.6f}\n"
            f"2 {1300/w:.6f} {800/h:.6f} {200/w:.6f} {200/h:.6f}\n"
        )
        with open(input_lbl_dir / f"uav_frame_{idx:03d}.txt", "w", encoding="utf-8") as f:
            f.write(lbl_content)

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

    assert summary["total_images"] == 2
    assert summary["processed_successfully"] == 2
    assert summary["failed_count"] == 0
    assert summary["total_tiles_generated"] > 0
    assert summary["total_annotations_generated"] > 0

    out_images = output_dir / "images" / "train"
    out_labels = output_dir / "labels" / "train"
    assert out_images.exists()
    assert out_labels.exists()

    generated_images = list(out_images.glob("*.jpg"))
    generated_labels = list(out_labels.glob("*.txt"))
    assert len(generated_images) == summary["total_tiles_generated"]
    assert len(generated_labels) == summary["total_tiles_generated"]

    for img_file in generated_images:
        loaded = cv2.imread(str(img_file))
        assert loaded is not None, f"Generated tile image {img_file} is unreadable"
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


# =============================================================================
# Test 5: DotaOBBAdapter conversion (8-point polygon -> YOLO HBB)
# =============================================================================
def test_dota_obb_adapter_conversion():
    """
    Test 5:
    Validates conversion of DOTA v1.5 oriented bounding boxes (OBB) to YOLO HBB:
    - Standard axis-aligned rectangle
    - 45-degree diamond rotated polygon
    - Multi-word class name (e.g. 'large vehicle' vs 'large-vehicle')
    """
    adapter = DotaOBBAdapter()
    img_w, img_h = 1000, 1000

    # 1. Axis-aligned rectangle: (100, 100) to (300, 200), class 'plane' (id 0)
    line1 = "100.0 100.0 300.0 100.0 300.0 200.0 100.0 200.0 plane 0"
    box1 = adapter.parse_line(line1, img_w=img_w, img_h=img_h)
    assert box1 is not None
    assert box1.class_id == 0
    assert pytest.approx(box1.x_center, abs=1e-5) == 0.200  # (100+300)/2 / 1000
    assert pytest.approx(box1.y_center, abs=1e-5) == 0.150  # (100+200)/2 / 1000
    assert pytest.approx(box1.width, abs=1e-5) == 0.200     # (300-100) / 1000
    assert pytest.approx(box1.height, abs=1e-5) == 0.100    # (200-100) / 1000

    # 2. Diamond polygon: vertices (150, 100), (200, 150), (150, 200), (100, 150), class 'storage-tank' (id 2)
    line2 = "150.0 100.0 200.0 150.0 150.0 200.0 100.0 150.0 storage-tank 0"
    box2 = adapter.parse_line(line2, img_w=img_w, img_h=img_h)
    assert box2 is not None
    assert box2.class_id == 2
    # x_min=100, x_max=200 -> xc=150/1000=0.15, w=100/1000=0.10
    # y_min=100, y_max=200 -> yc=150/1000=0.15, h=100/1000=0.10
    assert pytest.approx(box2.x_center, abs=1e-5) == 0.150
    assert pytest.approx(box2.y_center, abs=1e-5) == 0.150
    assert pytest.approx(box2.width, abs=1e-5) == 0.100
    assert pytest.approx(box2.height, abs=1e-5) == 0.100

    # 3. Space in class name: 'large vehicle' (id 9)
    line3 = "400.0 400.0 500.0 400.0 500.0 600.0 400.0 600.0 large vehicle 0"
    box3 = adapter.parse_line(line3, img_w=img_w, img_h=img_h)
    assert box3 is not None
    assert box3.class_id == 9

    # 4. Out of bounds coordinates: should be clamped
    line4 = "-50.0 100.0 200.0 100.0 200.0 1100.0 -50.0 1100.0 ship 0"
    box4 = adapter.parse_line(line4, img_w=img_w, img_h=img_h)
    assert box4 is not None
    assert box4.class_id == 1
    # Clamped x to [0, 200] -> xc = 100/1000 = 0.10, w = 200/1000 = 0.20
    # Clamped y to [100, 1000] -> yc = 550/1000 = 0.55, h = 900/1000 = 0.90
    assert pytest.approx(box4.x_center, abs=1e-5) == 0.100
    assert pytest.approx(box4.y_center, abs=1e-5) == 0.550
    assert pytest.approx(box4.width, abs=1e-5) == 0.200
    assert pytest.approx(box4.height, abs=1e-5) == 0.900


# =============================================================================
# Test 6: DotaOBBAdapter headers and difficult flag handling
# =============================================================================
def test_dota_header_and_difficult_handling(tmp_path: Path):
    """
    Test 6:
    Verifies that DOTA file headers (imagesource, gsd, comments) are safely ignored,
    and that difficult=1 flag is filtered when requested.
    """
    dota_file = tmp_path / "dota_sample.txt"
    content = (
        "imagesource:GoogleEarth\n"
        "gsd:0.1463435\n"
        "# Comment line\n"
        "100.0 100.0 200.0 100.0 200.0 200.0 100.0 200.0 plane 0\n"
        "300.0 300.0 400.0 300.0 400.0 400.0 300.0 400.0 bridge 1\n"
    )
    dota_file.write_text(content, encoding="utf-8")

    # 1. Standard adapter: retains difficult=1
    adapter_standard = DotaOBBAdapter(ignore_difficult=False)
    boxes = adapter_standard.parse_file(dota_file, img_w=1000, img_h=1000)
    assert len(boxes) == 2
    assert boxes[0].class_id == DOTA_V15_CLASSES["plane"]
    assert boxes[1].class_id == DOTA_V15_CLASSES["bridge"]

    # 2. Filter difficult: skips bridge (difficult=1)
    adapter_filter = DotaOBBAdapter(ignore_difficult=True)
    filtered_boxes = adapter_filter.parse_file(dota_file, img_w=1000, img_h=1000)
    assert len(filtered_boxes) == 1
    assert filtered_boxes[0].class_id == DOTA_V15_CLASSES["plane"]


# =============================================================================
# Test 7: Auto-detection of dataset format
# =============================================================================
def test_detect_format_auto(tmp_path: Path):
    """
    Test 7:
    Verifies detect_format correctly differentiates between YOLO HBB and DOTA OBB.
    """
    yolo_file = tmp_path / "label_yolo.txt"
    yolo_file.write_text("0 0.500000 0.500000 0.200000 0.200000\n1 0.100000 0.200000 0.050000 0.050000\n")

    yolo_obb_file = tmp_path / "label_yolo_obb.txt"
    yolo_obb_file.write_text(
        "10 0.579097 0.325518 0.581677 0.326245 0.579355 0.329517 0.577548 0.32879\n"
    )

    dota_file = tmp_path / "label_dota.txt"
    dota_file.write_text(
        "imagesource:GoogleEarth\ngsd:0.12\n100.0 100.0 200.0 100.0 200.0 200.0 100.0 200.0 plane 0\n"
    )

    empty_file = tmp_path / "label_empty.txt"
    empty_file.write_text("\n\n")

    assert detect_format(yolo_file) == "yolo"
    assert detect_format(yolo_obb_file) == "yolo_obb"
    assert detect_format(dota_file) == "dota"
    assert detect_format(empty_file) == "yolo"  # default fallback

    # Factory verification
    assert isinstance(get_adapter("yolo"), YoloHBBAdapter)
    assert isinstance(get_adapter("yolo_obb"), YoloOBBAdapter)
    assert isinstance(get_adapter("dota"), DotaOBBAdapter)
    with pytest.raises(ValueError):
        get_adapter("unknown_format")


# =============================================================================
# Test 8: Multi-extension image discovery (case-insensitive)
# =============================================================================
def test_multi_extension_image_matching(tmp_path: Path):
    """
    Test 8:
    Verifies that find_image_files detects .png, .tif, .tiff, .bmp, .jpg, .jpeg
    regardless of file extension case.
    """
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    out_dir = tmp_path / "out"
    img_dir.mkdir()
    lbl_dir.mkdir()

    exts = [".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIFF", ".bmp"]
    for idx, ext in enumerate(exts):
        f_name = f"sample_{idx:02d}{ext}"
        (img_dir / f_name).write_bytes(b"\x00")
        (lbl_dir / f"sample_{idx:02d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    slicer = AerialDatasetSlicer(
        image_dir=img_dir,
        annotation_dir=lbl_dir,
        output_dir=out_dir,
    )
    found = slicer.find_image_files()
    assert len(found) == len(exts)


# =============================================================================
# Test 9: FileNotFoundError on empty image dir or missing matching labels
# =============================================================================
def test_zero_matches_raises_file_not_found(tmp_path: Path):
    """
    Test 9:
    Verifies explicit FileNotFoundError is raised:
    1. If 0 images found in image_dir.
    2. If images found, but 0 matching .txt labels found in annotation_dir.
    """
    empty_img_dir = tmp_path / "empty_images"
    empty_img_dir.mkdir()
    some_lbl_dir = tmp_path / "labels"
    some_lbl_dir.mkdir()
    out_dir = tmp_path / "out"

    # 1. Zero images found
    slicer1 = AerialDatasetSlicer(image_dir=empty_img_dir, annotation_dir=some_lbl_dir, output_dir=out_dir)
    with pytest.raises(FileNotFoundError) as exc1:
        slicer1.process_dataset()
    assert "No image files found" in str(exc1.value)

    # 2. Images found, but zero matching labels
    img_dir_with_files = tmp_path / "valid_images"
    img_dir_with_files.mkdir()
    (img_dir_with_files / "frame_001.png").write_bytes(b"\x00")
    (img_dir_with_files / "frame_002.png").write_bytes(b"\x00")

    # Mismatched labels in some_lbl_dir (e.g. other_name.txt)
    (some_lbl_dir / "unrelated.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    slicer2 = AerialDatasetSlicer(image_dir=img_dir_with_files, annotation_dir=some_lbl_dir, output_dir=out_dir)
    with pytest.raises(FileNotFoundError) as exc2:
        slicer2.process_dataset()
    assert "No matching .txt annotation files found" in str(exc2.value)


# =============================================================================
# Test 10: Full End-to-End DOTA v1.5 Slicing with format='auto'
# =============================================================================
def test_end_to_end_dota_slicing(tmp_path: Path):
    """
    Test 10:
    Verifies full execution of AerialDatasetSlicer on DOTA v1.5 annotated images:
    - Mixed image extensions (.png and .tif)
    - DOTA OBB annotations with headers
    - Auto-detection enabled (format='auto')
    - Verifies output tiles are generated in valid YOLO format
    """
    input_img_dir = tmp_path / "dota_images"
    input_lbl_dir = tmp_path / "dota_labels"
    output_dir = tmp_path / "dota_sliced"

    input_img_dir.mkdir(parents=True)
    input_lbl_dir.mkdir(parents=True)

    w, h = 1600, 1200

    # Image 1: PNG format
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    img1[200:400, 200:400] = (255, 0, 0)
    cv2.imwrite(str(input_img_dir / "P0001.png"), img1)

    dota_lbl1 = (
        "imagesource:GoogleEarth\n"
        "gsd:0.15\n"
        "200.0 200.0 400.0 200.0 400.0 400.0 200.0 400.0 plane 0\n"
        "800.0 600.0 1000.0 600.0 1000.0 800.0 800.0 800.0 ship 0\n"
    )
    (input_lbl_dir / "P0001.txt").write_text(dota_lbl1, encoding="utf-8")

    # Image 2: TIF format
    img2 = np.zeros((h, w, 3), dtype=np.uint8)
    img2[500:700, 500:700] = (0, 255, 0)
    cv2.imwrite(str(input_img_dir / "P0002.tif"), img2)

    dota_lbl2 = (
        "imagesource:GoogleEarth\n"
        "500.0 500.0 700.0 500.0 700.0 700.0 500.0 700.0 storage-tank 0\n"
    )
    (input_lbl_dir / "P0002.txt").write_text(dota_lbl2, encoding="utf-8")

    slicer = AerialDatasetSlicer(
        image_dir=input_img_dir,
        annotation_dir=input_lbl_dir,
        output_dir=output_dir,
        altitude=120.0,
        vram_mb=2048,
        min_area_ratio=0.25,
        split="val",
        format="auto",
        num_workers=2,
    )

    summary = slicer.process_dataset()

    assert summary["total_images"] == 2
    assert summary["processed_successfully"] == 2
    assert summary["failed_count"] == 0
    assert summary["detected_format"] == "dota"
    assert summary["total_tiles_generated"] > 0
    assert summary["total_annotations_generated"] > 0

    out_images = output_dir / "images" / "val"
    out_labels = output_dir / "labels" / "val"
    assert out_images.exists()
    assert out_labels.exists()

    generated_images = list(out_images.glob("*.jpg"))
    generated_labels = list(out_labels.glob("*.txt"))
    assert len(generated_images) == summary["total_tiles_generated"]
    assert len(generated_labels) == summary["total_tiles_generated"]

    # Verify that all output labels are well-formed YOLO coordinates
    for lbl_file in generated_labels:
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    box = YOLOBox.from_yolo_line(line_str)
                    assert box.class_id in (0, 1, 2)  # plane, ship, storage-tank
                    assert 0.0 <= box.x_center <= 1.0
                    assert 0.0 <= box.y_center <= 1.0
                    assert 0.0 < box.width <= 1.0
                    assert 0.0 < box.height <= 1.0


# =============================================================================
# Test 11: YoloOBBAdapter conversion (9 numerical tokens -> YOLO HBB)
# =============================================================================
def test_yolo_obb_adapter_conversion(tmp_path: Path):
    """
    Test 11:
    Validates conversion of Ultralytics YOLO-OBB format:
    9 tokens: class_id x1 y1 x2 y2 x3 y3 x4 y4 (already normalized floats)
    """
    adapter = YoloOBBAdapter()

    # 1. Real Ultralytics DOTA line from task description
    line1 = "10 0.579097 0.325518 0.581677 0.326245 0.579355 0.329517 0.577548 0.32879"
    box1 = adapter.parse_line(line1, img_width=1024, img_height=1024)
    assert box1 is not None
    assert box1.class_id == 10

    # xs: [0.579097, 0.581677, 0.579355, 0.577548] -> x_min=0.577548, x_max=0.581677
    # ys: [0.325518, 0.326245, 0.329517, 0.328790] -> y_min=0.325518, y_max=0.329517
    expected_xc = (0.577548 + 0.581677) / 2.0
    expected_yc = (0.325518 + 0.329517) / 2.0
    expected_w = 0.581677 - 0.577548
    expected_h = 0.329517 - 0.325518

    assert pytest.approx(box1.x_center, abs=1e-5) == expected_xc
    assert pytest.approx(box1.y_center, abs=1e-5) == expected_yc
    assert pytest.approx(box1.width, abs=1e-5) == expected_w
    assert pytest.approx(box1.height, abs=1e-5) == expected_h

    # 2. Rotated diamond in normalized space
    # (0.50, 0.40), (0.60, 0.50), (0.50, 0.60), (0.40, 0.50) -> x_min=0.40, x_max=0.60, y_min=0.40, y_max=0.60
    line2 = "2 0.50 0.40 0.60 0.50 0.50 0.60 0.40 0.50"
    box2 = adapter.parse_line(line2)
    assert box2 is not None
    assert box2.class_id == 2
    assert pytest.approx(box2.x_center, abs=1e-5) == 0.500
    assert pytest.approx(box2.y_center, abs=1e-5) == 0.500
    assert pytest.approx(box2.width, abs=1e-5) == 0.200
    assert pytest.approx(box2.height, abs=1e-5) == 0.200

    # 3. Test parse_file with keyword arguments (img_w / img_width)
    test_file = tmp_path / "sample_obb.txt"
    test_file.write_text(f"{line1}\n{line2}\n", encoding="utf-8")

    boxes_a = adapter.parse_file(test_file, img_w=1920, img_h=1080)
    assert len(boxes_a) == 2

    boxes_b = adapter.parse_file(test_file, img_width=1920, img_height=1080)
    assert len(boxes_b) == 2
    assert boxes_a[0].x_center == boxes_b[0].x_center


# =============================================================================
# Test 12: End-to-End Slicing with YOLO-OBB Format (Auto-detect)
# =============================================================================
def test_end_to_end_yolo_obb_slicing(tmp_path: Path):
    """
    Test 12:
    Verifies full execution of AerialDatasetSlicer on YOLO-OBB annotated images
    with format='auto':
    - Auto-detection identifies format as 'yolo_obb'
    - Slices 1600x1200 image into tiles
    - Output tiles contain valid converted YOLO HBB annotations
    """
    input_img_dir = tmp_path / "obb_images"
    input_lbl_dir = tmp_path / "obb_labels"
    output_dir = tmp_path / "obb_sliced"

    input_img_dir.mkdir(parents=True)
    input_lbl_dir.mkdir(parents=True)

    w, h = 1600, 1200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[200:500, 300:700] = (100, 150, 200)
    cv2.imwrite(str(input_img_dir / "uav_obb_01.png"), img)

    # YOLO-OBB annotation (normalized coordinates around center 0.5, 0.5)
    obb_content = (
        "10 0.579097 0.325518 0.581677 0.326245 0.579355 0.329517 0.577548 0.32879\n"
        "1 0.400000 0.400000 0.600000 0.400000 0.600000 0.600000 0.400000 0.600000\n"
    )
    (input_lbl_dir / "uav_obb_01.txt").write_text(obb_content, encoding="utf-8")

    slicer = AerialDatasetSlicer(
        image_dir=input_img_dir,
        annotation_dir=input_lbl_dir,
        output_dir=output_dir,
        altitude=100.0,
        vram_mb=2048,
        min_area_ratio=0.20,
        split="train",
        format="auto",
        num_workers=1,
    )

    summary = slicer.process_dataset()

    assert summary["total_images"] == 1
    assert summary["processed_successfully"] == 1
    assert summary["failed_count"] == 0
    assert summary["detected_format"] == "yolo_obb"
    assert summary["total_tiles_generated"] > 0
    assert summary["total_annotations_generated"] > 0

    out_images = output_dir / "images" / "train"
    out_labels = output_dir / "labels" / "train"
    generated_labels = list(out_labels.glob("*.txt"))
    assert len(generated_labels) == summary["total_tiles_generated"]

    for lbl_file in generated_labels:
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    box = YOLOBox.from_yolo_line(line_str)
                    assert box.class_id in (1, 10)
                    assert 0.0 <= box.x_center <= 1.0
                    assert 0.0 <= box.y_center <= 1.0
                    assert 0.0 < box.width <= 1.0
                    assert 0.0 < box.height <= 1.0

