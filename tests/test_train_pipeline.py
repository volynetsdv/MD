"""
Unit and integration tests for automated train pipeline (dataset.yaml, slicing_meta.json,
safe batch size calculation, and train.py CLI execution).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import numpy as np
import pytest
import yaml

from train_pipeline.dataset_slicer import (
    DOTA_V15_CLASSES,
    VISDRONE_CLASSES,
    AerialDatasetSlicer,
)
from train_pipeline.train import (
    calculate_safe_batch_size,
    main as train_main,
    run_train,
)


@pytest.fixture
def mock_dataset_source(tmp_path: Path):
    """Create a minimal raw dataset with synthetic image and YOLO annotations."""
    img_dir = tmp_path / "raw_images"
    ann_dir = tmp_path / "raw_annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    # 1920x1080 synthetic test image
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.circle(img, (500, 500), 40, (0, 255, 0), -1)
    cv2.circle(img, (1200, 800), 50, (0, 0, 255), -1)
    cv2.imwrite(str(img_dir / "drone_sample_01.jpg"), img)

    # YOLO annotations: class_id x_center y_center width height
    ann_content = (
        "0 0.2604 0.4630 0.0416 0.0740\n"
        "3 0.6250 0.7407 0.0520 0.0925\n"
    )
    with open(ann_dir / "drone_sample_01.txt", "w", encoding="utf-8") as f:
        f.write(ann_content)

    return img_dir, ann_dir


@pytest.fixture
def mock_dota_source(tmp_path: Path):
    """Create a minimal raw dataset with DOTA annotations."""
    img_dir = tmp_path / "dota_images"
    ann_dir = tmp_path / "dota_annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    img = np.full((1200, 1600, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "P0001.png"), img)

    ann_content = (
        "imagesource:GoogleEarth\n"
        "gsd:0.15\n"
        "100.0 100.0 200.0 100.0 200.0 200.0 100.0 200.0 plane 0\n"
        "400.0 400.0 600.0 400.0 600.0 500.0 400.0 500.0 ship 0\n"
    )
    with open(ann_dir / "P0001.txt", "w", encoding="utf-8") as f:
        f.write(ann_content)

    return img_dir, ann_dir


def test_slicer_generates_dataset_yaml_and_slicing_meta(tmp_path: Path, mock_dataset_source):
    """
    Verify that AerialDatasetSlicer automatically produces both
    dataset.yaml and slicing_meta.json in --output-dir with correct structure.
    """
    img_dir, ann_dir = mock_dataset_source
    out_dir = tmp_path / "sliced_out"

    slicer = AerialDatasetSlicer(
        image_dir=img_dir,
        annotation_dir=ann_dir,
        output_dir=out_dir,
        altitude=120.0,
        vram_mb=4096,
        split="train",
    )
    summary = slicer.process_dataset()

    assert summary["processed_successfully"] == 1
    assert summary["failed_count"] == 0

    yaml_file = out_dir / "dataset.yaml"
    meta_file = out_dir / "slicing_meta.json"

    assert yaml_file.exists(), "dataset.yaml was not generated in output_dir"
    assert meta_file.exists(), "slicing_meta.json was not generated in output_dir"
    assert summary["dataset_yaml"] == str(yaml_file)
    assert summary["slicing_meta"] == str(meta_file)

    # 1. Validate dataset.yaml content
    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    assert "path" in yaml_data
    assert Path(yaml_data["path"]).resolve() == out_dir.resolve()
    assert "train" in yaml_data
    assert "val" in yaml_data
    assert "names" in yaml_data
    assert isinstance(yaml_data["names"], dict)
    assert len(yaml_data["names"]) > 0

    # 2. Validate slicing_meta.json content
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    assert "tile_size" in meta_data
    assert meta_data["tile_size"] in [320, 416, 512, 640]
    assert meta_data["altitude"] == 120.0
    assert meta_data["vram_mb"] == 4096
    assert meta_data["classes_count"] == len(yaml_data["names"])
    assert meta_data["split"] == "train"


def test_tile_size_override_reflected_in_meta_and_yaml(tmp_path: Path, mock_dataset_source):
    """
    Verify that an explicit target_size override (e.g. 416) is propagated
    to slicing_meta.json, dataset tiles, and summary report.
    """
    img_dir, ann_dir = mock_dataset_source
    out_dir = tmp_path / "sliced_416"

    slicer = AerialDatasetSlicer(
        image_dir=img_dir,
        annotation_dir=ann_dir,
        output_dir=out_dir,
        target_size=416,
    )
    summary = slicer.process_dataset()

    assert summary["tile_size"] == 416

    meta_file = out_dir / "slicing_meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["tile_size"] == 416

    # Verify tile resolution on disk
    tile_images = list((out_dir / "images" / "train").glob("*.jpg"))
    assert len(tile_images) > 0
    sample_tile = cv2.imread(str(tile_images[0]))
    assert sample_tile.shape[0] == 416
    assert sample_tile.shape[1] == 416


def test_dota_format_classes_in_yaml_and_meta(tmp_path: Path, mock_dota_source):
    """
    Verify DOTA dataset produces 16 classes in dataset.yaml and slicing_meta.json.
    """
    img_dir, ann_dir = mock_dota_source
    out_dir = tmp_path / "dota_sliced_out"

    slicer = AerialDatasetSlicer(
        image_dir=img_dir,
        annotation_dir=ann_dir,
        output_dir=out_dir,
        format="dota",
        target_size=512,
    )
    summary = slicer.process_dataset()
    assert summary["processed_successfully"] == 1

    with open(out_dir / "dataset.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(out_dir / "slicing_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["classes_count"] == 16
    assert meta["tile_size"] == 512
    assert len(cfg["names"]) == 16
    assert cfg["names"][0] == "plane"
    assert cfg["names"][1] == "ship"


def test_calculate_safe_batch_size():
    """
    Verify VRAM-safe batch size calculation logic across different devices and GPU sizes.
    """
    # 1. CPU execution
    cpu_batch = calculate_safe_batch_size(tile_size=640, device="cpu", default_cpu_batch=8)
    assert cpu_batch == 8

    # 2. Large GPU (16 GB VRAM)
    batch_16gb_512 = calculate_safe_batch_size(tile_size=512, total_vram_gb=16.0)
    assert batch_16gb_512 in [32, 64]

    batch_16gb_640 = calculate_safe_batch_size(tile_size=640, total_vram_gb=16.0)
    assert batch_16gb_640 in [32, 64]

    # 3. Medium GPU (8 GB VRAM)
    batch_8gb_640 = calculate_safe_batch_size(tile_size=640, total_vram_gb=8.0)
    assert batch_8gb_640 == 32

    # 4. Small GPU (4 GB VRAM)
    batch_4gb_640 = calculate_safe_batch_size(tile_size=640, total_vram_gb=4.0)
    assert batch_4gb_640 in [8, 16]

    # 5. Very small GPU (2 GB VRAM)
    batch_2gb_640 = calculate_safe_batch_size(tile_size=640, total_vram_gb=2.0)
    assert batch_2gb_640 in [2, 4]

    # 6. Scaling with tile size: 320px allows larger or equal batch than 640px
    batch_4gb_320 = calculate_safe_batch_size(tile_size=320, total_vram_gb=4.0)
    assert batch_4gb_320 >= batch_4gb_640


def test_run_train_dry_run_success(tmp_path: Path, mock_dataset_source):
    """
    Verify run_train with dry_run=True parses config, sets imgsz from slicing_meta.json,
    computes batch size, and successfully finishes without invoking heavy training.
    """
    img_dir, ann_dir = mock_dataset_source
    out_dir = tmp_path / "sliced_for_train"

    # 1. Slice dataset
    slicer = AerialDatasetSlicer(
        image_dir=img_dir,
        annotation_dir=ann_dir,
        output_dir=out_dir,
        target_size=512,
    )
    slicer.process_dataset()

    # 2. Run train pipeline in dry-run mode
    res = run_train(
        data_dir=out_dir,
        model="yolo11s.pt",
        epochs=30,
        batch="auto",
        device="cpu",
        dry_run=True,
    )

    assert res["status"] == "dry_run_success"
    assert res["config"]["imgsz"] == 512
    assert res["config"]["epochs"] == 30
    assert res["config"]["device"] == "cpu"
    assert res["config"]["batch"] == 8
    assert res["config"]["data"] == str(out_dir / "dataset.yaml")
    assert res["meta"]["tile_size"] == 512


def test_train_cli_dry_run(tmp_path: Path, mock_dataset_source):
    """
    Verify train_main CLI entry point handles arguments and runs in dry-run mode.
    """
    img_dir, ann_dir = mock_dataset_source
    out_dir = tmp_path / "cli_train_dir"

    slicer = AerialDatasetSlicer(
        image_dir=img_dir,
        annotation_dir=ann_dir,
        output_dir=out_dir,
        target_size=416,
    )
    slicer.process_dataset()

    # CLI invocation with explicit batch
    exit_code = train_main([
        "--data-dir", str(out_dir),
        "--model", "yolo11s.pt",
        "--epochs", "25",
        "--batch", "16",
        "--device", "cpu",
        "--dry-run",
    ])
    assert exit_code == 0


def test_run_train_missing_files_error_handling(tmp_path: Path):
    """
    Verify run_train and CLI fail gracefully with informative FileNotFoundError
    when dataset.yaml or slicing_meta.json are absent.
    """
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    # Missing dataset.yaml
    with pytest.raises(FileNotFoundError, match="dataset.yaml"):
        run_train(data_dir=empty_dir, dry_run=True)

    # Missing slicing_meta.json
    (empty_dir / "dataset.yaml").write_text("dummy: data", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="slicing_meta.json"):
        run_train(data_dir=empty_dir, dry_run=True)

    # CLI returns code 1 on error
    cli_code = train_main(["--data-dir", str(empty_dir), "--dry-run"])
    assert cli_code == 1
