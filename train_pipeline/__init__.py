"""
Train Pipeline module for aerial imagery dynamic tiling, dataset preparation, and automated training.
"""

from .dataset_slicer import (
    DOTA_V15_CLASSES,
    VISDRONE_CLASSES,
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
from .export_models import build_trt_engine, export_all_profiles
from .train import calculate_safe_batch_size, main as train_main, run_train

__all__ = [
    "AerialDatasetSlicer",
    "YOLOBox",
    "clip_and_normalize_bbox",
    "BaseAnnotationAdapter",
    "YoloHBBAdapter",
    "YoloOBBAdapter",
    "DotaOBBAdapter",
    "detect_format",
    "get_adapter",
    "DOTA_V15_CLASSES",
    "VISDRONE_CLASSES",
    "export_all_profiles",
    "build_trt_engine",
    "calculate_safe_batch_size",
    "run_train",
    "train_main",
]
