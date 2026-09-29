#!/usr/bin/env python3
"""
CLI entry point for AerialDatasetSlicer from scripts directory.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

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
    main,
)

if __name__ == "__main__":
    sys.exit(main())

