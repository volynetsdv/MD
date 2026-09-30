#!/usr/bin/env python3
"""
CLI entry point for exporting multi-scale ONNX and TensorRT model pool.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from train_pipeline.export_models import (
    GRID_RESOLUTIONS,
    export_all_profiles,
    export_single_profile,
    main,
    parse_arguments,
)

if __name__ == "__main__":
    sys.exit(main())
