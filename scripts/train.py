#!/usr/bin/env python3
"""
CLI entry point for training pipeline from scripts directory.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from train_pipeline.train import (
    calculate_safe_batch_size,
    main,
    run_train,
)

if __name__ == "__main__":
    sys.exit(main())
