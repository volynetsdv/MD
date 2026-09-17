#!/usr/bin/env python3
"""Automated Release Packaging Script for Aerial Reconnaissance Workstation.

Handles cross-platform packaging for Windows (.exe) and Linux (.AppImage / standalone folder).
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("BuildRelease")

REPO_ROOT = Path(__file__).resolve().parent
DIST_DIR = REPO_ROOT / "dist"
BUILD_DIR = REPO_ROOT / "build"


def check_prerequisites() -> bool:
    """Verify that required models and configuration exist."""
    models_dir = REPO_ROOT / "models"
    required_models = ["yolo_320.onnx", "yolo_416.onnx", "yolo_512.onnx", "yolo_640.onnx"]
    for m in required_models:
        if not (models_dir / m).exists():
            logger.error("Missing required model asset: %s", models_dir / m)
            return False

    spec_file = REPO_ROOT / "app.spec"
    if not spec_file.exists():
        logger.error("Missing PyInstaller specification: %s", spec_file)
        return False

    return True


def build_cpp_core() -> bool:
    """Build C++ core library via CMake if CMake is available."""
    cmake_path = shutil.which("cmake")
    if not cmake_path:
        logger.warning("CMake not found in PATH; skipping C++ build step (using Python fallback).")
        return True

    logger.info("Building C++ core modules via CMake...")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [cmake_path, "-B", str(BUILD_DIR), "-S", str(REPO_ROOT), "-DCMAKE_BUILD_TYPE=Release"],
            check=True,
            cwd=str(REPO_ROOT),
        )
        subprocess.run(
            [cmake_path, "--build", str(BUILD_DIR), "--config", "Release", "-j4"],
            check=True,
            cwd=str(REPO_ROOT),
        )
        logger.info("C++ core successfully built.")
        return True
    except subprocess.CalledProcessError as err:
        logger.warning("C++ core build failed (%s); system will use ONNX/Python fallback.", err)
        return False


def run_pyinstaller(clean: bool = False) -> bool:
    """Execute PyInstaller to produce standalone deployment."""
    pyinstaller = shutil.which("pyinstaller")
    if not pyinstaller:
        logger.warning(
            "PyInstaller not found in current environment. "
            "Install it via: pip install pyinstaller"
        )
        return False

    cmd = [pyinstaller, "app.spec"]
    if clean:
        cmd.append("--clean")

    logger.info("Running PyInstaller: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        logger.info("Package created in: %s", DIST_DIR / "AerialWorkstation")
        return True
    except subprocess.CalledProcessError as err:
        logger.error("PyInstaller packaging failed: %s", err)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Standalone Distribution")
    parser.add_argument("--clean", action="store_true", help="Clean cache before build")
    parser.add_argument("--skip-cpp", action="store_true", help="Skip C++ compilation")
    parser.add_argument("--dry-run", action="store_true", help="Validate prerequisites only")
    args = parser.parse_args()

    logger.info("=== Starting Release Build Pipeline for %s ===", platform.system())

    if not check_prerequisites():
        return 1

    if args.dry_run:
        logger.info("Dry run complete. All assets and configurations are valid.")
        return 0

    if not args.skip_cpp:
        build_cpp_core()

    success = run_pyinstaller(clean=args.clean)
    if success:
        logger.info("=== Build Pipeline Finished Successfully ===")
        return 0
    else:
        logger.info("=== Build Finished with Warnings (PyInstaller skipped or failed) ===")
        return 0


if __name__ == "__main__":
    sys.exit(main())

