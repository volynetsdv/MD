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
import tarfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DIST_DIR = REPO_ROOT / "dist"
BUILD_DIR = REPO_ROOT / "build"


def check_prerequisites() -> bool:
    """Verify that required models and configuration exist."""
    models_dir = REPO_ROOT / "models"
    required_models = ["yolo_320.onnx", "yolo_416.onnx", "yolo_512.onnx", "yolo_640.onnx"]
    missing = [m for m in required_models if not (models_dir / m).exists()]
    if missing:
        logger.info("Missing model assets: %s. Ensuring models...", missing)
        try:
            from scripts.download_weights import ensure_models
            ensure_models(models_dir)
        except Exception as exc:
            logger.warning("Could not auto-fetch models: %s", exc)

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
            [
                cmake_path,
                "-B",
                str(BUILD_DIR),
                "-S",
                str(REPO_ROOT),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared",
            ],
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

    cmd = [pyinstaller, "app.spec", "--noconfirm"]
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


def _format_size(bytes_num: int) -> str:
    """Format bytes count into human-readable size."""
    size = float(bytes_num)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


MAX_SINGLE_ARCHIVE_BYTES: int = 1950 * 1024 * 1024  # 1950 MB safe limit for GitHub Releases (2048 MB max)
CHUNK_BYTES: int = 1800 * 1024 * 1024  # 1800 MB chunks when splitting is required


def _split_archive_if_needed(archive_path: Path) -> list[Path]:
    """Split archive into <= 1800 MB chunks if it exceeds GitHub Releases 2GB limit."""
    file_size = archive_path.stat().st_size
    if file_size <= MAX_SINGLE_ARCHIVE_BYTES:
        return [archive_path]

    logger.warning(
        "Archive %s (%s) exceeds GitHub Releases 2GB limit (threshold: 1950 MB). "
        "Splitting into %s chunks...",
        archive_path.name,
        _format_size(file_size),
        _format_size(CHUNK_BYTES),
    )

    part_paths: list[Path] = []
    part_idx = 0
    buffer_size = 64 * 1024 * 1024  # 64 MB buffer

    with open(archive_path, "rb") as src:
        while True:
            chunk_bytes_written = 0
            part_name = f"{archive_path.name}.part{part_idx:02d}"
            part_path = archive_path.parent / part_name
            if part_path.exists():
                part_path.unlink()

            with open(part_path, "wb") as dst:
                while chunk_bytes_written < CHUNK_BYTES:
                    to_read = min(buffer_size, CHUNK_BYTES - chunk_bytes_written)
                    data = src.read(to_read)
                    if not data:
                        break
                    dst.write(data)
                    chunk_bytes_written += len(data)

            if chunk_bytes_written > 0:
                part_paths.append(part_path)
                logger.info(
                    "Created archive chunk: %s (%s)",
                    part_path.name,
                    _format_size(part_path.stat().st_size),
                )
                part_idx += 1
            else:
                if part_path.exists():
                    part_path.unlink()
                break

    archive_path.unlink()
    logger.info(
        "Removed original oversized archive %s; generated %d part(s): %s",
        archive_path.name,
        len(part_paths),
        [p.name for p in part_paths],
    )
    return part_paths


def create_archive() -> Optional[list[Path]]:
    """Package dist/AerialWorkstation into a clean distribution archive with max compression."""
    app_dir = DIST_DIR / "AerialWorkstation"
    if not app_dir.exists():
        logger.error("Cannot archive: directory %s does not exist.", app_dir)
        return None

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Linux":
        arch_name = "x86_64" if machine in ("x86_64", "amd64") else machine
        archive_name = f"AerialWorkstation-Linux-{arch_name}.tar.gz"
        archive_path = DIST_DIR / archive_name
        for old_file in DIST_DIR.glob(f"{archive_name}*"):
            old_file.unlink()
        logger.info("Creating Linux release archive (compresslevel=9): %s ...", archive_path)
        with tarfile.open(archive_path, "w:gz", compresslevel=9) as tar:
            tar.add(app_dir, arcname="AerialWorkstation")
        logger.info(
            "Archive successfully created: %s (%s)",
            archive_path,
            _format_size(archive_path.stat().st_size),
        )
        return _split_archive_if_needed(archive_path)
    elif system == "Windows":
        arch_name = "x64" if machine in ("x86_64", "amd64") else machine
        archive_name = f"AerialWorkstation-Windows-{arch_name}.zip"
        archive_path = DIST_DIR / archive_name
        for old_file in DIST_DIR.glob(f"{archive_name}*"):
            old_file.unlink()
        logger.info("Creating Windows release archive: %s ...", archive_path)
        base_name = str(archive_path.with_suffix(""))
        shutil.make_archive(
            base_name,
            "zip",
            root_dir=str(DIST_DIR),
            base_dir="AerialWorkstation",
        )
        logger.info(
            "Archive successfully created: %s (%s)",
            archive_path,
            _format_size(archive_path.stat().st_size),
        )
        return _split_archive_if_needed(archive_path)
    else:
        archive_name = f"AerialWorkstation-{system}.tar.gz"
        archive_path = DIST_DIR / archive_name
        for old_file in DIST_DIR.glob(f"{archive_name}*"):
            old_file.unlink()
        with tarfile.open(archive_path, "w:gz", compresslevel=9) as tar:
            tar.add(app_dir, arcname="AerialWorkstation")
        return _split_archive_if_needed(archive_path)


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
    if not success:
        logger.error("=== Build Pipeline Failed (PyInstaller error) ===")
        return 1

    archives = create_archive()
    if not archives:
        logger.error("=== Build Pipeline Failed (Archive creation error) ===")
        return 1

    logger.info("=== Build Pipeline Finished Successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

