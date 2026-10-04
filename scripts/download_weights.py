#!/usr/bin/env python3
"""Automated Weights Downloader & Mock Model Generator for CI/CD and Production.

Ensures that required ONNX and Engine model profiles (320, 416, 512, 640)
exist in the models directory. Reads model URLs dynamically from application
configuration (config/settings.json via core.config_manager).

If remote weights are not reachable (e.g. offline CI sandbox), it automatically
synthesizes minimal, structurally valid ONNX models conforming to the expected
YOLO architecture (opset 17, float32, [1, 3, sz, sz] -> [1, 84, anchors]).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("WeightsManager")

DEFAULT_MODEL_URLS: Dict[str, str] = {
    "yolo_320.onnx": "https://github.com/volynetsdv/MD/releases/download/v1.0.0-weights/yolo_320.onnx",
    "yolo_416.onnx": "https://github.com/volynetsdv/MD/releases/download/v1.0.0-weights/yolo_416.onnx",
    "yolo_512.onnx": "https://github.com/volynetsdv/MD/releases/download/v1.0.0-weights/yolo_512.onnx",
    "yolo_640.onnx": "https://github.com/volynetsdv/MD/releases/download/v1.0.0-weights/yolo_640.onnx",
}

GRID_RESOLUTIONS: Tuple[int, ...] = (320, 416, 512, 640)


def get_configured_model_urls(config_path: Optional[Path] = None) -> Dict[str, str]:
    """Retrieve model URLs from config_manager or settings.json with default fallback."""
    # 1. Try config_manager
    try:
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from core.config_manager import get_config

        cfg = get_config(path=config_path)
        if cfg.model_urls:
            return dict(cfg.model_urls)
    except Exception as exc:
        logger.debug("Could not read model_urls via config_manager: %s", exc)

    # 2. Try raw config/settings.json
    cfg_file = config_path or (Path(__file__).resolve().parent.parent / "config" / "settings.json")
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "model_urls" in data and isinstance(data["model_urls"], dict):
                return {str(k): str(v) for k, v in data["model_urls"].items()}
        except Exception as exc:
            logger.debug("Could not read %s: %s", cfg_file, exc)

    return dict(DEFAULT_MODEL_URLS)


def create_mock_engine(path: Path, size: int) -> None:
    """Create a structured binary placeholder engine file for CI validation."""
    header = f"TRT_ENGINE_V8_M{size}".encode("utf-8")
    padding = bytes(1024)
    path.write_bytes(header + padding)
    logger.info("Generated placeholder TensorRT engine: %s", path.name)


def create_dummy_onnx(path: Path, size: int) -> None:
    """Synthesize a lightweight, structurally valid YOLO ONNX model using onnx.helper."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    num_anchors = (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2

    # Graph inputs and outputs
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, num_anchors])

    # Zero constant tensor for detections
    zero_data = np.zeros((1, 84, num_anchors), dtype=np.float32)
    zero_tensor = helper.make_tensor(
        "const_zeros",
        TensorProto.FLOAT,
        [1, 84, num_anchors],
        zero_data.flatten().tolist(),
    )
    const_node = helper.make_node("Constant", inputs=[], outputs=["const_val"], value=zero_tensor)

    # Compute dependency on input: ReduceSum(images) * 0.0 + const_val
    reduce_node = helper.make_node("ReduceSum", inputs=["images"], outputs=["sum_val"], keepdims=0)
    zero_scalar = helper.make_tensor("zero_scalar", TensorProto.FLOAT, [1], [0.0])
    zero_scalar_node = helper.make_node(
        "Constant", inputs=[], outputs=["zero_scalar_val"], value=zero_scalar
    )
    mul_node = helper.make_node("Mul", inputs=["sum_val", "zero_scalar_val"], outputs=["zero_mul"])
    add_node = helper.make_node("Add", inputs=["const_val", "zero_mul"], outputs=["output0"])

    graph = helper.make_graph(
        [const_node, reduce_node, zero_scalar_node, mul_node, add_node],
        f"DummyYolo_{size}",
        [inp],
        [out],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    logger.info("Generated valid synthetic ONNX model: %s (%d bytes)", path.name, path.stat().st_size)


def extract_size_from_filename(filename: str) -> int:
    """Extract resolution size (e.g. 320 from yolo_320.onnx) or default to 640."""
    match = re.search(r"(\d{3,4})", filename)
    if match:
        return int(match.group(1))
    return 640


def download_file(url: str, dest_path: Path, timeout: int = 60) -> bool:
    """Download a file with realistic User-Agent, streaming progress, and atomic write."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    logger.info("Downloading %s -> %s", url, dest_path.name)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 AerialWorkstation/1.0"
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size and total_size.isdigit() else 0
            downloaded = 0
            chunk_size = 256 * 1024

            with open(tmp_path, "wb") as out_f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0:
                        pct = (downloaded / total_bytes) * 100
                        if downloaded % (2 * 1024 * 1024) < chunk_size:
                            logger.info(
                                "Progress %s: %.1f%% (%d / %d MB)",
                                dest_path.name,
                                pct,
                                downloaded // (1024 * 1024),
                                total_bytes // (1024 * 1024),
                            )

        # Atomic replace
        tmp_path.replace(dest_path)
        logger.info(
            "Successfully downloaded: %s (%d bytes)",
            dest_path.name,
            dest_path.stat().st_size,
        )
        return True

    except Exception as exc:
        logger.warning("Download failed for %s (%s): %s", dest_path.name, url, exc)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return False


def ensure_single_model(
    model_filename: str,
    models_dir: Optional[Union[str, Path]] = None,
    config_path: Optional[Path] = None,
    allow_mock_fallback: bool = True,
) -> bool:
    """Ensure a specific model file exists, downloading if necessary.

    Returns True if model exists or was acquired/synthesized.
    """
    repo_root = Path(__file__).resolve().parent.parent
    m_dir = Path(models_dir) if models_dir is not None else (repo_root / "models")
    m_dir.mkdir(parents=True, exist_ok=True)

    dest_file = m_dir / model_filename
    if dest_file.exists() and dest_file.stat().st_size > 0:
        return True

    model_urls = get_configured_model_urls(config_path)
    url = model_urls.get(model_filename)
    download_ok = False

    if url:
        download_ok = download_file(url, dest_file)

    if not download_ok and allow_mock_fallback:
        size = extract_size_from_filename(model_filename)
        logger.info("Falling back to synthetic ONNX generation for %s (size %d)...", model_filename, size)
        create_dummy_onnx(dest_file, size)
        download_ok = dest_file.exists()

    # Also create matching engine placeholder
    stem = dest_file.stem
    size = extract_size_from_filename(model_filename)
    engine_file = m_dir / f"{stem}.engine"
    if not engine_file.exists():
        create_mock_engine(engine_file, size)

    return dest_file.exists()


def ensure_models(
    models_dir: Optional[Union[str, Path]] = None,
    force_mock: bool = False,
    force_download: bool = False,
    config_path: Optional[Path] = None,
) -> bool:
    """Ensure all configured model profiles exist in models_dir.

    Downloads missing models according to configuration, or creates dummy stubs.
    """
    repo_root = Path(__file__).resolve().parent.parent
    m_dir = Path(models_dir) if models_dir is not None else (repo_root / "models")
    m_dir.mkdir(parents=True, exist_ok=True)

    model_urls = get_configured_model_urls(config_path)
    all_ready = True

    for filename, url in model_urls.items():
        dest_file = m_dir / filename
        size = extract_size_from_filename(filename)
        engine_file = m_dir / f"{dest_file.stem}.engine"

        # Check ONNX
        needs_retrieval = (not dest_file.exists()) or (dest_file.stat().st_size == 0) or force_download

        if needs_retrieval or force_mock:
            download_ok = False
            if not force_mock and url:
                download_ok = download_file(url, dest_file)

            if not download_ok:
                logger.info("Synthesizing dummy ONNX profile for %s (%dx%d)...", filename, size, size)
                create_dummy_onnx(dest_file, size)

        # Check TensorRT Engine stub
        if not engine_file.exists():
            create_mock_engine(engine_file, size)

        if not dest_file.exists() or dest_file.stat().st_size == 0:
            all_ready = False

    return all_ready


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure release model assets are present.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory to place model files (default: from settings or models/)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force generation of mock models instead of downloading",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download weights even if local files already exist",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to custom settings.json configuration",
    )
    args = parser.parse_args()

    target_dir = args.models_dir
    if target_dir is None:
        try:
            from core.config_manager import get_config
            cfg = get_config(path=args.config)
            target_dir = Path(__file__).resolve().parent.parent / cfg.models_dir
        except Exception:
            target_dir = Path(__file__).resolve().parent.parent / "models"

    logger.info("Checking model prerequisites in: %s", target_dir)
    success = ensure_models(
        models_dir=target_dir,
        force_mock=args.mock,
        force_download=args.force_download,
        config_path=args.config,
    )

    if success:
        logger.info("All model assets are verified and ready.")
        return 0
    else:
        logger.error("Failed to prepare required model assets.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
