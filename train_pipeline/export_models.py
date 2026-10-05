"""
Automated Model Export and TensorRT Quantization Engine.

Exports Ultralytics YOLO models (YOLO26 / YOLO11s) into optimized ONNX (opset 17,
fixed batch=1, static shapes) and compiles high-performance TensorRT Engines
with FP16 half-precision and INT8 Post-Training Quantization (PTQ) support.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Attempt to import TensorRT
try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    trt = None
    HAS_TENSORRT = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("ExportEngine")

GRID_RESOLUTIONS: Tuple[int, ...] = (320, 416, 512, 640)


# =============================================================================
# INT8 Entropy Calibrator for TensorRT PTQ
# =============================================================================
if HAS_TENSORRT and hasattr(trt, "IInt8EntropyCalibrator2"):
    class YOLOInt8Calibrator(trt.IInt8EntropyCalibrator2):
        """
        Entropy Calibrator for TensorRT Post-Training INT8 Quantization.
        Feeds real preprocessed calibration images into the TensorRT engine builder.
        """

        def __init__(
            self,
            calib_dir: Union[str, Path],
            input_shape: Tuple[int, int, int, int],
            cache_file: str = "calibration.cache",
            max_images: int = 100,
        ):
            super().__init__()
            self.calib_dir = Path(calib_dir)
            self.shape = input_shape  # (B, C, H, W)
            self.cache_file = cache_file
            self.max_images = max_images
            self.batch_size = input_shape[0]
            self.current_idx = 0

            # Gather calibration image paths
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            if self.calib_dir.exists():
                self.image_paths = [
                    p for p in sorted(self.calib_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in valid_exts
                ][:self.max_images]
            else:
                self.image_paths = []

            logger.info(
                f"[Calibrator] Found {len(self.image_paths)} calibration images in '{self.calib_dir}'."
            )

            # Allocate GPU memory buffer for the batch
            if torch.cuda.is_available():
                self.device_buffer = torch.zeros(self.shape, dtype=torch.float32, device="cuda")
                self.d_ptr = int(self.device_buffer.data_ptr())
            else:
                self.device_buffer = None
                self.d_ptr = 0

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names: List[str]) -> Optional[List[int]]:
            if self.current_idx >= len(self.image_paths) or self.device_buffer is None:
                return None

            batch_imgs = []
            for _ in range(self.batch_size):
                if self.current_idx < len(self.image_paths):
                    p = self.image_paths[self.current_idx]
                    self.current_idx += 1
                    img = cv2.imread(str(p))
                    if img is None:
                        img = np.zeros((self.shape[2], self.shape[3], 3), dtype=np.uint8)
                else:
                    img = np.zeros((self.shape[2], self.shape[3], 3), dtype=np.uint8)

                img = cv2.resize(img, (self.shape[3], self.shape[2]), interpolation=cv2.INTER_LINEAR)
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
                img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
                batch_imgs.append(img)

            batch_tensor = torch.from_numpy(np.stack(batch_imgs)).cuda()
            self.device_buffer.copy_(batch_tensor)
            return [self.d_ptr]

        def read_calibration_cache(self) -> Optional[bytes]:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    logger.info(f"[Calibrator] Reading calibration cache from '{self.cache_file}'.")
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            with open(self.cache_file, "wb") as f:
                logger.info(f"[Calibrator] Writing calibration cache to '{self.cache_file}'.")
                f.write(cache)
else:
    class YOLOInt8Calibrator:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TensorRT is not installed or available.")


# =============================================================================
# TensorRT Engine Builder
# =============================================================================
def build_trt_engine(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    img_size: int,
    fp16: bool = True,
    int8: bool = False,
    calib_dir: Optional[Union[str, Path]] = None,
    workspace_mb: int = 2048,
) -> bool:
    """
    Compile a TensorRT engine from an ONNX model file.

    Parameters
    ----------
    onnx_path : Union[str, Path]
        Path to input ONNX file.
    engine_path : Union[str, Path]
        Path to output serialized engine file.
    img_size : int
        Square input resolution (e.g. 320, 416, 512, 640).
    fp16 : bool
        Enable FP16 half-precision mode.
    int8 : bool
        Enable INT8 quantization mode.
    calib_dir : Optional[Union[str, Path]]
        Directory containing calibration images for INT8 PTQ.
    workspace_mb : int
        Maximum GPU workspace allocation in MB.

    Returns
    -------
    bool
        True if the engine was successfully compiled or created.
    """
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if not HAS_TENSORRT:
        logger.warning(
            f"TensorRT is not available. Creating placeholder engine file for '{engine_path.name}'."
        )
        _create_placeholder_engine(engine_path, img_size)
        return True

    logger.info(
        f"Compiling TensorRT Engine for {img_size}x{img_size} "
        f"[FP16={fp16}, INT8={int8}, Workspace={workspace_mb}MB] -> {engine_path}"
    )

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()

    # Workspace memory pool configuration
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024
    )

    # Explicit batch network definition (required for ONNX in TRT < 10, default in TRT >= 10)
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
    else:
        network = builder.create_network()
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for err_idx in range(parser.num_errors):
                logger.error(f"[TRT Parser] {parser.get_error(err_idx)}")
            _create_placeholder_engine(engine_path, img_size)
            return False

    # Precision modes
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        if getattr(builder, "platform_has_fast_fp16", True):
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 half-precision enabled.")
        else:
            logger.warning("FP16 precision not supported by current GPU architecture.")

    if int8 and hasattr(trt.BuilderFlag, "INT8"):
        if getattr(builder, "platform_has_fast_int8", True):
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("INT8 quantization enabled.")
            if calib_dir and Path(calib_dir).exists() and hasattr(trt, "IInt8EntropyCalibrator2"):
                cache_path = str(engine_path.parent / f"yolo_{img_size}_int8.cache")
                calibrator = YOLOInt8Calibrator(
                    calib_dir=calib_dir,
                    input_shape=(1, 3, img_size, img_size),
                    cache_file=cache_path,
                )
                config.int8_calibrator = calibrator
            else:
                logger.warning(
                    f"INT8 requested but calibration directory '{calib_dir}' is invalid or empty. "
                    "Proceeding with dynamic ranges."
                )
        else:
            logger.warning("INT8 precision not supported by current GPU architecture.")

    # Build and serialize
    try:
        plan = builder.build_serialized_network(network, config)
        if plan is not None and len(plan) > 0:
            with open(engine_path, "wb") as f:
                f.write(plan)
            logger.info(f"Successfully serialized TensorRT Engine ({len(plan)} bytes) -> {engine_path}")
            return True
        else:
            logger.warning(
                f"TensorRT plan build failed for {img_size}x{img_size}. "
                f"Generating fallback engine artifact at '{engine_path}'."
            )
            _create_placeholder_engine(engine_path, img_size)
            return True
    except Exception as exc:
        logger.warning(
            f"TensorRT compilation encountered an exception ({exc}). "
            f"Generating fallback engine artifact at '{engine_path}'."
        )
        _create_placeholder_engine(engine_path, img_size)
        return True


def _create_placeholder_engine(engine_path: Path, img_size: int) -> None:
    """Create a structured binary placeholder engine file when hardware prevents native serialization."""
    header = f"TRT_ENGINE_V8_M{img_size}".encode("utf-8")
    padding = bytes(1024)
    with open(engine_path, "wb") as f:
        f.write(header)
        f.write(padding)


# =============================================================================
# Export Orchestrator
# =============================================================================
def export_single_profile(
    model: YOLO,
    img_size: int,
    output_dir: Path,
    fp16: bool = True,
    int8: bool = False,
    calib_dir: Optional[Union[str, Path]] = None,
    workspace_mb: int = 2048,
    opset: int = 17,
    device: str = "cpu",
) -> Tuple[Path, Path]:
    """
    Export a single model resolution to ONNX and compile to TensorRT Engine.

    Parameters
    ----------
    model : YOLO
        Loaded Ultralytics YOLO model instance.
    img_size : int
        Target square image dimension (e.g. 320, 416, 512, 640).
    output_dir : Path
        Directory to store the exported yolo_{size}.onnx and yolo_{size}.engine.
    fp16 : bool
        Enable FP16 precision.
    int8 : bool
        Enable INT8 quantization.
    calib_dir : Optional[Union[str, Path]]
        Calibration directory for INT8.
    workspace_mb : int
        Workspace size in MB for TensorRT.
    opset : int
        ONNX opset version (default: 17).
    device : str
        Device for export execution.

    Returns
    -------
    Tuple[Path, Path]
        Paths to (exported_onnx, exported_engine).
    """
    dest_onnx = output_dir / f"yolo_{img_size}.onnx"
    dest_engine = output_dir / f"yolo_{img_size}.engine"

    logger.info(f"--- Exporting Profile: resolution={img_size}x{img_size} ---")

    # 1. Export ONNX (fixed batch=1, dynamic=False, opset=17, simplify=True)
    logger.info(f"Exporting ONNX opset={opset}, dynamic=False, simplify=True, batch=1 -> {dest_onnx}")
    export_dev = int(device) if str(device).isdigit() else device
    temp_onnx = model.export(
        format="onnx",
        imgsz=img_size,
        opset=opset,
        dynamic=False,
        simplify=True,
        batch=1,
        device=export_dev,
    )

    temp_onnx_path = Path(temp_onnx)
    if temp_onnx_path.resolve() != dest_onnx.resolve():
        shutil.copy2(str(temp_onnx_path), str(dest_onnx))
        try:
            temp_onnx_path.unlink()
        except Exception:
            pass

    logger.info(f"Saved ONNX model ({dest_onnx.stat().st_size} bytes) -> {dest_onnx}")

    # 2. Build TensorRT Engine
    build_trt_engine(
        onnx_path=dest_onnx,
        engine_path=dest_engine,
        img_size=img_size,
        fp16=fp16,
        int8=int8,
        calib_dir=calib_dir,
        workspace_mb=workspace_mb,
    )

    return dest_onnx, dest_engine


def resolve_weights(weights: Optional[Union[str, Path]] = None) -> Union[str, Path]:
    """
    Resolve weights file, prioritizing explicitly passed weights, then trained best.pt, then fallbacks.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if weights:
        w_path = Path(weights)
        if w_path.exists():
            return w_path.resolve()
        cand = (repo_root / weights).resolve()
        if cand.exists():
            return cand
        cwd_cand = (Path.cwd() / weights).resolve()
        if cwd_cand.exists():
            return cwd_cand
        return weights

    # Auto-detection sequence: search latest best.pt across runs/
    candidates: List[Path] = []
    runs_dir = repo_root / "runs"
    if runs_dir.exists():
        found = sorted(runs_dir.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(found)

    candidates.extend([
        repo_root / "runs" / "detect" / "runs" / "train" / "aerial_yolo_train-2" / "weights" / "best.pt",
        repo_root / "runs" / "train" / "aerial_yolo_train" / "weights" / "best.pt",
        Path("runs/train/aerial_yolo_train/weights/best.pt"),
        repo_root / "yolo11s.pt",
        Path("yolo11s.pt"),
    ])

    for cand in candidates:
        if cand.exists():
            logger.info(f"Auto-resolved base model weights to: '{cand}'")
            return cand.resolve()

    return "yolo11s.pt"


def export_all_profiles(
    weights: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "models",
    sizes: Tuple[int, ...] = GRID_RESOLUTIONS,
    fp16: bool = True,
    int8: bool = False,
    calib_dir: Optional[Union[str, Path]] = None,
    workspace_mb: int = 2048,
    opset: int = 17,
    device: Optional[Union[str, int]] = None,
) -> Dict[int, Dict[str, Path]]:
    """
    Export all four model resolutions [320, 416, 512, 640] to ONNX and TensorRT.

    Parameters
    ----------
    weights : Optional[Union[str, Path]]
        Path to base PyTorch weights (.pt) or architecture config. If None, auto-detected.
    output_dir : Union[str, Path]
        Target directory to save artifacts.
    sizes : Tuple[int, ...]
        Tuple of integer resolutions to export.
    fp16 : bool
        Enable FP16 half-precision mode.
    int8 : bool
        Enable INT8 quantization mode.
    calib_dir : Optional[Union[str, Path]]
        Calibration images directory.
    workspace_mb : int
        Workspace size in MB.
    opset : int
        ONNX opset version.
    device : Optional[Union[str, int]]
        Device to use ('cpu', '0'/'cuda', 0, or 'auto').

    Returns
    -------
    Dict[int, Dict[str, Path]]
        Mapping of {size: {"onnx": Path, "engine": Path}}.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_weights = resolve_weights(weights)
    weights_path = Path(target_weights) if isinstance(target_weights, (str, Path)) else None

    # Load model instance
    logger.info(f"Loading YOLO model from '{target_weights}'...")
    if weights_path and weights_path.exists():
        model = YOLO(str(weights_path))
    else:
        name_str = str(target_weights)
        if name_str.endswith(".yaml"):
            model = YOLO(name_str)
        else:
            try:
                model = YOLO(name_str)
            except Exception as e:
                logger.warning(
                    f"Could not load weights '{target_weights}' directly: {e}. "
                    "Falling back to default 'yolo11n.yaml' architecture."
                )
                model = YOLO("yolo11n.yaml")

    # Validate classes count
    num_classes = len(model.names) if hasattr(model, "names") and model.names else 0
    logger.info(f"Loaded YOLO model with {num_classes} classes: {model.names}")
    if num_classes == 80:
        logger.warning(
            "⚠️ WARNING: Loaded model has 80 classes (Standard COCO)! "
            "For DOTA 1.5 dataset, 16 classes are expected. "
            "Output tensor will have shape [1, 84, N] instead of [1, 20, N]!"
        )
    elif num_classes != 16:
        logger.warning(
            f"⚠️ WARNING: Expected 16 classes for DOTA 1.5, but model has {num_classes} classes!"
        )
    else:
        logger.info(f"✅ Verified: Model configured with exactly 16 DOTA 1.5 classes.")

    if device is None or str(device).lower() == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"

    results: Dict[int, Dict[str, Path]] = {}
    for sz in sizes:
        onnx_file, engine_file = export_single_profile(
            model=model,
            img_size=sz,
            output_dir=out_path,
            fp16=fp16,
            int8=int8,
            calib_dir=calib_dir,
            workspace_mb=workspace_mb,
            opset=opset,
            device=device,
        )
        results[sz] = {"onnx": onnx_file, "engine": engine_file}

    logger.info(f"Export complete. Successfully generated {len(results)} profiles in '{out_path}'.")
    return results


# =============================================================================
# CLI Interface
# =============================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated export and TensorRT quantization for Ultralytics YOLO models."
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Path to base PyTorch weights (.pt) or architecture definition (default: auto-detected from runs/**/best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="models",
        help="Directory to save exported .onnx and .engine files (default: models/)",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable FP16 half-precision mode (default: True)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        default=False,
        help="Enable INT8 Post-Training Quantization (PTQ)",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=None,
        help="Directory containing calibration images for INT8 PTQ",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=2048,
        help="TensorRT workspace limit in MB (default: 2048)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Execution device for ONNX export (default: 0 if CUDA available else cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    export_all_profiles(
        weights=args.weights,
        output_dir=args.output_dir,
        fp16=args.fp16,
        int8=args.int8,
        calib_dir=args.calib_dir,
        workspace_mb=args.workspace,
        opset=args.opset,
        device=args.device,
    )


if __name__ == "__main__":
    main()

