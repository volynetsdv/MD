"""Unified Inference Dispatcher with Cross-Platform Windows Fallback.

Provides a unified detection interface across C++ TensorRT and ONNX Runtime backends.
Ensures seamless operation on Windows 10/11 without requiring TensorRT SDK compilation
by automatically falling back to DirectML (DmlExecutionProvider), CUDA, or CPU.
"""

from __future__ import annotations

import enum
import logging
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Safe import of pytiling_core if compiled
_PyTilingDetection = None
try:
    import pytiling_core

    _PyTilingDetection = getattr(pytiling_core, "Detection", None)
except ImportError:
    repo_root = Path(__file__).resolve().parent.parent
    for candidate in [repo_root / "build", repo_root / "build" / "bindings"]:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            try:
                import pytiling_core

                _PyTilingDetection = getattr(pytiling_core, "Detection", None)
                break
            except ImportError:
                pass

logger = logging.getLogger("DetectorDispatcher")


# =============================================================================
# Unified Detection Data Structure
# =============================================================================
class Detection:
    """A single object detection candidate in tile-local coordinates (pixels).

    Seamlessly delegates to `pytiling_core.Detection` if compiled, or provides
    an identical pure-Python data structure for non-TensorRT / non-compiled environments.
    """

    __slots__ = ("x_local", "y_local", "w", "h", "conf", "class_id", "_cpp_inst")

    def __init__(
        self,
        x_local: float = 0.0,
        y_local: float = 0.0,
        w: float = 0.0,
        h: float = 0.0,
        conf: float = 0.0,
        class_id: int = 0,
    ) -> None:
        self.x_local = float(x_local)
        self.y_local = float(y_local)
        self.w = float(w)
        self.h = float(h)
        self.conf = float(conf)
        self.class_id = int(class_id)
        self._cpp_inst = None

    def to_cpp(self) -> Any:
        """Convert to pytiling_core.Detection if the C++ module is available."""
        if _PyTilingDetection is not None:
            d = _PyTilingDetection()
            d.x_local = self.x_local
            d.y_local = self.y_local
            d.w = self.w
            d.h = self.h
            d.conf = self.conf
            d.class_id = self.class_id
            return d
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary representation."""
        return {
            "x_local": self.x_local,
            "y_local": self.y_local,
            "w": self.w,
            "h": self.h,
            "conf": self.conf,
            "class_id": self.class_id,
        }

    def __repr__(self) -> str:
        return (
            f"<Detection x_local={self.x_local:.6f} y_local={self.y_local:.6f} "
            f"w={self.w:.6f} h={self.h:.6f} conf={self.conf:.6f} "
            f"class_id={self.class_id}>"
        )

    def __eq__(self, other: Any) -> bool:
        if not hasattr(other, "x_local"):
            return False
        return (
            abs(self.x_local - other.x_local) < 1e-4
            and abs(self.y_local - other.y_local) < 1e-4
            and abs(self.w - other.w) < 1e-4
            and abs(self.h - other.h) < 1e-4
            and abs(self.conf - other.conf) < 1e-4
            and self.class_id == other.class_id
        )


# =============================================================================
# Hardware & Backend Discovery
# =============================================================================
class BackendType(str, enum.Enum):
    """Inference execution backend type."""

    TENSORRT = "TensorRT"
    ONNXRUNTIME = "ONNXRuntime"


GRID_RESOLUTIONS: Tuple[int, ...] = (320, 416, 512, 640)


def check_tensorrt_available(models_dir: Optional[Path] = None) -> bool:
    """Check if TensorRT C++ SDK and NVIDIA GPU are present and functional.

    Looks for:
    1. C++ library artifacts (`libtrt_detector.so` or `trt_detector.pyd`).
    2. Python `tensorrt` module.
    3. CUDA-capable GPU.
    """
    # 1. Check for shared library / pyd
    repo_root = Path(__file__).resolve().parent.parent
    lib_candidates = [
        repo_root / "build" / "libtrt_detector.so",
        repo_root / "build" / "trt_detector.pyd",
        repo_root / "build" / "Release" / "trt_detector.pyd",
    ]
    has_lib = any(cand.exists() for cand in lib_candidates)

    # 2. Check for python tensorrt module
    has_py_trt = False
    try:
        import tensorrt as trt

        has_py_trt = trt is not None
    except ImportError:
        pass

    if not (has_lib or has_py_trt):
        return False

    # 3. Check for CUDA GPU availability
    has_cuda = False
    try:
        import torch

        has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        # Fallback to checking cuda driver
        has_cuda = os.path.exists("/dev/nvidia0") or sys.platform == "win32"

    return has_cuda


def is_windows_system() -> bool:
    """Check whether current OS is Windows."""
    return (platform.system() == "Windows") or (sys.platform == "win32")


def get_onnx_execution_providers(
    available_providers: Optional[List[str]] = None,
    is_windows: Optional[bool] = None,
) -> Tuple[List[str], str]:
    """Determine the optimal execution providers for ONNX Runtime.

    Priorities:
    - Windows 10/11: DmlExecutionProvider (DirectML) for universal GPU acceleration
      (NVIDIA, AMD, Intel, Qualcomm) without CUDA SDK dependency.
    - Linux / Other with CUDA: CUDAExecutionProvider.
    - CPU Fallback: CPUExecutionProvider with informative logging.

    Returns:
        Tuple of (provider_list, active_primary_provider_name)
    """
    if available_providers is None:
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
        except ImportError:
            available = ["CPUExecutionProvider"]
    else:
        available = list(available_providers)

    if is_windows is None:
        is_windows = is_windows_system()

    if is_windows:
        # On Windows, DirectML is the top priority for zero-install GPU execution
        if "DmlExecutionProvider" in available:
            logger.info("Hardware Discovery: DirectML (DmlExecutionProvider) selected for Windows GPU acceleration.")
            return ["DmlExecutionProvider", "CPUExecutionProvider"], "DmlExecutionProvider"
        if "CUDAExecutionProvider" in available:
            logger.info("Hardware Discovery: NVIDIA CUDA (CUDAExecutionProvider) selected on Windows.")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"
    else:
        # On Linux, CUDA is the top priority
        if "CUDAExecutionProvider" in available:
            logger.info("Hardware Discovery: NVIDIA CUDA (CUDAExecutionProvider) selected on Linux.")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"

    # CPU Fallback
    logger.info(
        "Hardware Discovery: No GPU execution provider active (DirectML/CUDA). "
        "Seamlessly falling back to CPUExecutionProvider."
    )
    return ["CPUExecutionProvider"], "CPUExecutionProvider"


# =============================================================================
# Fast NumPy Post-Processing (NMS)
# =============================================================================
def nms_fast(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Pure NumPy fast non-maximum suppression (vectorized).

    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Array of shape (N,) containing confidence scores.
        iou_threshold: Intersection-over-Union threshold.

    Returns:
        List of indices of surviving boxes.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-7)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


# =============================================================================
# Unified Detector
# =============================================================================
class UnifiedDetector:
    """Unified cross-platform detection engine with transparent hardware dispatch.

    Dispatches tile inference to either C++ TensorRT or ONNX Runtime (DirectML / CUDA / CPU)
    while exposing an identical, fail-safe predict_tile() API and caching model sessions.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        prefer_tensorrt: bool = True,
        force_provider: Optional[str] = None,
        is_windows: Optional[bool] = None,
        available_providers: Optional[List[str]] = None,
    ) -> None:
        """Initialize the unified detector and probe hardware environment.

        Args:
            models_dir: Directory containing .onnx and .engine models.
            prefer_tensorrt: If True, attempts TensorRT if available.
            force_provider: Explicitly specify an execution provider for testing/overrides
                            (e.g., 'DmlExecutionProvider', 'CPUExecutionProvider').
            is_windows: Optional override for Windows OS detection (testing).
            available_providers: Optional override for available execution providers list (testing).
        """
        if models_dir is None:
            self.models_dir = Path(__file__).resolve().parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)

        # Model session cache: {tile_size: session_or_engine}
        self._sessions: Dict[int, Any] = {}

        # 1. Probe hardware and determine backend
        self._force_provider = force_provider
        self._prefer_tensorrt = prefer_tensorrt
        self._is_windows_override = is_windows
        self._available_providers_override = available_providers

        self._discover_backend()

    def _discover_backend(self) -> None:
        """Evaluate hardware and configure inference backend and execution provider."""
        trt_available = False
        if self._prefer_tensorrt and not self._force_provider:
            trt_available = check_tensorrt_available(self.models_dir)

        if trt_available:
            self._backend = BackendType.TENSORRT
            self._execution_provider = "TensorRT"
            self._is_gpu_accelerated = True
            logger.info("UnifiedDetector: Initialized with C++ TensorRT Backend.")
        else:
            self._backend = BackendType.ONNXRUNTIME
            if self._force_provider:
                raw_providers = [self._force_provider, "CPUExecutionProvider"]
                self._providers = list(dict.fromkeys(raw_providers))
                self._execution_provider = self._force_provider
                self._is_gpu_accelerated = (
                    "Dml" in self._force_provider or "CUDA" in self._force_provider
                )
                logger.info("UnifiedDetector: Forced provider: %s", self._force_provider)
            else:
                provs, self._execution_provider = get_onnx_execution_providers(
                    available_providers=self._available_providers_override,
                    is_windows=self._is_windows_override,
                )
                self._providers = list(dict.fromkeys(provs))
                self._is_gpu_accelerated = (
                    "Dml" in self._execution_provider or "CUDA" in self._execution_provider
                )
            logger.info(
                "UnifiedDetector: Initialized with ONNX Runtime Backend (Provider: %s).",
                self._execution_provider,
            )

    @property
    def backend(self) -> BackendType:
        """Return active backend (TensorRT or ONNXRuntime)."""
        return self._backend

    @property
    def execution_provider(self) -> str:
        """Return active execution provider name (e.g. DmlExecutionProvider, CPUExecutionProvider)."""
        return self._execution_provider

    @property
    def is_gpu_accelerated(self) -> bool:
        """Check if hardware GPU acceleration is active."""
        return self._is_gpu_accelerated

    @property
    def cached_resolutions(self) -> List[int]:
        """Return list of currently cached model tile sizes."""
        return list(self._sessions.keys())

    # --------------------------------------------------------------------------
    # Model Pool Management & Session Caching
    # --------------------------------------------------------------------------
    def _snap_resolution(self, tile_size: int) -> int:
        """Snap requested tile size to the nearest supported model pool resolution."""
        return min(GRID_RESOLUTIONS, key=lambda s: abs(s - tile_size))

    def get_session(self, tile_size: int) -> Any:
        """Retrieve or create a cached inference session for the given tile resolution.

        Args:
            tile_size: Desired resolution (e.g. 320, 416, 512, 640).

        Returns:
            Cached ONNX Runtime InferenceSession or TensorRT engine wrapper.
        """
        snapped_size = self._snap_resolution(tile_size)

        if snapped_size in self._sessions:
            return self._sessions[snapped_size]

        # Load session for the snapped resolution
        if self._backend == BackendType.ONNXRUNTIME:
            session = self._create_onnx_session(snapped_size)
        else:
            session = self._create_tensorrt_session(snapped_size)

        self._sessions[snapped_size] = session
        logger.info(
            "UnifiedDetector: Cached new %s session for tile size %d.",
            self._backend.value,
            snapped_size,
        )
        return session

    def _create_onnx_session(self, size: int) -> Any:
        """Create and configure an ONNX Runtime InferenceSession."""
        import onnxruntime as ort

        model_path = self.models_dir / f"yolo_{size}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = min(4, os.cpu_count() or 1)

        try:
            session = ort.InferenceSession(
                str(model_path), sess_options, providers=self._providers
            )
        except Exception as err:
            # Safe fallback: if provider failed (e.g. Dml failure on unsupported device), fallback to CPU
            logger.warning(
                "UnifiedDetector: Provider %s failed (%s). Falling back to CPUExecutionProvider.",
                self._providers,
                err,
            )
            self._providers = ["CPUExecutionProvider"]
            self._execution_provider = "CPUExecutionProvider"
            self._is_gpu_accelerated = False
            session = ort.InferenceSession(
                str(model_path), sess_options, providers=self._providers
            )

        return session

    def _create_tensorrt_session(self, size: int) -> Any:
        """Create a TensorRT engine session or fall back to ONNX if engine is missing."""
        engine_path = self.models_dir / f"yolo_{size}.engine"
        if not engine_path.exists():
            logger.warning(
                "UnifiedDetector: TensorRT engine %s not found. Falling back to ONNX Runtime.",
                engine_path,
            )
            self._backend = BackendType.ONNXRUNTIME
            self._providers, self._execution_provider = get_onnx_execution_providers()
            return self._create_onnx_session(size)

        # Attempt to load TensorRT engine
        try:
            # Check if C++ wrapper or tensorrt is available
            import tensorrt as trt

            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(engine_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine.")
            context = engine.create_execution_context()
            return {"engine": engine, "context": context, "size": size}
        except Exception as exc:
            logger.warning(
                "UnifiedDetector: Failed to load TensorRT engine (%s). "
                "Transparently falling back to ONNX Runtime.",
                exc,
            )
            self._backend = BackendType.ONNXRUNTIME
            self._providers, self._execution_provider = get_onnx_execution_providers()
            return self._create_onnx_session(size)

    def preload_all_models(self) -> None:
        """Pre-warm and cache inference sessions for all supported resolutions."""
        for size in GRID_RESOLUTIONS:
            self.get_session(size)

    # --------------------------------------------------------------------------
    # Inference & Output Decoding
    # --------------------------------------------------------------------------
    def predict_tile(
        self,
        tile_tensor_or_numpy: Any,
        tile_size: Optional[int] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Detection]:
        """Run object detection on a single tile with identical cross-backend output.

        Args:
            tile_tensor_or_numpy: Input tile as NumPy array or PyTorch Tensor.
                Accepts:
                - (H, W, 3) uint8 or float32 [0..255] or [0..1]
                - (3, H, W) or (1, 3, H, W) float32
            tile_size: Target resolution model size. If None, inferred from input.
            conf_threshold: Minimum confidence score to accept detection.
            iou_threshold: IoU threshold for Non-Maximum Suppression.

        Returns:
            List of `Detection` instances in tile-local pixel coordinates.
        """
        # 1. Preprocess input to normalized float32 NCHW (1, 3, H, W)
        input_array, in_h, in_w = self._preprocess_input(tile_tensor_or_numpy)

        target_size = self._snap_resolution(tile_size or in_h)

        # Resize if dimensions differ from model input
        if (in_h != target_size) or (in_w != target_size):
            # Reshape: (1, 3, H, W) -> (H, W, 3) -> resize -> (1, 3, target_size, target_size)
            hwc = np.transpose(input_array[0], (1, 2, 0))
            resized = cv2.resize(
                hwc, (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )
            input_array = np.expand_dims(np.transpose(resized, (2, 0, 1)), axis=0)

        # 2. Execute inference
        session = self.get_session(target_size)
        raw_output = self._run_inference(session, input_array, target_size)

        # 3. Decode output and apply NMS
        detections = self._decode_yolo_output(
            raw_output=raw_output,
            model_size=target_size,
            original_size=(in_w, in_h),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        return detections

    def _preprocess_input(self, data: Any) -> Tuple[np.ndarray, int, int]:
        """Convert input data (numpy or torch tensor) into normalized float32 NCHW."""
        # Convert PyTorch tensor to NumPy
        if hasattr(data, "detach") and hasattr(data, "cpu"):
            data = data.detach().cpu().numpy()

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        # Standardize shape to (1, 3, H, W)
        if data.ndim == 3:
            # Could be (H, W, 3) or (3, H, W)
            if data.shape[2] in (1, 3, 4):  # HWC
                h, w = data.shape[0], data.shape[1]
                if data.shape[2] == 4:
                    data = data[:, :, :3]
                data = np.transpose(data, (2, 0, 1))  # to CHW
                data = np.expand_dims(data, axis=0)  # to NCHW
            else:  # CHW
                h, w = data.shape[1], data.shape[2]
                data = np.expand_dims(data, axis=0)
        elif data.ndim == 4:  # NCHW
            h, w = data.shape[2], data.shape[3]
        elif data.ndim == 2:  # Grayscale HW
            h, w = data.shape[0], data.shape[1]
            data = np.stack([data, data, data], axis=0)
            data = np.expand_dims(data, axis=0)
        else:
            raise ValueError(f"Unsupported input tensor dimensions: {data.shape}")

        # Normalize data to float32 [0.0, 1.0]
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        elif np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
            if data.max() > 1.5:  # [0..255] float
                data /= 255.0

        return data, h, w

    def _run_inference(self, session: Any, input_array: np.ndarray, size: int) -> np.ndarray:
        """Run forward pass through ONNX Runtime or TensorRT."""
        if self._backend == BackendType.ONNXRUNTIME:
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_array})
            return outputs[0]

        # TensorRT execution
        import torch

        engine = session["engine"]
        context = session["context"]

        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)

        d_input = torch.from_numpy(input_array).cuda()
        out_shape = engine.get_tensor_shape(output_name)
        d_output = torch.empty(tuple(out_shape), dtype=torch.float32, device="cuda")

        context.set_tensor_address(input_name, d_input.data_ptr())
        context.set_tensor_address(output_name, d_output.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        return d_output.cpu().numpy()

    def _decode_yolo_output(
        self,
        raw_output: np.ndarray,
        model_size: int,
        original_size: Tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
    ) -> List[Detection]:
        """Decode YOLO output tensor (1, 84, N) and apply NMS.

        Returns coordinates scaled back to the original tile dimensions.
        """
        # Ensure shape is (1, 84, N)
        if raw_output.ndim == 3 and raw_output.shape[2] == 84:
            raw_output = np.transpose(raw_output, (0, 2, 1))

        if raw_output.ndim != 3 or raw_output.shape[1] < 5:
            return []

        coords = raw_output[0, :4, :]  # (4, N) -> xc, yc, w, h
        class_scores = raw_output[0, 4:, :]  # (num_classes, N)

        # Max class confidence for each candidate
        class_ids = np.argmax(class_scores, axis=0)  # (N,)
        confs = np.max(class_scores, axis=0)  # (N,)

        # Filter by confidence threshold
        mask = confs >= conf_threshold
        if not np.any(mask):
            return []

        filtered_coords = coords[:, mask]  # (4, M)
        filtered_class_ids = class_ids[mask]  # (M,)
        filtered_confs = confs[mask]  # (M,)

        xc = filtered_coords[0]
        yc = filtered_coords[1]
        w = filtered_coords[2]
        h = filtered_coords[3]

        # Convert center (xc, yc, w, h) to top-left (x1, y1)
        x1 = xc - (w / 2.0)
        y1 = yc - (h / 2.0)
        x2 = x1 + w
        y2 = y1 + h

        # Rescale coordinates if model_size != original_size
        orig_w, orig_h = original_size
        if (orig_w != model_size) or (orig_h != model_size):
            scale_x = orig_w / float(model_size)
            scale_y = orig_h / float(model_size)
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y
            w *= scale_x
            h *= scale_y

        # Batched NMS (offset by class_id to preserve distinct classes)
        offset = filtered_class_ids.astype(np.float32) * 4096.0
        boxes_for_nms = np.stack(
            [x1 + offset, y1 + offset, x2 + offset, y2 + offset], axis=1
        )

        keep_indices = nms_fast(boxes_for_nms, filtered_confs, iou_threshold)

        # Construct final unified Detection objects
        detections: List[Detection] = []
        for idx in keep_indices:
            det = Detection(
                x_local=float(x1[idx]),
                y_local=float(y1[idx]),
                w=float(w[idx]),
                h=float(h[idx]),
                conf=float(filtered_confs[idx]),
                class_id=int(filtered_class_ids[idx]),
            )
            detections.append(det)

        return detections
