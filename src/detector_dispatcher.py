"""Unified Inference Dispatcher with Cross-Platform Windows Fallback.

Provides a unified detection interface across C++ TensorRT and ONNX Runtime backends.
Ensures seamless operation on Windows 10/11 without requiring TensorRT SDK compilation
by automatically falling back to DirectML (DmlExecutionProvider), CUDA, or CPU.
"""

from __future__ import annotations

import abc
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
    for candidate in [repo_root / "build", repo_root / "build" / "Release", repo_root / "build" / "bindings"]:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            try:
                import pytiling_core

                _PyTilingDetection = getattr(pytiling_core, "Detection", None)
                break
            except ImportError:
                pass

logger = logging.getLogger("DetectorDispatcher")

# Ensure NVIDIA cuDNN runtime libraries are accessible by ONNX Runtime if installed via pip
try:
    for _p in sys.path:
        _cudnn_dir = Path(_p) / "nvidia" / "cudnn" / "lib"
        if _cudnn_dir.exists():
            import ctypes
            for _lib_name in ["libcudnn.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9", "libcudnn_adv.so.9"]:
                _lib_path = _cudnn_dir / _lib_name
                if _lib_path.exists():
                    try:
                        ctypes.CDLL(str(_lib_path), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass
except Exception:
    pass


# =============================================================================
# Default & Fallback Model Class Names (DOTA 1.5 - 16 Classes)
# =============================================================================
DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "Літак (plane)",
    1: "Судно / Корабель (ship)",
    2: "Резервуар (storage tank)",
    3: "Бейсбольне поле (baseball diamond)",
    4: "Тенісний корт (tennis court)",
    5: "Баскетбольний майданчик (basketball court)",
    6: "Бігова доріжка / Стадіон (ground track field)",
    7: "Гавань / Порт (harbor)",
    8: "Міст (bridge)",
    9: "Великогабаритний транспорт (large vehicle)",
    10: "Малогабаритний транспорт (small vehicle)",
    11: "Гелікоптер (helicopter)",
    12: "Кільцева розв'язка (roundabout)",
    13: "Футбольне поле (soccer ball field)",
    14: "Басейн (swimming pool)",
    15: "Портовий кран (container crane)",
}


def _parse_names_metadata(raw_names: Any) -> Optional[Dict[int, str]]:
    """Parse YOLO names metadata from string, dict, or list representation."""
    if not raw_names:
        return None
    parsed: Any = None
    if isinstance(raw_names, dict):
        parsed = raw_names
    elif isinstance(raw_names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(raw_names)}
    elif isinstance(raw_names, str):
        try:
            import json

            parsed = json.loads(raw_names)
        except Exception:
            pass
        if parsed is None:
            try:
                import ast

                parsed = ast.literal_eval(raw_names)
            except Exception:
                pass
        if parsed is None:
            try:
                import yaml

                parsed = yaml.safe_load(raw_names)
            except Exception:
                pass

    if isinstance(parsed, dict) and parsed:
        return {int(k): str(v) for k, v in parsed.items()}
    elif isinstance(parsed, (list, tuple)) and parsed:
        return {i: str(v) for i, v in enumerate(parsed)}
    return None


def _try_load_yaml_class_names(models_dir: Path) -> Optional[Dict[int, str]]:
    """Attempt loading class names from dataset.yaml in models_dir or common data dirs."""
    candidates = [
        models_dir / "dataset.yaml",
        models_dir.parent / "data" / "sliced_dota" / "dataset.yaml",
        models_dir.parent / "data" / "dota_sliced.yaml",
    ]
    for c in candidates:
        if c.exists():
            try:
                import yaml

                with open(c, "r", encoding="utf-8") as f:
                    ydata = yaml.safe_load(f)
                if isinstance(ydata, dict) and "names" in ydata:
                    res = _parse_names_metadata(ydata["names"])
                    if res:
                        return res
            except Exception:
                pass
    return None


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


MIN_ENGINE_SIZE_BYTES: int = 1048576  # 1 MB


def check_tensorrt_available(models_dir: Optional[Path] = None) -> bool:
    """Check if TensorRT C++ SDK, NVIDIA GPU, and valid .engine plans are present.

    Looks for:
    1. C++ library artifacts (`libtrt_detector.so` or `trt_detector.pyd`).
    2. Python `tensorrt` module.
    3. CUDA-capable GPU.
    4. Valid compiled .engine plan files (>= 1MB).
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

    if not has_cuda:
        return False

    # 4. Check for valid compiled TensorRT engine files (>= 1MB)
    m_dir = Path(models_dir) if models_dir is not None else repo_root / "models"
    if m_dir.exists():
        engine_files = list(m_dir.glob("*.engine"))
        if engine_files and not any(f.stat().st_size >= MIN_ENGINE_SIZE_BYTES for f in engine_files):
            logger.info("UnifiedDetector: All .engine files are stubs (<1MB). Bypassing TensorRT.")
            return False

    return True


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
# Inference Backend Strategy Hierarchy
# =============================================================================
class InferenceBackend(abc.ABC):
    """Abstract Strategy base class for inference backend engines."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self._sessions: Dict[int, Any] = {}

    @property
    @abc.abstractmethod
    def name(self) -> BackendType:
        """Return backend type (TensorRT or ONNXRuntime)."""
        ...

    @property
    @abc.abstractmethod
    def execution_provider(self) -> str:
        """Return active execution provider identifier."""
        ...

    @property
    @abc.abstractmethod
    def is_gpu_accelerated(self) -> bool:
        """Indicate whether the backend leverages hardware GPU acceleration."""
        ...

    @property
    def providers(self) -> List[str]:
        """Return execution providers list for this backend."""
        return []

    @property
    def cached_resolutions(self) -> List[int]:
        """Return tile resolutions currently cached in memory."""
        return list(self._sessions.keys())

    @abc.abstractmethod
    def get_session(self, size: int) -> Any:
        """Retrieve or instantiate an inference session for the given resolution."""
        ...

    @abc.abstractmethod
    def run_inference(self, session: Any, input_array: np.ndarray, size: int) -> np.ndarray:
        """Execute forward pass on normalized NCHW float32 input."""
        ...

    @abc.abstractmethod
    def get_class_names(self) -> Dict[int, str]:
        """Retrieve class ID to name dictionary from active model metadata or fallback."""
        ...


class TensorRTBackend(InferenceBackend):
    """Inference strategy utilizing NVIDIA TensorRT C++ SDK execution plans (.engine)."""

    @property
    def name(self) -> BackendType:
        return BackendType.TENSORRT

    @property
    def execution_provider(self) -> str:
        return "TensorRT"

    @property
    def is_gpu_accelerated(self) -> bool:
        return True

    def get_class_names(self) -> Dict[int, str]:
        """Attempt to extract class names from companion ONNX model or dataset.yaml, else fallback."""
        for sz in (640, 512, 416, 320):
            onnx_path = self.models_dir / f"yolo_{sz}.onnx"
            if onnx_path.exists():
                try:
                    import onnxruntime as ort

                    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                    meta = sess.get_modelmeta().custom_metadata_map
                    if meta and "names" in meta:
                        parsed = _parse_names_metadata(meta["names"])
                        if parsed:
                            return parsed
                except Exception:
                    pass

        yaml_names = _try_load_yaml_class_names(self.models_dir)
        if yaml_names:
            return yaml_names

        return dict(DEFAULT_CLASS_NAMES)

    def get_session(self, size: int) -> Any:
        """Load and cache a TensorRT engine execution context."""
        if size in self._sessions:
            return self._sessions[size]

        engine_path = self.models_dir / f"yolo_{size}.engine"
        if not engine_path.exists() or engine_path.stat().st_size < MIN_ENGINE_SIZE_BYTES:
            raise FileNotFoundError(
                f"TensorRT engine {engine_path} not found or invalid size (< 1MB)."
            )

        import tensorrt as trt

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        context = engine.create_execution_context()
        session = {"engine": engine, "context": context, "size": size}
        self._sessions[size] = session
        logger.info("TensorRTBackend: Cached session for resolution %d.", size)
        return session

    def run_inference(self, session: Any, input_array: np.ndarray, size: int) -> np.ndarray:
        """Execute TensorRT forward pass directly in VRAM without host copies."""
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


class ONNXRuntimeBackendBase(InferenceBackend):
    """Base inference strategy for ONNX Runtime execution providers."""

    def __init__(
        self,
        models_dir: Path,
        providers: Optional[List[str]] = None,
    ) -> None:
        super().__init__(models_dir)
        self._providers: List[str] = list(providers) if providers else ["CPUExecutionProvider"]

    @property
    def name(self) -> BackendType:
        return BackendType.ONNXRUNTIME

    @property
    def providers(self) -> List[str]:
        return self._providers

    def get_class_names(self) -> Dict[int, str]:
        """Extract class names from ONNX model metadata with fallback."""
        session = None
        for sz in (640, 512, 416, 320):
            if sz in self._sessions:
                session = self._sessions[sz]
                break
        if session is None:
            for sz in (640, 512, 416, 320):
                try:
                    session = self.get_session(sz)
                    break
                except Exception:
                    continue

        if session is not None and hasattr(session, "get_modelmeta"):
            try:
                meta = session.get_modelmeta().custom_metadata_map
                if meta and "names" in meta:
                    parsed = _parse_names_metadata(meta["names"])
                    if parsed:
                        return parsed
            except Exception as exc:
                logger.debug("Failed to read class names from ONNX metadata: %s", exc)

        yaml_names = _try_load_yaml_class_names(self.models_dir)
        if yaml_names:
            return yaml_names

        return dict(DEFAULT_CLASS_NAMES)

    def get_session(self, size: int) -> Any:
        """Create and cache an ONNX Runtime InferenceSession."""
        if size in self._sessions:
            return self._sessions[size]

        import onnxruntime as ort

        model_path = self.models_dir / f"yolo_{size}.onnx"
        if not model_path.exists():
            logger.info(
                "ONNX model %s not found on disk. Attempting automatic download from configured model_urls...",
                model_path.name,
            )
            try:
                from scripts.download_weights import ensure_single_model

                ensure_single_model(model_path.name, self.models_dir)
            except Exception as exc:
                logger.warning("Failed to automatically acquire model %s: %s", model_path.name, exc)

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
            logger.warning(
                "UnifiedDetector: Provider %s failed (%s). Falling back to CPUExecutionProvider.",
                self._providers,
                err,
            )
            self._providers = ["CPUExecutionProvider"]
            session = ort.InferenceSession(
                str(model_path), sess_options, providers=self._providers
            )

        self._sessions[size] = session
        logger.info(
            "ONNXRuntime (%s): Cached session for tile size %d.",
            self.execution_provider,
            size,
        )
        return session

    def run_inference(self, session: Any, input_array: np.ndarray, size: int) -> np.ndarray:
        """Run forward pass via ONNX Runtime."""
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_array})
        return outputs[0]


class DirectMLBackend(ONNXRuntimeBackendBase):
    """DirectML (DmlExecutionProvider) backend strategy for universal Windows GPU acceleration."""

    def __init__(
        self,
        models_dir: Path,
        custom_providers: Optional[List[str]] = None,
    ) -> None:
        provs = custom_providers or ["DmlExecutionProvider", "CPUExecutionProvider"]
        super().__init__(models_dir, providers=provs)

    @property
    def execution_provider(self) -> str:
        return "DmlExecutionProvider"

    @property
    def is_gpu_accelerated(self) -> bool:
        return True


class CUDAExecutionProviderBackend(ONNXRuntimeBackendBase):
    """NVIDIA CUDA (CUDAExecutionProvider) backend strategy for ONNX Runtime."""

    def __init__(
        self,
        models_dir: Path,
        custom_providers: Optional[List[str]] = None,
    ) -> None:
        provs = custom_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        super().__init__(models_dir, providers=provs)

    @property
    def execution_provider(self) -> str:
        return "CUDAExecutionProvider"

    @property
    def is_gpu_accelerated(self) -> bool:
        return True


class CPUExecutionProviderBackend(ONNXRuntimeBackendBase):
    """CPU execution provider strategy for deterministic portable inference."""

    def __init__(
        self,
        models_dir: Path,
        custom_providers: Optional[List[str]] = None,
    ) -> None:
        provs = custom_providers or ["CPUExecutionProvider"]
        super().__init__(models_dir, providers=provs)

    @property
    def execution_provider(self) -> str:
        return "CPUExecutionProvider"

    @property
    def is_gpu_accelerated(self) -> bool:
        return False


# =============================================================================
# Backend Factory
# =============================================================================
class BackendFactory:
    """Factory for polymorphic selection and instantiation of inference backends."""

    @staticmethod
    def create(
        models_dir: Path,
        prefer_tensorrt: bool = True,
        force_provider: Optional[str] = None,
        is_windows: Optional[bool] = None,
        available_providers: Optional[List[str]] = None,
    ) -> InferenceBackend:
        """Instantiate the optimal InferenceBackend based on available hardware and OS."""
        if prefer_tensorrt and not force_provider and check_tensorrt_available(models_dir):
            logger.info("BackendFactory: Selected TensorRTBackend.")
            return TensorRTBackend(models_dir)

        if force_provider:
            raw_providers = [force_provider, "CPUExecutionProvider"]
            clean_providers = list(dict.fromkeys(raw_providers))
            if "Dml" in force_provider:
                logger.info("BackendFactory: Forced DirectMLBackend (%s).", force_provider)
                return DirectMLBackend(models_dir, custom_providers=clean_providers)
            if "CUDA" in force_provider:
                logger.info("BackendFactory: Forced CUDAExecutionProviderBackend (%s).", force_provider)
                return CUDAExecutionProviderBackend(models_dir, custom_providers=clean_providers)
            logger.info("BackendFactory: Forced CPUExecutionProviderBackend (%s).", force_provider)
            return CPUExecutionProviderBackend(models_dir, custom_providers=clean_providers)

        provs, active_provider = get_onnx_execution_providers(
            available_providers=available_providers,
            is_windows=is_windows,
        )
        if active_provider == "DmlExecutionProvider":
            logger.info("BackendFactory: Selected DirectMLBackend.")
            return DirectMLBackend(models_dir, custom_providers=provs)
        if active_provider == "CUDAExecutionProvider":
            logger.info("BackendFactory: Selected CUDAExecutionProviderBackend.")
            return CUDAExecutionProviderBackend(models_dir, custom_providers=provs)
        logger.info("BackendFactory: Selected CPUExecutionProviderBackend.")
        return CPUExecutionProviderBackend(models_dir, custom_providers=provs)


# =============================================================================
# Unified Detector
# =============================================================================
class UnifiedDetector:
    """Unified cross-platform detection engine with transparent hardware dispatch.

    Dispatches tile inference to either C++ TensorRT or ONNX Runtime (DirectML / CUDA / CPU)
    via polymorphic InferenceBackend strategies while exposing an identical, fail-safe API.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        prefer_tensorrt: bool = True,
        force_provider: Optional[str] = None,
        is_windows: Optional[bool] = None,
        available_providers: Optional[List[str]] = None,
    ) -> None:
        """Initialize the unified detector and instantiate the optimal backend strategy.

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

        self._force_provider = force_provider
        self._prefer_tensorrt = prefer_tensorrt
        self._is_windows_override = is_windows
        self._available_providers_override = available_providers

        # Strategy instance
        self._backend_strategy: InferenceBackend = BackendFactory.create(
            models_dir=self.models_dir,
            prefer_tensorrt=prefer_tensorrt,
            force_provider=force_provider,
            is_windows=is_windows,
            available_providers=available_providers,
        )

    @property
    def backend_strategy(self) -> InferenceBackend:
        """Return the active backend strategy instance."""
        return self._backend_strategy

    @property
    def backend(self) -> BackendType:
        """Return active backend enum (TensorRT or ONNXRuntime)."""
        return self._backend_strategy.name

    @property
    def execution_provider(self) -> str:
        """Return active execution provider name (e.g. DmlExecutionProvider, CPUExecutionProvider)."""
        return self._backend_strategy.execution_provider

    @property
    def is_gpu_accelerated(self) -> bool:
        """Check if hardware GPU acceleration is active."""
        return self._backend_strategy.is_gpu_accelerated

    @property
    def cached_resolutions(self) -> List[int]:
        """Return list of currently cached model tile sizes."""
        return self._backend_strategy.cached_resolutions

    @property
    def _providers(self) -> List[str]:
        """Return execution provider list of the active backend."""
        return self._backend_strategy.providers

    @property
    def _sessions(self) -> Dict[int, Any]:
        """Return model session cache dictionary."""
        return self._backend_strategy._sessions

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

        try:
            return self._backend_strategy.get_session(snapped_size)
        except Exception as exc:
            if isinstance(self._backend_strategy, TensorRTBackend):
                logger.warning(
                    "UnifiedDetector: TensorRT engine session failed (%s). "
                    "Transparently falling back to ONNX Runtime.",
                    exc,
                )
                self._backend_strategy = BackendFactory.create(
                    models_dir=self.models_dir,
                    prefer_tensorrt=False,
                    force_provider=self._force_provider,
                    is_windows=self._is_windows_override,
                    available_providers=self._available_providers_override,
                )
                return self._backend_strategy.get_session(snapped_size)
            raise

    def preload_all_models(self) -> None:
        """Pre-warm and cache inference sessions for all supported resolutions."""
        for size in GRID_RESOLUTIONS:
            self.get_session(size)

    def get_class_names(self) -> Dict[int, str]:
        """Retrieve class ID to name dictionary from active backend with fallback to DEFAULT_CLASS_NAMES."""
        if hasattr(self, "_custom_class_names") and self._custom_class_names:
            return dict(self._custom_class_names)
        if hasattr(self, "model") and hasattr(self.model, "names") and self.model.names:
            parsed = _parse_names_metadata(self.model.names)
            if parsed:
                return parsed

        names = self._backend_strategy.get_class_names()
        if names:
            return names
        return dict(DEFAULT_CLASS_NAMES)

    def set_class_names(self, names: Optional[Dict[int, str]]) -> None:
        """Explicitly override class names dictionary."""
        if names:
            self._custom_class_names = {int(k): str(v) for k, v in names.items()}
        else:
            self._custom_class_names = dict(DEFAULT_CLASS_NAMES)

    # --------------------------------------------------------------------------
    # Inference & Output Decoding
    # --------------------------------------------------------------------------
    def predict_tile(
        self,
        tile_tensor_or_numpy: Any,
        tile_size: Optional[int] = None,
        conf_threshold: float = 0.20,
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

        target_size = self._snap_resolution(640 if tile_size == 736 else (tile_size or in_h))

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
        # For 736px slices, detections stay in 640px model space for host-device offset mapping
        decode_orig_size = (
            (target_size, target_size)
            if (tile_size == 736 or (in_h == 736 and in_w == 736))
            else (in_w, in_h)
        )
        detections = self._decode_yolo_output(
            raw_output=raw_output,
            model_size=target_size,
            original_size=decode_orig_size,
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
        """Run forward pass through active backend strategy (TensorRT or ONNX Runtime)."""
        return self._backend_strategy.run_inference(session, input_array, size)

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
        # Ensure raw_output is 3D (1, C, N) or (1, N, C)
        if raw_output.ndim == 2:
            raw_output = np.expand_dims(raw_output, axis=0)

        if raw_output.ndim != 3:
            logger.warning("Unexpected raw_output ndim: %s", getattr(raw_output, "ndim", None))
            return []

        # Ensure shape is (1, C, N) where C is channels (4 + num_classes)
        # Standard YOLO output is (1, C, N) with C in [5, 200] and N anchors (e.g. 2100, 8400).
        # If model outputs transposed (1, N, C), transpose to (1, C, N).
        if raw_output.shape[1] >= 200 and raw_output.shape[2] < 200:
            raw_output = np.transpose(raw_output, (0, 2, 1))
        elif raw_output.shape[2] in (16, 20, 80, 84) and raw_output.shape[1] not in (16, 20, 80, 84):
            raw_output = np.transpose(raw_output, (0, 2, 1))

        if raw_output.shape[1] < 5:
            logger.warning("Output tensor has fewer than 5 channels (4 coords + classes): %s", raw_output.shape)
            return []

        coords = raw_output[0, :4, :]  # (4, N) -> xc, yc, w, h
        class_scores = raw_output[0, 4:, :]  # (num_classes, N)

        # Apply Sigmoid if logits are unnormalized (e.g. negative values or > 1.0)
        if np.any(class_scores > 1.0) or np.any(class_scores < 0.0):
            class_scores = 1.0 / (1.0 + np.exp(-np.clip(class_scores, -25.0, 25.0)))

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
