"""Unified Inference Dispatcher with Cross-Platform Windows Fallback.

Provides a unified detection interface across C++ TensorRT and ONNX Runtime backends.
Ensures seamless operation on Windows 10/11 without requiring TensorRT SDK compilation
by automatically falling back to DirectML (DmlExecutionProvider), CUDA, or CPU.
"""

from __future__ import annotations

import abc
import ctypes
import enum
import glob
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _register_runtime_library_paths() -> None:
    """Register bundled CUDA, cuDNN, and TensorRT shared library directories.

    Supports both PyInstaller frozen bundles (`sys._MEIPASS`) and virtual-env
    `nvidia-*` wheel layouts on Linux (`RTLD_GLOBAL`) and Windows (`add_dll_directory`).
    """
    try:
        candidate_dirs: List[Path] = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            mp_path = Path(meipass)
            candidate_dirs.append(mp_path)
            for sub in ("nvidia", "tensorrt", "tensorrt_libs"):
                sub_path = mp_path / sub
                if sub_path.exists():
                    candidate_dirs.append(sub_path)
                    for child in sub_path.glob("*"):
                        for lib_sub in (child, child / "lib", child / "bin"):
                            if lib_sub.exists():
                                candidate_dirs.append(lib_sub)

        for entry in list(sys.path):
            if not entry:
                continue
            ep = Path(entry)
            nvidia_dir = ep / "nvidia"
            if nvidia_dir.exists():
                for pkg_dir in nvidia_dir.glob("*"):
                    for lib_sub in (pkg_dir / "lib", pkg_dir / "bin"):
                        if lib_sub.exists():
                            candidate_dirs.append(lib_sub)
            for trt_pkg in ("tensorrt_libs", "tensorrt"):
                trt_dir = ep / trt_pkg
                if trt_dir.exists():
                    candidate_dirs.append(trt_dir)

        seen: set[str] = set()
        unique_dirs: List[Path] = []
        for d in candidate_dirs:
            resolved = str(d.resolve()) if d.exists() else ""
            if resolved and resolved not in seen:
                seen.add(resolved)
                unique_dirs.append(Path(resolved))

        if sys.platform == "win32":
            add_dll_dir = getattr(os, "add_dll_directory", None)
            for d in unique_dirs:
                if add_dll_dir is not None:
                    try:
                        add_dll_dir(str(d))
                    except Exception:
                        pass
                os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"
        else:
            preload_patterns = (
                "libcudart.so.12*",
                "libnvJitLink.so.12*",
                "libcublasLt.so.12*",
                "libcublas.so.12*",
                "libcufft.so.11*",
                "libcurand.so.10*",
                "libcudnn.so.9*",
                "libcudnn_*.so.9*",
                "libnvinfer.so.*",
                "libnvinfer_plugin.so.*",
                "libnvonnxparser.so.*",
            )
            for d in unique_dirs:
                for pat in preload_patterns:
                    for lib_file in sorted(glob.glob(str(d / pat))):
                        if os.path.isfile(lib_file):
                            try:
                                ctypes.CDLL(lib_file, mode=ctypes.RTLD_GLOBAL)
                            except Exception:
                                pass
    except Exception:
        pass


_register_runtime_library_paths()

# Safe import of pytiling_core if compiled
_PyTilingDetection = None
pytiling_core = None
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

_VRAM_UNAVAILABLE_LOGGED: bool = False
_NVIDIA_SMI_FAILED: bool = False
_PYNVML_FAILED: bool = False
_NVIDIA_SMI_CACHE: Optional[Tuple[float, Tuple[float, float]]] = None
_NVIDIA_SMI_CACHE_TTL_S: float = 2.0


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


def _has_cuda_gpu() -> bool:
    """Return True if an active CUDA-capable GPU and functional CUDA runtime are accessible."""
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_vis is not None and cuda_vis.strip().lower() in ("", "-1", "none", "nodevfiles"):
        return False

    if pytiling_core is not None and hasattr(pytiling_core, "get_gpu_memory_info"):
        try:
            _, total_bytes = pytiling_core.get_gpu_memory_info()
            if total_bytes > 0:
                return True
        except Exception:
            pass

    try:
        import ctypes

        lib_names = (
            ("cudart64_12.dll", "cudart64_13.dll", "cudart64_110.dll")
            if sys.platform == "win32"
            else ("libcudart.so", "libcudart.so.12", "libcudart.so.13")
        )
        for lib_name in lib_names:
            try:
                cudart = ctypes.CDLL(lib_name)
                count = ctypes.c_int(0)
                if cudart.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0:
                    return True
            except OSError:
                continue
    except Exception:
        pass

    try:
        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return True
    except Exception:
        pass

    return False


def calculate_safe_batch_size(
    target_tile_size: int = 640,
    is_turbo: bool = False,
    max_cap: Optional[int] = None,
    safety_factor: float = 0.80,
    execution_provider: Optional[str] = None,
    supports_dynamic_batch: bool = True,
) -> int:
    """Analytical calculation of the maximum safe tile batch size for Turbo mode (OOM-Safety).

    Mathematical model:
        B_calc = floor((safety_factor * VRAM_free_MB) / Delta_tile(S_m))
        B_target = clamp(B_calc, 1, effective_cap)
    where:
        Delta_tile(S_m) = 110.0 * (S_m / 640.0)^2 [MB]
        If not is_turbo or not supports_dynamic_batch -> B_target = 1 (strict)
        If execution_provider == "CPUExecutionProvider" -> B_target = 1 (sequential CPU)
        If VRAM_free_MB < 200 MB -> B_target = 1 (strict OOM guard)
        If max_cap is None:
            If VRAM_free_MB >= 6000 MB: effective_cap = 12 (if S_m <= 416) else 8
            Else: effective_cap = 4
    """
    if not is_turbo or not supports_dynamic_batch:
        return 1

    if execution_provider == "CPUExecutionProvider":
        return 1

    try:
        free_mb, total_mb = UnifiedDetector.query_vram_mb()

        if free_mb <= 0.0 or total_mb <= 0.0:
            if execution_provider == "DmlExecutionProvider":
                return min(2, max_cap) if (max_cap is not None and max_cap > 0) else 2
            return 1

        if free_mb < 200.0:
            return 1

        if max_cap is None or max_cap <= 0:
            if free_mb >= 6000.0:
                effective_cap = 12 if target_tile_size <= 416 else 8
            else:
                effective_cap = 4
        else:
            effective_cap = max_cap

        norm_size = max(1.0, float(target_tile_size))
        cost_mb = 110.0 * ((norm_size / 640.0) ** 2)
        if cost_mb <= 0:
            cost_mb = 110.0

        b = int((free_mb * safety_factor) // cost_mb)
        return max(1, min(b, effective_cap))
    except Exception as exc:
        logger.warning("calculate_safe_batch_size: fallback to 1 due to exception: %s", exc)
        return 1


MIN_ENGINE_SIZE_BYTES: int = 1048576  # 1 MB


def check_tensorrt_available(models_dir: Optional[Path] = None) -> bool:
    """Check if TensorRT SDK/runtime, NVIDIA GPU, and engine/ONNX plans are present.

    Looks for:
    1. C++ library artifacts (`libtrt_detector.so` or `trt_detector.pyd`) or Python `tensorrt`.
    2. CUDA-capable GPU (queried natively without PyTorch).
    3. Valid compiled `.engine` plan files (>= 1MB) or `.onnx` files (>= 1MB) for local engine build.
    """
    repo_root = Path(__file__).resolve().parent.parent
    lib_candidates = [
        repo_root / "build" / "libtrt_detector.so",
        repo_root / "build" / "trt_detector.pyd",
        repo_root / "build" / "Release" / "trt_detector.pyd",
    ]
    has_lib = any(cand.exists() for cand in lib_candidates)

    has_py_trt = False
    try:
        import tensorrt as trt

        has_py_trt = trt is not None
    except ImportError:
        pass

    if not (has_lib or has_py_trt):
        return False

    if not _has_cuda_gpu():
        return False

    m_dir = Path(models_dir) if models_dir is not None else repo_root / "models"
    if m_dir.exists():
        engine_files = list(m_dir.glob("*.engine"))
        has_valid_engine = any(f.stat().st_size >= MIN_ENGINE_SIZE_BYTES for f in engine_files)
        onnx_files = list(m_dir.glob("*.onnx"))
        has_valid_onnx = any(f.stat().st_size >= MIN_ENGINE_SIZE_BYTES for f in onnx_files)
        if engine_files and not has_valid_engine and not has_valid_onnx:
            logger.info("UnifiedDetector: All .engine files are stubs (<1MB) and no ONNX source found. Bypassing TensorRT.")
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
    def supports_dynamic_batch(self) -> bool:
        """Indicate whether the backend supports dynamic batch size > 1."""
        return True

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

    @staticmethod
    def _get_engine_cache_dir() -> Path:
        """Return writable local cache directory for compiled TensorRT engines."""
        if sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        else:
            base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        cache_dir = base / "AerialWorkstation" / "engines"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _find_onnx_path(self, size: int) -> Optional[Path]:
        """Locate source ONNX file matching requested tile resolution."""
        for name in (f"yolo_{size}.onnx", f"yolo26s_{size}.onnx"):
            cand = self.models_dir / name
            if cand.exists() and cand.stat().st_size >= MIN_ENGINE_SIZE_BYTES:
                return cand
        return None

    @staticmethod
    def _build_engine_from_onnx(onnx_path: Path, engine_out_path: Path, size: int) -> bytes:
        """Compile an ONNX model into a GPU-native TensorRT engine plan on first run."""
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        explicit_batch_flag = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
        network_flags = (1 << int(explicit_batch_flag)) if explicit_batch_flag is not None else 0
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, trt_logger)

        if not parser.parse_from_file(str(onnx_path)):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(
                f"Failed to parse ONNX for TensorRT ({onnx_path}): {'; '.join(errors)}"
            )

        config = builder.create_builder_config()
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        input_tensor = network.get_input(0)
        in_name = input_tensor.name
        profile = builder.create_optimization_profile()
        profile.set_shape(
            in_name,
            (1, 3, size, size),
            (4, 3, size, size),
            (12, 3, size, size),
        )
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT failed to build serialized engine from {onnx_path}")

        raw_bytes = bytes(serialized)
        try:
            engine_out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = engine_out_path.with_suffix(".engine.tmp")
            tmp_path.write_bytes(raw_bytes)
            tmp_path.replace(engine_out_path)
            logger.info(
                "TensorRTBackend: Cached compiled engine (%d MB) to %s",
                len(raw_bytes) // (1024 * 1024),
                engine_out_path,
            )
        except Exception as save_err:
            logger.warning("TensorRTBackend: Could not save engine cache %s: %s", engine_out_path, save_err)
        return raw_bytes

    def get_session(self, size: int) -> Any:
        """Load and cache a TensorRT engine execution context, building locally if needed."""
        if size in self._sessions:
            return self._sessions[size]

        import tensorrt as trt

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        cache_dir = self._get_engine_cache_dir()
        bundled_engine = self.models_dir / f"yolo_{size}.engine"
        cached_engine = cache_dir / f"yolo_{size}.engine"

        engine = None
        for candidate in (cached_engine, bundled_engine):
            if candidate.exists() and candidate.stat().st_size >= MIN_ENGINE_SIZE_BYTES:
                engine = runtime.deserialize_cuda_engine(candidate.read_bytes())
                if engine is not None:
                    logger.info("TensorRTBackend: Loaded engine for size %d from %s", size, candidate)
                    break
                logger.warning(
                    "TensorRTBackend: Engine %s incompatible with current GPU/TensorRT; rebuilding locally.",
                    candidate,
                )

        if engine is None:
            onnx_path = self._find_onnx_path(size)
            if onnx_path is None:
                raise FileNotFoundError(
                    f"TensorRT engine yolo_{size}.engine unavailable and no ONNX model found in {self.models_dir}."
                )
            logger.info("TensorRTBackend: Building local TensorRT engine for resolution %d from %s...", size, onnx_path)
            raw_bytes = self._build_engine_from_onnx(onnx_path, cached_engine, size)
            engine = runtime.deserialize_cuda_engine(raw_bytes)
            if engine is None:
                raise RuntimeError(f"Failed to deserialize newly built TensorRT engine for size {size}")

        context = engine.create_execution_context()
        session = {"engine": engine, "context": context, "size": size}
        self._sessions[size] = session
        logger.info("TensorRTBackend: Cached session for resolution %d.", size)
        return session

    def run_inference(self, session: Any, input_array: Any, size: int) -> np.ndarray:
        """Execute TensorRT forward pass directly in VRAM without importing PyTorch."""
        if pytiling_core is None or not hasattr(pytiling_core, "gpu_malloc"):
            raise RuntimeError("pytiling_core CUDA buffer helpers are unavailable for TensorRT inference.")

        engine = session["engine"]
        context = session["context"]

        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)

        is_gpu_tensor = (
            hasattr(input_array, "data_ptr")
            and hasattr(input_array, "shape")
            and (
                getattr(input_array, "is_cuda", False)
                or getattr(getattr(input_array, "device", None), "type", None) == "cuda"
            )
        )

        d_input_ptr = 0
        owns_d_input = False
        host_arr: Optional[np.ndarray] = None

        if is_gpu_tensor:
            tensor = input_array
            if hasattr(tensor, "is_contiguous") and not tensor.is_contiguous():
                tensor = tensor.contiguous()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            input_shape = tuple(int(dim) for dim in tensor.shape)
            b = input_shape[0]
            d_input_ptr = int(tensor.data_ptr())
        else:
            host_arr = np.asarray(input_array, dtype=np.float32)
            if host_arr.ndim == 3:
                host_arr = np.expand_dims(host_arr, axis=0)
            if not host_arr.flags.c_contiguous:
                host_arr = np.ascontiguousarray(host_arr)
            input_shape = tuple(int(dim) for dim in host_arr.shape)
            b = input_shape[0]
            d_input_ptr = int(pytiling_core.gpu_malloc(host_arr.nbytes))
            if d_input_ptr == 0:
                raise RuntimeError(f"CUDA gpu_malloc failed for input tensor ({host_arr.nbytes} bytes).")
            owns_d_input = True
            err = pytiling_core.gpu_memcpy_h2d(d_input_ptr, host_arr.ctypes.data, host_arr.nbytes)
            if err != 0:
                pytiling_core.gpu_free(d_input_ptr)
                raise RuntimeError(f"CUDA gpu_memcpy_h2d failed with error code {err}.")

        d_output_ptr = 0
        try:
            try:
                context.set_input_shape(input_name, input_shape)
                out_shape = list(context.get_tensor_shape(output_name))
            except Exception:
                out_shape = list(engine.get_tensor_shape(output_name))
                out_shape[0] = b

            if any(dim <= 0 for dim in out_shape):
                engine_out = list(engine.get_tensor_shape(output_name))
                out_shape = [b if i == 0 else d for i, d in enumerate(engine_out)]

            out_host = np.empty(tuple(out_shape), dtype=np.float32)
            d_output_ptr = int(pytiling_core.gpu_malloc(out_host.nbytes))
            if d_output_ptr == 0:
                raise RuntimeError(f"CUDA gpu_malloc failed for output tensor ({out_host.nbytes} bytes).")

            if b > 1:
                try:
                    context.set_input_shape(input_name, input_shape)
                    context.set_tensor_address(input_name, d_input_ptr)
                    context.set_tensor_address(output_name, d_output_ptr)
                    if not context.execute_async_v3(0):
                        raise RuntimeError("TensorRT execute_async_v3 returned False.")
                    pytiling_core.gpu_synchronize()
                    pytiling_core.gpu_memcpy_d2h(out_host.ctypes.data, d_output_ptr, out_host.nbytes)
                    return out_host
                except Exception as exc:
                    logger.warning(
                        "TensorRT native dynamic batch execution failed for B=%d (size %d): %s. "
                        "Executing per-slice in VRAM.",
                        b,
                        size,
                        exc,
                    )
                    single_in_shape = (1, input_shape[1], input_shape[2], input_shape[3])
                    single_out_shape = (1, *out_shape[1:])
                    single_in_bytes = int(np.prod(single_in_shape)) * 4
                    single_out_bytes = int(np.prod(single_out_shape)) * 4
                    for i in range(b):
                        context.set_input_shape(input_name, single_in_shape)
                        context.set_tensor_address(input_name, d_input_ptr + i * single_in_bytes)
                        context.set_tensor_address(output_name, d_output_ptr + i * single_out_bytes)
                        if not context.execute_async_v3(0):
                            raise RuntimeError(f"TensorRT per-slice execute_async_v3 failed at slice {i}.")
                    pytiling_core.gpu_synchronize()
                    pytiling_core.gpu_memcpy_d2h(out_host.ctypes.data, d_output_ptr, out_host.nbytes)
                    return out_host
            else:
                context.set_tensor_address(input_name, d_input_ptr)
                context.set_tensor_address(output_name, d_output_ptr)
                if not context.execute_async_v3(0):
                    raise RuntimeError("TensorRT execute_async_v3 returned False.")
                pytiling_core.gpu_synchronize()
                pytiling_core.gpu_memcpy_d2h(out_host.ctypes.data, d_output_ptr, out_host.nbytes)
                return out_host
        finally:
            if d_output_ptr:
                pytiling_core.gpu_free(d_output_ptr)
            if owns_d_input and d_input_ptr:
                pytiling_core.gpu_free(d_input_ptr)


class ONNXRuntimeBackendBase(InferenceBackend):
    """Base inference strategy for ONNX Runtime execution providers."""

    def __init__(
        self,
        models_dir: Path,
        providers: Optional[List[str]] = None,
    ) -> None:
        super().__init__(models_dir)
        self._providers: List[str] = list(providers) if providers else ["CPUExecutionProvider"]
        self._active_provider: str = self._providers[0] if self._providers else "CPUExecutionProvider"
        self._supports_dynamic_batch: bool = True

    @property
    def name(self) -> BackendType:
        return BackendType.ONNXRUNTIME

    @property
    def supports_dynamic_batch(self) -> bool:
        """Indicate whether the backend supports dynamic batch size > 1."""
        return self._supports_dynamic_batch

    @property
    def providers(self) -> List[str]:
        return self._providers

    @property
    def execution_provider(self) -> str:
        return self._active_provider

    @property
    def is_gpu_accelerated(self) -> bool:
        return self._active_provider in (
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
        )

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

        requested_providers = list(self._providers)
        try:
            session = ort.InferenceSession(
                str(model_path), sess_options, providers=requested_providers
            )
        except Exception as err:
            logger.warning(
                "UnifiedDetector: Provider %s failed (%s). Falling back to CPUExecutionProvider.",
                requested_providers,
                err,
            )
            self._providers = ["CPUExecutionProvider"]
            self._active_provider = "CPUExecutionProvider"
            session = ort.InferenceSession(
                str(model_path), sess_options, providers=self._providers
            )

        actual_providers = (
            list(session.get_providers())
            if hasattr(session, "get_providers")
            else list(self._providers)
        )
        requested_gpu = [
            p
            for p in requested_providers
            if p in ("CUDAExecutionProvider", "DmlExecutionProvider", "TensorrtExecutionProvider")
        ]
        if requested_gpu and not any(p in actual_providers for p in requested_gpu):
            logger.warning(
                "UnifiedDetector: Requested GPU execution provider(s) %s, but ONNX Runtime "
                "silently fell back to %s (verify CUDA 12 / cuDNN 9 / DirectML shared libraries). "
                "Updating active execution provider to %s.",
                requested_gpu,
                actual_providers,
                actual_providers[0] if actual_providers else "CPUExecutionProvider",
            )
            self._providers = actual_providers if actual_providers else ["CPUExecutionProvider"]
            self._active_provider = self._providers[0]
        elif actual_providers:
            self._active_provider = actual_providers[0]

        inputs = session.get_inputs() if hasattr(session, "get_inputs") else []
        if inputs:
            inp_shape = inputs[0].shape
            if inp_shape and len(inp_shape) > 0:
                first_dim = inp_shape[0]
                if isinstance(first_dim, int) and first_dim == 1:
                    self._supports_dynamic_batch = False
                    logger.info(
                        "ONNXRuntime (%s): Input '%s' has static batch size 1 (shape %s). "
                        "Configuring supports_dynamic_batch = False.",
                        self.execution_provider,
                        inputs[0].name,
                        inp_shape,
                    )

        self._sessions[size] = session
        logger.info(
            "ONNXRuntime (%s): Cached session for tile size %d (dynamic_batch=%s).",
            self.execution_provider,
            size,
            self._supports_dynamic_batch,
        )
        return session

    def run_inference(self, session: Any, input_array: np.ndarray, size: int) -> np.ndarray:
        """Run forward pass via ONNX Runtime with native dynamic batching support."""
        if hasattr(session, "get_inputs") and session.get_inputs():
            input_name = session.get_inputs()[0].name
        else:
            input_name = "images"
        b = input_array.shape[0] if input_array.ndim == 4 else 1

        if not input_array.flags.c_contiguous:
            input_array = np.ascontiguousarray(input_array)

        if b > 1:
            if not self._supports_dynamic_batch:
                out_slices = []
                for i in range(b):
                    slice_i = np.ascontiguousarray(input_array[i : i + 1])
                    single_out = session.run(None, {input_name: slice_i})
                    out_slices.append(single_out[0])
                return np.concatenate(out_slices, axis=0)

            try:
                outputs = session.run(None, {input_name: input_array})
                return outputs[0]
            except Exception as exc:
                self._supports_dynamic_batch = False
                logger.warning(
                    "ONNXRuntime dynamic batch execution failed for B=%d (size %d): %s. "
                    "Marking session as fixed batch=1 and falling back to sequential slice evaluation.",
                    b,
                    size,
                    exc,
                )
                out_slices = []
                for i in range(b):
                    slice_i = np.ascontiguousarray(input_array[i : i + 1])
                    single_out = session.run(None, {input_name: slice_i})
                    out_slices.append(single_out[0])
                return np.concatenate(out_slices, axis=0)
        else:
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


class CUDAExecutionProviderBackend(ONNXRuntimeBackendBase):
    """NVIDIA CUDA (CUDAExecutionProvider) backend strategy for ONNX Runtime."""

    def __init__(
        self,
        models_dir: Path,
        custom_providers: Optional[List[str]] = None,
    ) -> None:
        provs = custom_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        super().__init__(models_dir, providers=provs)


class CPUExecutionProviderBackend(ONNXRuntimeBackendBase):
    """CPU execution provider strategy for deterministic portable inference."""

    def __init__(
        self,
        models_dir: Path,
        custom_providers: Optional[List[str]] = None,
    ) -> None:
        provs = custom_providers or ["CPUExecutionProvider"]
        super().__init__(models_dir, providers=provs)


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

    @staticmethod
    def query_vram_mb() -> Tuple[float, float]:
        """Query free and total GPU VRAM in MiB without importing PyTorch.

        Cascade:
            1. Native C++ `pytiling_core.get_gpu_memory_info()` (via `cudaMemGetInfo`).
            2. `nvidia-smi` CLI query (1.5s timeout, cached on success/failure).
            3. `pynvml` NVML Python bindings, if available.
            4. Silent `(0.0, 0.0)` with one-time informational log if unavailable.

        Returns:
            Tuple `(free_vram_mb, total_vram_mb)` in MiB.
        """
        global _VRAM_UNAVAILABLE_LOGGED, _NVIDIA_SMI_FAILED, _PYNVML_FAILED, _NVIDIA_SMI_CACHE

        # 1. Native CUDA call via pytiling_core (Zero PyTorch dependency)
        if pytiling_core is not None and hasattr(pytiling_core, "get_gpu_memory_info"):
            try:
                free_b, total_b = pytiling_core.get_gpu_memory_info()
                if total_b > 0:
                    return float(free_b) / (1024.0 * 1024.0), float(total_b) / (1024.0 * 1024.0)
            except Exception:
                pass

        # 2. nvidia-smi subprocess fallback (cached to avoid frame-loop overhead/spam)
        if not _NVIDIA_SMI_FAILED:
            now = time.monotonic()
            if _NVIDIA_SMI_CACHE is not None and (now - _NVIDIA_SMI_CACHE[0]) < _NVIDIA_SMI_CACHE_TTL_S:
                return _NVIDIA_SMI_CACHE[1]
            try:
                run_kwargs: Dict[str, Any] = {
                    "capture_output": True,
                    "text": True,
                    "timeout": 1.5,
                    "check": False,
                }
                if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                    run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    **run_kwargs,
                )
                if proc.returncode == 0 and proc.stdout:
                    first_line = proc.stdout.strip().splitlines()[0]
                    parts = [p.strip() for p in first_line.split(",")]
                    if len(parts) >= 2:
                        free_mb = float(parts[0])
                        total_mb = float(parts[1])
                        if total_mb > 0.0:
                            res = (free_mb, total_mb)
                            _NVIDIA_SMI_CACHE = (now, res)
                            return res
                _NVIDIA_SMI_FAILED = True
            except Exception:
                _NVIDIA_SMI_FAILED = True

        # 3. pynvml fallback
        if not _PYNVML_FAILED:
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return float(info.free) / (1024.0 * 1024.0), float(info.total) / (1024.0 * 1024.0)
            except Exception:
                _PYNVML_FAILED = True

        # 4. Unavailable: log status at most once, no per-frame spam
        if not _VRAM_UNAVAILABLE_LOGGED:
            logger.info(
                "UnifiedDetector: CUDA VRAM query unavailable (pytiling_core/nvidia-smi/pynvml); "
                "using conservative batch defaults."
            )
            _VRAM_UNAVAILABLE_LOGGED = True
        return 0.0, 0.0

    @property
    def active_backend_name(self) -> str:
        """Return actual active backend/provider name (e.g. 'TensorRT', 'CUDAExecutionProvider', 'CPUExecutionProvider')."""
        return self._backend_strategy.execution_provider

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
    def supports_dynamic_batch(self) -> bool:
        """Indicate whether the active backend strategy and models support dynamic batch size > 1."""
        return getattr(self._backend_strategy, "supports_dynamic_batch", True)

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

    def predict_batch(
        self,
        tile_slices: Union[List[Any], np.ndarray],
        tile_size: Optional[int] = None,
        conf_threshold: float = 0.20,
        iou_threshold: float = 0.45,
    ) -> List[List[Detection]]:
        """Run object detection on a batch of tiles.

        Args:
            tile_slices: List of B tile images (NumPy arrays or PyTorch Tensors)
                         or a 4D array/tensor of shape (B, 3, H, W).
            tile_size: Target resolution model size. If None, inferred from first tile.
            conf_threshold: Minimum confidence score to accept detection.
            iou_threshold: IoU threshold for Non-Maximum Suppression.

        Returns:
            List of length B, where each element is a List[Detection] in tile-local coordinates.
        """
        if tile_slices is None:
            return []
        if isinstance(tile_slices, (list, tuple)) and len(tile_slices) == 0:
            return []
        if hasattr(tile_slices, "numel") and tile_slices.numel() == 0:
            return []

        # If passed a single image, wrap appropriately
        if isinstance(tile_slices, np.ndarray) and tile_slices.ndim == 3:
            return [self.predict_tile(tile_slices, tile_size, conf_threshold, iou_threshold)]

        # Fast path for 4D input (torch.Tensor or ndarray of shape B, 3, H, W)
        if hasattr(tile_slices, "ndim") and tile_slices.ndim == 4:
            b_size = tile_slices.shape[0]
            if b_size == 0:
                return []
            target_size = self._snap_resolution(tile_size or tile_slices.shape[-1])
            orig_size = (target_size, target_size)
            session = self.get_session(target_size)
            raw_output = self._backend_strategy.run_inference(session, tile_slices, target_size)

            if raw_output.ndim == 2:
                raw_output = np.expand_dims(raw_output, axis=0)
            if raw_output.shape[1] >= 200 and raw_output.shape[2] < 200:
                raw_output = np.transpose(raw_output, (0, 2, 1))
            elif raw_output.shape[2] in (16, 20, 80, 84) and raw_output.shape[1] not in (16, 20, 80, 84):
                raw_output = np.transpose(raw_output, (0, 2, 1))

            batch_detections: List[List[Detection]] = []
            for i in range(b_size):
                single_raw = raw_output[i : i + 1]
                dets = self._decode_yolo_output(
                    raw_output=single_raw,
                    model_size=target_size,
                    original_size=orig_size,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )
                batch_detections.append(dets)
            return batch_detections

        # Preprocess items into contiguous 4D batch array (B, 3, target_size, target_size) directly
        b_size = len(tile_slices)
        first_item = tile_slices[0]
        if hasattr(first_item, "shape"):
            in_h0 = (
                first_item.shape[0]
                if (first_item.ndim == 3 and first_item.shape[2] in (1, 3, 4))
                else first_item.shape[-2]
            )
        else:
            in_h0 = 640
        target_size = self._snap_resolution(640 if tile_size == 736 else (tile_size or in_h0))

        batch_inputs = np.empty((b_size, 3, target_size, target_size), dtype=np.float32)
        orig_sizes = []
        for i, item in enumerate(tile_slices):
            if hasattr(item, "detach") and hasattr(item, "cpu"):
                item = item.detach().cpu().numpy()
            if not isinstance(item, np.ndarray):
                item = np.asarray(item)

            if item.ndim == 3 and item.shape[2] in (1, 3, 4):
                h_i, w_i = item.shape[0], item.shape[1]
                orig_sizes.append((w_i, h_i))
                if (h_i != target_size) or (w_i != target_size):
                    if item.shape[2] == 4:
                        item = item[:, :, :3]
                    elif item.shape[2] == 1:
                        item = np.repeat(item, 3, axis=2)
                    item = cv2.resize(
                        item, (target_size, target_size), interpolation=cv2.INTER_LINEAR
                    )
                    if item.ndim == 2:
                        item = np.repeat(item[:, :, None], 3, axis=2)

                if item.shape[2] == 4:
                    item = item[:, :, :3]
                elif item.shape[2] == 1:
                    item = np.repeat(item, 3, axis=2)

                if item.dtype == np.uint8:
                    np.multiply(
                        np.transpose(item, (2, 0, 1)),
                        1.0 / 255.0,
                        out=batch_inputs[i],
                        dtype=np.float32,
                    )
                else:
                    item_f = item.astype(np.float32) if item.dtype != np.float32 else item
                    if item_f.max() > 1.5:
                        item_f = item_f / 255.0
                    batch_inputs[i] = np.transpose(item_f, (2, 0, 1))
            else:
                arr, h_i, w_i = self._preprocess_input(item)
                orig_sizes.append((w_i, h_i))
                if (h_i != target_size) or (w_i != target_size):
                    hwc = np.transpose(arr[0], (1, 2, 0))
                    resized = cv2.resize(
                        hwc, (target_size, target_size), interpolation=cv2.INTER_LINEAR
                    )
                    batch_inputs[i] = np.transpose(resized, (2, 0, 1))
                else:
                    batch_inputs[i] = arr[0]

        # Execute inference
        session = self.get_session(target_size)
        raw_output = self._run_inference(session, batch_inputs, target_size)

        # Ensure raw_output is 3D with shape (B, C, N)
        if raw_output.ndim == 2:
            raw_output = np.expand_dims(raw_output, axis=0)

        if raw_output.shape[1] >= 200 and raw_output.shape[2] < 200:
            raw_output = np.transpose(raw_output, (0, 2, 1))
        elif raw_output.shape[2] in (16, 20, 80, 84) and raw_output.shape[1] not in (16, 20, 80, 84):
            raw_output = np.transpose(raw_output, (0, 2, 1))

        # Decode each item in batch
        batch_detections: List[List[Detection]] = []
        for i in range(b_size):
            single_raw = raw_output[i : i + 1]
            in_w_i, in_h_i = orig_sizes[i]
            decode_orig_size = (
                (target_size, target_size)
                if (tile_size == 736 or (in_h_i == 736 and in_w_i == 736))
                else (in_w_i, in_h_i)
            )
            dets = self._decode_yolo_output(
                raw_output=single_raw,
                model_size=target_size,
                original_size=decode_orig_size,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            batch_detections.append(dets)

        return batch_detections

    infer_batch = predict_batch

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

        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)

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
