# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for Aerial Reconnaissance Workstation.

Builds a standalone executable bundle for Windows (.exe) and Linux (.AppImage/binary).
Excludes heavy training dependencies to keep distribution minimal.
"""

import importlib.util
import site
import sys
from pathlib import Path

block_cipher = None
repo_root = Path.cwd()

# Data and binary files to bundle
datas = []
binaries = []

models_dir = repo_root / "models"
if models_dir.exists():
    for m in sorted(models_dir.glob("*.onnx")):
        datas.append((str(m), "models"))

# Include compiled C++ binaries if present (only existing files)
for b_dir in [repo_root / "build", repo_root / "build" / "bindings", repo_root / "build" / "Release"]:
    if b_dir.exists():
        for ext in ("*.so", "*.so.*", "*.pyd", "*.dll"):
            for f in sorted(b_dir.glob(ext)):
                if f.is_file():
                    binaries.append((str(f), "."))

# Dynamically collect bundled NVIDIA CUDA 12 / cuDNN 9 / TensorRT runtime libraries
# so PyInstaller bundles remain autonomous on systems with CUDA 13 or no CUDA toolkit.
_site_roots = []
try:
    _site_roots.extend([Path(p) for p in site.getsitepackages()])
except Exception:
    pass
_site_roots.extend([Path(p) for p in sys.path if p])

_seen_bins = set()
_lib_ext = "*.dll" if sys.platform == "win32" else "*.so*"
_ALLOWED_NVIDIA_SUBSTRS = ("cublas", "cudnn", "cudart", "cuda_runtime", "nvjitlink", "cufft")
_EXCLUDED_CUDA_MODULES = (
    "cusparse",
    "cusolver",
    "curand",
    "nccl",
    "nvshmem",
    "cufile",
    "cupti",
    "checkpoint",
    "pcsampling",
)


def _is_unwanted_lib(name: str) -> bool:
    nl = name.lower()
    if sys.platform.startswith("linux"):
        # Exclude Windows cross-compilation binaries on Linux
        if "_win_" in nl or "_win." in nl or nl.endswith(".dll"):
            return True
        # Exclude datacenter-only TensorRT builder resources (sm90 Hopper, sm100 Blackwell DC)
        # while keeping universal JIT/PTX (_ptx) and consumer SMs (sm75, sm80, sm86, sm89, sm120).
        if "builder_resource_sm90" in nl or "builder_resource_sm100" in nl:
            return True
    elif sys.platform == "win32":
        # Exclude Linux cross-compilation binaries on Windows
        if "_linux_" in nl or "_linux." in nl or nl.endswith(".so") or ".so." in nl:
            return True
        if "builder_resource_sm90" in nl or "builder_resource_sm100" in nl:
            return True
    for mod in _EXCLUDED_CUDA_MODULES:
        if mod in nl:
            return True
    return False


for _root in _site_roots:
    if not _root.exists():
        continue
    _nvidia_root = _root / "nvidia"
    if _nvidia_root.exists():
        for _pkg in sorted(_nvidia_root.glob("*")):
            _pkg_name = _pkg.name.lower()
            is_generic_cu = _pkg_name.startswith("cu") and any(c.isdigit() for c in _pkg_name)
            if not (any(sub in _pkg_name for sub in _ALLOWED_NVIDIA_SUBSTRS) or is_generic_cu):
                continue
            for _sub in (_pkg / "lib", _pkg / "bin", _pkg):
                if _sub.exists() and _sub.is_dir():
                    for _f in sorted(_sub.glob(_lib_ext)):
                        if _f.is_file() and not _is_unwanted_lib(_f.name):
                            if is_generic_cu and not any(sub in _f.name.lower() for sub in _ALLOWED_NVIDIA_SUBSTRS):
                                continue
                            _rf = str(_f.resolve())
                            if _rf not in _seen_bins:
                                _seen_bins.add(_rf)
                                binaries.append((str(_f), "."))
    for _trt_dir_name in ("tensorrt_libs", "tensorrt"):
        _trt_dir = _root / _trt_dir_name
        if _trt_dir.exists():
            for _f in sorted(_trt_dir.glob(_lib_ext)):
                if _f.is_file() and not _is_unwanted_lib(_f.name):
                    _rf = str(_f.resolve())
                    if _rf not in _seen_bins:
                        _seen_bins.add(_rf)
                        binaries.append((str(_f), "."))

# Also search system library paths for dynamically loaded builder resources and CUDA libraries
if sys.platform.startswith("linux"):
    for _sys_dir in (
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/lib64",
        "/usr/lib",
    ):
        _sd = Path(_sys_dir)
        if _sd.exists():
            for _pat in ("*builder_resource*", "*libcufft*"):
                for _f in sorted(_sd.glob(_pat)):
                    if _f.is_file() and not _f.name.startswith("do_not_link") and not _is_unwanted_lib(_f.name):
                        _rf = str(_f.resolve())
                        if _rf not in _seen_bins:
                            _seen_bins.add(_rf)
                            binaries.append((str(_f), "."))

# Include config files
config_dir = repo_root / "config"
if config_dir.exists():
    for c in sorted(config_dir.glob("*.json")):
        datas.append((str(c), "config"))

# Hidden imports required by dynamic dispatch
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtOpenGLWidgets",
    "onnxruntime",
    "cv2",
    "numpy",
    "core.config_manager",
    "core.metadata_cache",
    "scripts.download_weights",
    "gui.canvas_viewer",
    "gui.main_window",
    "gui.settings_dialog",
    "gui.styles",
    "gui.async_worker",
    "src.detector_dispatcher",
]

for _opt_mod in ("tensorrt", "tensorrt_libs", "pynvml"):
    if importlib.util.find_spec(_opt_mod) is not None:
        hiddenimports.append(_opt_mod)

# Exclude heavy training frameworks and dev tools from inference distribution
excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "torch.testing",
    "ultralytics",
    "pytest",
    "pytestqt",
    "IPython",
    "jupyter",
    "matplotlib",
    "tensorboard",
]

a = Analysis(
    ["main.py"],
    pathex=[str(repo_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# =============================================================================
# Binary Filtering & Deduplication (Prevent ~8GB duplication & cross-OS ballast)
# =============================================================================
_filtered_binaries_by_base = {}

for dest, src, typecode in a.binaries:
    dest_path = Path(dest)
    base_name = dest_path.name
    if _is_unwanted_lib(base_name) or _is_unwanted_lib(str(dest)):
        continue

    lower_base = base_name.lower()
    # Flatten all CUDA and TensorRT binaries directly into root of _internal ('.')
    # so dynamic linker dlopen() resolves them via $ORIGIN/_internal RPATH without subfolder issues
    if (
        any(sub in lower_base for sub in ("cublas", "cudnn", "cudart", "cufft", "nvjitlink", "nvinfer", "nvonnxparser"))
        or "builder_resource" in lower_base
    ):
        dest = base_name

    # On Windows, dynamic library filenames are case-insensitive
    key = base_name.lower() if sys.platform == "win32" else base_name

    if key not in _filtered_binaries_by_base:
        _filtered_binaries_by_base[key] = (dest, src, typecode)
    else:
        # Prefer flat root destination (dest == base_name)
        existing_dest = _filtered_binaries_by_base[key][0]
        if dest == base_name and existing_dest != base_name:
            _filtered_binaries_by_base[key] = (dest, src, typecode)

a.binaries = list(_filtered_binaries_by_base.values())

_filtered_datas = []
for dest, src, typecode in a.datas:
    base_name = Path(dest).name
    if _is_unwanted_lib(base_name) or _is_unwanted_lib(str(dest)):
        continue
    # Do not duplicate shared libraries as datas
    if base_name in _filtered_binaries_by_base:
        continue
    _filtered_datas.append((dest, src, typecode))

a.datas = _filtered_datas

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AerialWorkstation",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AerialWorkstation",
)

