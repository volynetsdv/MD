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
for _root in _site_roots:
    if not _root.exists():
        continue
    _nvidia_root = _root / "nvidia"
    if _nvidia_root.exists():
        for _pkg in sorted(_nvidia_root.glob("*")):
            for _sub in (_pkg / "lib", _pkg / "bin"):
                if _sub.exists():
                    for _f in sorted(_sub.glob(_lib_ext)):
                        if _f.is_file():
                            _rf = str(_f.resolve())
                            if _rf not in _seen_bins:
                                _seen_bins.add(_rf)
                                binaries.append((str(_f), "."))
    for _trt_dir_name in ("tensorrt_libs", "tensorrt"):
        _trt_dir = _root / _trt_dir_name
        if _trt_dir.exists():
            for _f in sorted(_trt_dir.glob(_lib_ext)):
                if _f.is_file():
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

