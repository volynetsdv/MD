# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for Aerial Reconnaissance Workstation.

Builds a standalone executable bundle for Windows (.exe) and Linux (.AppImage/binary).
Excludes heavy training dependencies to keep distribution minimal.
"""

import sys
from pathlib import Path

block_cipher = None
repo_root = Path.cwd()

# Data files to bundle
datas = []
models_dir = repo_root / "models"
if models_dir.exists():
    for m in sorted(models_dir.glob("*.onnx")):
        datas.append((str(m), "models"))

# Include compiled C++ binaries if present (only existing files)
for b_dir in [repo_root / "build", repo_root / "build" / "bindings", repo_root / "build" / "Release"]:
    if b_dir.exists():
        for ext in ("*.so", "*.pyd", "*.dll"):
            for f in sorted(b_dir.glob(ext)):
                datas.append((str(f), "."))

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

# Exclude heavy training frameworks and dev tools from inference distribution
excludes = [
    "ultralytics",
    "pytest",
    "pytestqt",
    "IPython",
    "jupyter",
    "matplotlib",
    "tensorboard",
    "torch.testing",
]

a = Analysis(
    ["main.py"],
    pathex=[str(repo_root)],
    binaries=[],
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

