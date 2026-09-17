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
datas = [
    ("models/*.onnx", "models"),
]

# Include compiled C++ binaries if present
if (repo_root / "build").exists():
    datas.append(("build/*.so", "."))
    datas.append(("build/*.pyd", "."))
    datas.append(("build/bindings/*.so", "bindings"))
    datas.append(("build/bindings/*.pyd", "bindings"))

# Hidden imports required by dynamic dispatch
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtOpenGLWidgets",
    "onnxruntime",
    "cv2",
    "numpy",
    "gui.canvas_viewer",
    "gui.main_window",
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

