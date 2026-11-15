"""Application Entry Point for Aerial Reconnaissance Workstation."""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _configure_bundled_runtime_libs() -> None:
    """Register and preload CUDA 12 / cuDNN 9 / TensorRT shared libraries before ML/GUI imports.

    Handles PyInstaller frozen bundles (`sys._MEIPASS`) and virtual-environment wheel layouts
    so `onnxruntime`, `tensorrt`, and `pytiling_core` bind bundled GPU libraries rather than
    mismatched system-wide CUDA versions.
    """
    try:
        candidate_dirs: list[Path] = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            mp = Path(meipass)
            candidate_dirs.append(mp)
            for sub in ("nvidia", "tensorrt", "tensorrt_libs"):
                sp = mp / sub
                if sp.exists():
                    candidate_dirs.append(sp)
                    for child in sp.glob("*"):
                        for lib_sub in (child, child / "lib", child / "bin"):
                            if lib_sub.exists():
                                candidate_dirs.append(lib_sub)

        for entry in list(sys.path):
            if not entry:
                continue
            ep = Path(entry)
            nv = ep / "nvidia"
            if nv.exists():
                for pkg_dir in nv.glob("*"):
                    for lib_sub in (pkg_dir / "lib", pkg_dir / "bin"):
                        if lib_sub.exists():
                            candidate_dirs.append(lib_sub)
            for trt_pkg in ("tensorrt_libs", "tensorrt"):
                td = ep / trt_pkg
                if td.exists():
                    candidate_dirs.append(td)

        unique_dirs: list[Path] = []
        seen: set[str] = set()
        for d in candidate_dirs:
            if d.exists():
                resolved = str(d.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    unique_dirs.append(Path(resolved))

        if not unique_dirs:
            return

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
            dir_strs = [str(d) for d in unique_dirs]
            existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                dir_strs + ([existing_ld] if existing_ld else [])
            )
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


_configure_bundled_runtime_libs()

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.styles import DARK_TACTICAL_STYLE

logger = logging.getLogger(__name__)


def main() -> int:
    """Initialize and run the Qt application event loop.

    Returns:
        Application exit code.
    """
    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Aerial Reconnaissance Workstation")
    app.setOrganizationName("Defense Intelligence Systems")

    # Dark tactical base palette to eliminate any unstyled white fallback areas (QPalette.Base = #080f1e)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0b1326"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#080f1e"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#0c162d"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#0f1d3a"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#0f1d3a"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#38bdf8"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#1e3a6e"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(DARK_TACTICAL_STYLE)

    # Verify model weights availability (download if missing)
    try:
        from core.config_manager import get_config
        from scripts.download_weights import ensure_models

        cfg = get_config()
        models_path = repo_root / cfg.models_dir
        if not models_path.exists() or not any(models_path.glob("*.onnx")):
            ensure_models(models_dir=models_path)
    except Exception as exc:
        logger.warning("Failed to auto-verify weights on startup: %s", exc)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

