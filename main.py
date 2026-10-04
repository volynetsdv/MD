"""Application Entry Point for Aerial Reconnaissance Workstation."""

import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> int:
    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Aerial Reconnaissance Workstation")
    app.setOrganizationName("Defense Intelligence Systems")

    # Verify model weights availability (download if missing)
    try:
        from core.config_manager import get_config
        from scripts.download_weights import ensure_models

        cfg = get_config()
        models_path = repo_root / cfg.models_dir
        if not models_path.exists() or not any(models_path.glob("*.onnx")):
            ensure_models(models_dir=models_path)
    except Exception as exc:
        import logging

        logging.getLogger("Main").warning("Failed to auto-verify weights on startup: %s", exc)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

