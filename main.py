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

    # Dark tactical base palette to eliminate any unstyled white fallback areas (QPalette.Base = #080f1e)
    from PySide6.QtGui import QColor, QPalette
    from gui.styles import DARK_TACTICAL_STYLE

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
        import logging

        logging.getLogger("Main").warning("Failed to auto-verify weights on startup: %s", exc)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

