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

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

