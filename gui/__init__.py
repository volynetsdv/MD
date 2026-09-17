"""GUI Workstation Package for Ultra-High-Resolution Aerial Image Processing.

Provides:
    - CanvasViewer: High-performance QGraphicsView with hardware acceleration
      and non-destructive vector detection overlay.
    - MainWindow: Operator workstation main window with modern dark theme.
"""

from gui.async_worker import InferenceWorker
from gui.canvas_viewer import CanvasViewer, DetectionItem
from gui.main_window import MainWindow

__all__ = ["CanvasViewer", "DetectionItem", "InferenceWorker", "MainWindow"]

