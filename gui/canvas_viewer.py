"""Custom high-performance canvas viewer for ultra-high-resolution aerial imagery.

Provides smooth zooming and panning for images up to 8K (7680x4320) using hardware
acceleration (QOpenGLWidget viewport) with automatic fallback for software/offscreen
environments. Implements a strictly non-destructive vector overlay for object detections.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QWidget,
)

logger = logging.getLogger(__name__)

# High-contrast tactical color palette mapped by class_id
TACTICAL_PALETTE: Dict[int, QColor] = {
    0: QColor("#00FF66"),  # Neon Green   - Light vehicle / Person
    1: QColor("#00E5FF"),  # Neon Cyan    - Heavy truck / Transport
    2: QColor("#FFD600"),  # Bright Gold  - Armored / Combat Vehicle
    3: QColor("#FF3366"),  # Neon Crimson - Artillery / Air Defense / High Value
    4: QColor("#B388FF"),  # Soft Violet  - Aircraft / UAV
    5: QColor("#FF9100"),  # Amber Orange - Infrastructure / Building
    6: QColor("#76FF03"),  # Lime Green   - Secondary target
    7: QColor("#E040FB"),  # Bright Pink  - Unclassified / Radar
}

DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "Легкова техніка",
    1: "Вантажний транспорт",
    2: "Бронетехніка",
    3: "Артилерія / ППО",
    4: "Авіація / БПЛА",
    5: "Інфраструктура",
}


def get_class_color(class_id: int) -> QColor:
    """Return a consistent high-visibility color for the given class_id."""
    if class_id in TACTICAL_PALETTE:
        return TACTICAL_PALETTE[class_id]
    # Deterministic golden-ratio hue generation for arbitrary classes
    hue = int((class_id * 137.508) % 360)
    return QColor.fromHsv(hue, 230, 255)


class DetectionItem:
    """Container holding vector graphical primitives for a single detection."""

    def __init__(
        self,
        raw_data: Dict[str, Any],
        rect_item: QGraphicsRectItem,
        badge_bg: QGraphicsRectItem,
        text_item: QGraphicsSimpleTextItem,
        class_id: int,
        confidence: float,
        rect: QRectF,
    ) -> None:
        self.raw_data = raw_data
        self.rect_item = rect_item
        self.badge_bg = badge_bg
        self.text_item = text_item
        self.class_id = class_id
        self.confidence = confidence
        self.rect = rect

    def set_visible(self, visible: bool) -> None:
        """Toggle visibility for all vector primitives of this detection."""
        self.rect_item.setVisible(visible)
        self.badge_bg.setVisible(visible)
        self.text_item.setVisible(visible)

    def is_visible(self) -> bool:
        return self.rect_item.isVisible()

    def remove_from_scene(self, scene: QGraphicsScene) -> None:
        """Remove graphical items from the scene cleanly."""
        scene.removeItem(self.rect_item)
        scene.removeItem(self.badge_bg)
        scene.removeItem(self.text_item)


class CanvasViewer(QGraphicsView):
    """High-performance QGraphicsView for viewing and navigating 8K aerial imagery.

    Features:
        - Non-destructive vector overlay on top of pristine QGraphicsPixmapItem.
        - Hardware accelerated OpenGL viewport with safe fallback for headless/software envs.
        - Smooth cursor-centered wheel zoom with scale limits (0.02x to 50x).
        - Smooth pan via middle mouse button or left mouse drag.
        - Dynamic filtering by minimum confidence and class visibility.
    """

    # Signals
    zoom_changed = Signal(float)  # current scale (1.0 = 100%)
    cursor_position_changed = Signal(int, int)  # scene (x, y) pixel coordinates
    detection_selected = Signal(dict)  # emitted when a detection box is clicked
    detections_updated = Signal(int)  # total visible detections count

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Background raster layer
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._original_pixmap: Optional[QPixmap] = None

        # Detection items storage
        self._detection_items: List[DetectionItem] = []
        self._detections_raw: List[Dict[str, Any]] = []

        # Filtering parameters
        self._min_confidence: float = 0.0
        self._visible_classes: Set[int] = set(range(100))  # all visible by default
        self._class_names: Dict[int, str] = dict(DEFAULT_CLASS_NAMES)

        # Navigation state
        self._current_zoom: float = 1.0
        self._min_zoom: float = 0.02
        self._max_zoom: float = 50.0
        self._is_panning: bool = False
        self._pan_start_pos: QPoint = QPoint()

        # Viewport and hardware acceleration
        self._is_opengl_active: bool = False
        self._setup_rendering_pipeline()

    # --------------------------------------------------------------------------
    # Rendering and Hardware Acceleration Pipeline
    # --------------------------------------------------------------------------
    def _setup_rendering_pipeline(self) -> None:
        """Initialize OpenGL viewport with graceful software fallback."""
        # Visual quality and performance flags
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor("#0b0f19")))  # Tactical deep navy background

        # Attempt OpenGL acceleration
        self._init_opengl_viewport()

    def _init_opengl_viewport(self) -> None:
        """Attempt to use QOpenGLWidget as viewport, falling back to standard QWidget."""
        # Detect purely offscreen or software environments
        platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
        if "offscreen" in platform or "minimal" in platform:
            logger.info("CanvasViewer: Headless/offscreen platform detected; using raster viewport.")
            self._is_opengl_active = False
            return

        try:
            from PySide6.QtOpenGLWidgets import QOpenGLWidget

            gl_widget = QOpenGLWidget()
            self.setViewport(gl_widget)
            self._is_opengl_active = True
            logger.info("CanvasViewer: OpenGL hardware accelerated viewport successfully initialized.")
        except Exception as exc:
            logger.warning(
                "CanvasViewer: Failed to initialize QOpenGLWidget (%s). "
                "Falling back to standard raster viewport.",
                exc,
            )
            self.setViewport(QWidget())
            self._is_opengl_active = False

    @property
    def is_opengl_accelerated(self) -> bool:
        """Check if OpenGL hardware acceleration is active."""
        return self._is_opengl_active

    # --------------------------------------------------------------------------
    # Image Management (Non-destructive)
    # --------------------------------------------------------------------------
    def load_image(self, file_path: str) -> bool:
        """Load an image file safely into the canvas.

        Args:
            file_path: Absolute or relative path to image file.

        Returns:
            True if image loaded successfully, False otherwise.
        """
        if not os.path.exists(file_path):
            logger.error("CanvasViewer: File does not exist: %s", file_path)
            return False

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            logger.error("CanvasViewer: Failed to decode image file: %s", file_path)
            return False

        self.set_image(pixmap)
        return True

    def set_image(self, pixmap: QPixmap) -> None:
        """Display an aerial image in the background layer without altering its pixels.

        Args:
            pixmap: High-resolution QPixmap (up to 8K, e.g. 7680x4320).
        """
        self._original_pixmap = pixmap

        # Remove existing pixmap item if present
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None

        # Clean detections for the new image
        self.clear_detections()

        # Create pristine background raster layer at Z=0
        self._pixmap_item = QGraphicsPixmapItem(self._original_pixmap)
        self._pixmap_item.setZValue(0.0)
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)

        # Set scene bounding box strictly to match image dimensions
        img_rect = QRectF(0.0, 0.0, float(pixmap.width()), float(pixmap.height()))
        self._scene.setSceneRect(img_rect)

        # Fit image in view initially
        self.fit_to_view()

    def get_image_size(self) -> Tuple[int, int]:
        """Return (width, height) in pixels of current image, or (0, 0) if none loaded."""
        if self._original_pixmap and not self._original_pixmap.isNull():
            return self._original_pixmap.width(), self._original_pixmap.height()
        return 0, 0

    def has_image(self) -> bool:
        """Check if an image is currently loaded."""
        return self._pixmap_item is not None and not self._original_pixmap.isNull()

    # --------------------------------------------------------------------------
    # Non-Destructive Vector Overlay
    # --------------------------------------------------------------------------
    def update_detections(self, detections_json: List[Dict[str, Any]]) -> None:
        """Add non-destructive vector overlays for detection results.

        Accepts absolute pixel coordinates corresponding to the global image dimensions
        (e.g., 8K space: x=4200, y=1800, w=64, h=48) as computed by C++ Offset Mapping.

        Args:
            detections_json: List of detection dictionaries. Expected keys:
                - Coordinates: either ('x', 'y', 'w', 'h') or ('xmin', 'ymin', 'xmax', 'ymax')
                - 'conf': confidence score float [0.0, 1.0]
                - 'class_id': integer class index
                - Optional: 'class_name'
        """
        self.clear_detections()
        self._detections_raw = list(detections_json)

        for det in detections_json:
            item = self._create_detection_primitive(det)
            if item is not None:
                self._detection_items.append(item)

        # Apply current active filters
        self._apply_filters()
        self._emit_visible_count()

    def _parse_coordinates(self, det: Dict[str, Any]) -> Optional[QRectF]:
        """Extract absolute pixel coordinates from detection dictionary."""
        try:
            if "x" in det and "y" in det and "w" in det and "h" in det:
                x = float(det["x"])
                y = float(det["y"])
                w = float(det["w"])
                h = float(det["h"])
            elif "xmin" in det and "ymin" in det and "xmax" in det and "ymax" in det:
                x = float(det["xmin"])
                y = float(det["ymin"])
                w = float(det["xmax"]) - x
                h = float(det["ymax"]) - y
            else:
                logger.warning("Detection missing coordinate keys: %s", det)
                return None

            if w <= 0 or h <= 0:
                return None
            return QRectF(x, y, w, h)
        except (ValueError, TypeError) as err:
            logger.warning("Invalid coordinate values in detection: %s (%s)", det, err)
            return None

    def _create_detection_primitive(self, det: Dict[str, Any]) -> Optional[DetectionItem]:
        """Construct QGraphicsRectItem and QGraphicsSimpleTextItem for a detection."""
        rect = self._parse_coordinates(det)
        if rect is None:
            return None

        class_id = int(det.get("class_id", 0))
        conf = float(det.get("conf", det.get("confidence", 0.0)))
        class_name = det.get("class_name", self._class_names.get(class_id, f"Клас {class_id}"))

        color = get_class_color(class_id)

        # 1. Bounding box rectangle: cosmetic pen preserves crispness at any zoom level
        rect_item = QGraphicsRectItem(rect)
        pen = QPen(color, 2.0)
        pen.setCosmetic(True)  # Keeps line width constant on screen regardless of zoom
        rect_item.setPen(pen)

        # Subtle translucent fill to highlight target area
        fill_color = QColor(color.red(), color.green(), color.blue(), 28)
        rect_item.setBrush(QBrush(fill_color))
        rect_item.setZValue(10.0)

        tooltip_text = (
            f"ID: {det.get('id', '-')}\n"
            f"Клас: {class_name} ({class_id})\n"
            f"Впевненість: {conf:.1%}\n"
            f"Координати: X={int(rect.x())}, Y={int(rect.y())}, W={int(rect.width())}, H={int(rect.height())}"
        )
        rect_item.setToolTip(tooltip_text)

        # 2. Text label and contrast background pill for readability over any terrain
        label_text = f"{class_name} {conf:.1%}"
        font = QFont("SansSerif", 10, QFont.Weight.Bold)

        text_item = QGraphicsSimpleTextItem(label_text)
        text_item.setFont(font)
        text_item.setBrush(QBrush(Qt.GlobalColor.white))
        text_item.setZValue(12.0)

        # Text bounding dimensions
        text_rect = text_item.boundingRect()
        badge_padding_x = 4.0
        badge_padding_y = 2.0
        badge_w = text_rect.width() + (badge_padding_x * 2.0)
        badge_h = text_rect.height() + (badge_padding_y * 2.0)

        # Position badge above top-left of box, or inside if too close to image top edge
        badge_x = rect.x()
        badge_y = rect.y() - badge_h
        if badge_y < 0:
            badge_y = rect.y()

        badge_rect = QRectF(badge_x, badge_y, badge_w, badge_h)
        badge_bg = QGraphicsRectItem(badge_rect)

        # Dark high-contrast background with colored accent border
        bg_color = QColor(15, 23, 42, 220)  # semi-opaque dark slate
        badge_pen = QPen(color, 1.0)
        badge_pen.setCosmetic(True)
        badge_bg.setPen(badge_pen)
        badge_bg.setBrush(QBrush(bg_color))
        badge_bg.setZValue(11.0)

        # Set text position inside badge
        text_item.setPos(badge_x + badge_padding_x, badge_y + badge_padding_y)

        # Add all vector items to scene
        self._scene.addItem(rect_item)
        self._scene.addItem(badge_bg)
        self._scene.addItem(text_item)

        return DetectionItem(
            raw_data=det,
            rect_item=rect_item,
            badge_bg=badge_bg,
            text_item=text_item,
            class_id=class_id,
            confidence=conf,
            rect=rect,
        )

    def clear_detections(self) -> None:
        """Remove all detection vector overlay items from the scene."""
        for item in self._detection_items:
            item.remove_from_scene(self._scene)
        self._detection_items.clear()
        self._detections_raw.clear()
        self._emit_visible_count()

    def get_detections(self) -> List[Dict[str, Any]]:
        """Return raw list of all loaded detections."""
        return list(self._detections_raw)

    def get_visible_detections(self) -> List[Dict[str, Any]]:
        """Return list of currently visible detections based on active filters."""
        return [item.raw_data for item in self._detection_items if item.is_visible()]

    # --------------------------------------------------------------------------
    # Filtering (Classes & Confidence)
    # --------------------------------------------------------------------------
    def set_min_confidence(self, min_conf: float) -> None:
        """Set minimum confidence threshold [0.0, 1.0] and update visibility."""
        self._min_confidence = max(0.0, min(1.0, float(min_conf)))
        self._apply_filters()
        self._emit_visible_count()

    def set_class_visibility(self, class_id: int, visible: bool) -> None:
        """Toggle visibility for a specific class_id."""
        if visible:
            self._visible_classes.add(class_id)
        else:
            self._visible_classes.discard(class_id)
        self._apply_filters()
        self._emit_visible_count()

    def set_all_classes_visibility(self, visible: bool) -> None:
        """Show or hide all classes."""
        if visible:
            # Re-add all known classes
            for item in self._detection_items:
                self._visible_classes.add(item.class_id)
        else:
            self._visible_classes.clear()
        self._apply_filters()
        self._emit_visible_count()

    def set_class_names(self, names: Dict[int, str]) -> None:
        """Update class ID to name dictionary."""
        self._class_names.update(names)

    def _apply_filters(self) -> None:
        """Update visibility of each detection item according to current filters."""
        for item in self._detection_items:
            visible = (item.confidence >= self._min_confidence) and (
                item.class_id in self._visible_classes
            )
            item.set_visible(visible)

    def _emit_visible_count(self) -> None:
        """Emit count of currently visible detections."""
        count = sum(1 for item in self._detection_items if item.is_visible())
        self.detections_updated.emit(count)

    # --------------------------------------------------------------------------
    # Navigation: Smooth Zoom & Pan
    # --------------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Smooth zoom anchored precisely to mouse cursor position."""
        if not self.has_image():
            super().wheelEvent(event)
            return

        angle_delta = event.angleDelta().y()
        if angle_delta == 0:
            return

        zoom_factor = 1.15 if angle_delta > 0 else (1.0 / 1.15)
        new_zoom = self._current_zoom * zoom_factor

        # Clamp zoom to prevent degenerate transformations
        if new_zoom < self._min_zoom:
            zoom_factor = self._min_zoom / self._current_zoom
            new_zoom = self._min_zoom
        elif new_zoom > self._max_zoom:
            zoom_factor = self._max_zoom / self._current_zoom
            new_zoom = self._max_zoom

        if abs(zoom_factor - 1.0) < 1e-4:
            return

        # Cursor-centered scaling math
        mouse_viewport_pos = event.position().toPoint()
        scene_pos = self.mapToScene(mouse_viewport_pos)

        self.scale(zoom_factor, zoom_factor)
        self._current_zoom = new_zoom

        new_viewport_pos = self.mapFromScene(scene_pos)
        viewport_delta = new_viewport_pos - mouse_viewport_pos

        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() + viewport_delta.x()
        )
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() + viewport_delta.y()
        )

        self.zoom_changed.emit(self._current_zoom)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle panning initiation and detection selection."""
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            # Check if clicked on a detection item first
            item = self.itemAt(event.position().toPoint())
            if item is not None and item != self._pixmap_item:
                # Find matching detection
                for det_item in self._detection_items:
                    if item in (det_item.rect_item, det_item.badge_bg, det_item.text_item):
                        self.detection_selected.emit(det_item.raw_data)
                        break

            # Initiate panning
            self._is_panning = True
            self._pan_start_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle smooth drag panning and emit scene cursor coordinate telemetry."""
        # Telemetry: emit image coordinates under cursor
        scene_pt = self.mapToScene(event.position().toPoint())
        self.cursor_position_changed.emit(int(scene_pt.x()), int(scene_pt.y()))

        if self._is_panning:
            current_pos = event.position().toPoint()
            delta = current_pos - self._pan_start_pos
            self._pan_start_pos = current_pos

            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End panning drag."""
        if self._is_panning and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def fit_to_view(self) -> None:
        """Fit entire image within the current viewport preserving aspect ratio."""
        if not self.has_image() or self._original_pixmap is None:
            return

        img_rect = QRectF(0.0, 0.0, float(self._original_pixmap.width()), float(self._original_pixmap.height()))
        self.fitInView(img_rect, Qt.AspectRatioMode.KeepAspectRatio)

        # Update current zoom level from transform
        transform = self.transform()
        self._current_zoom = transform.m11()
        self.zoom_changed.emit(self._current_zoom)

    def reset_zoom(self) -> None:
        """Reset view to 100% original scale (1 pixel on image = 1 screen pixel)."""
        if not self.has_image():
            return
        self.resetTransform()
        self._current_zoom = 1.0
        self.zoom_changed.emit(self._current_zoom)

    def focus_on_detection(self, index: int, target_zoom: Optional[float] = 1.5) -> bool:
        """Center the viewport on a specific detection by index and optionally zoom in."""
        if index < 0 or index >= len(self._detection_items):
            return False

        item = self._detection_items[index]
        center = item.rect.center()

        if target_zoom is not None and target_zoom > 0:
            scale_ratio = target_zoom / max(1e-4, self._current_zoom)
            self.scale(scale_ratio, scale_ratio)
            self._current_zoom = target_zoom
            self.zoom_changed.emit(self._current_zoom)

        self.centerOn(center)
        return True

    def get_zoom_level(self) -> float:
        """Return current zoom level factor."""
        return self._current_zoom

