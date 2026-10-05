"""Asynchronous Inference Worker for PySide6 Operator Workstation.

Executes the full inference pipeline on a background QThread:
1. Zero-copy sub-tensor / NumPy image tile slicing.
2. Hardware-aware dynamic tiling calculation (via pytiling_core or Python fallback).
3. Tile inference through UnifiedDetector (TensorRT or DirectML/CUDA ONNX Runtime).
4. Real-time progress updates after each tile.
5. Offset mapping and Cluster-DIoU-NMS boundary merging.
6. Immediate intermediate VRAM memory deallocation.
7. Thread-safe cancellation and error handling.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

# Safe import of pytiling_core and UnifiedDetector
from gui.canvas_viewer import DEFAULT_CLASS_NAMES
from src.detector_dispatcher import Detection, UnifiedDetector

logger = logging.getLogger("InferenceWorker")

# Attempt pytiling_core import
_pytiling_available = False
try:
    import pytiling_core

    _pytiling_available = True
except ImportError:
    repo_root = Path(__file__).resolve().parent.parent
    for cand in [repo_root / "build", repo_root / "build" / "Release", repo_root / "build" / "bindings"]:
        if cand.exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
            try:
                import pytiling_core

                _pytiling_available = True
                break
            except ImportError:
                pass


# =============================================================================
# Python Fallback Math (DRY match to tiling_math.cpp & postprocess.cpp)
# =============================================================================
@dataclass
class Rect:
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


@dataclass
class GlobalDet:
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    conf: float = 0.0
    class_id: int = 0
    tile_id: int = -1


def py_calculate_tiling_params(
    width: int, height: int, altitude: float, vram_mb: int
) -> Tuple[int, List[Rect]]:
    """Python fallback for calculate_tiling_params if C++ module is unavailable."""
    is_ultra_large = width >= 5000 or height >= 5000
    effective_slice_size = 736 if is_ultra_large else 640

    grid = [320, 416, 512, 640]
    t_calc = (vram_mb / 4.0) + (altitude * 10.0)

    # Pick tile size (largest <= t_calc)
    tile_size = grid[0]
    for s in reversed(grid):
        if s <= math.floor(t_calc):
            tile_size = s
            break

    if tile_size == 640 and is_ultra_large:
        tile_size = effective_slice_size

    # Overlap clamped [0.1, 0.4]
    overlap = max(0.1, min(0.4, 0.1 + (altitude / 500.0)))
    step_x = max(1, int(round(tile_size * (1.0 - overlap))))
    step_y = step_x

    cols = (width + step_x - 1) // step_x
    rows = (height + step_y - 1) // step_y

    tiles: List[Rect] = []
    for r in range(rows):
        y = r * step_y
        if y >= height:
            break
        h = min(tile_size, height - y)
        for c in range(cols):
            x = c * step_x
            if x >= width:
                break
            w = min(tile_size, width - x)
            tiles.append(Rect(x=x, y=y, w=w, h=h))

    return tile_size, tiles


def py_calculate_diou(a: GlobalDet, b: GlobalDet) -> float:
    """Calculate DIoU between two detections."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - inter_area
    if union <= 1e-7:
        return 0.0
    iou = inter_area / union

    # Centers
    ca_x = a.x + a.w * 0.5
    ca_y = a.y + a.h * 0.5
    cb_x = b.x + b.w * 0.5
    cb_y = b.y + b.h * 0.5
    d2 = (ca_x - cb_x) ** 2 + (ca_y - cb_y) ** 2

    # Enclosing box
    enc_x1 = min(a.x, b.x)
    enc_y1 = min(a.y, b.y)
    enc_x2 = max(a.x + a.w, b.x + b.w)
    enc_y2 = max(a.y + a.h, b.y + b.h)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

    if c2 <= 1e-7:
        return iou
    return iou - (d2 / c2)


def py_cluster_diou_nms(
    detections: List[GlobalDet],
    tiles: List[Any],
    diou_threshold: float = 0.5,
    conf_threshold: float = 0.0,
) -> List[GlobalDet]:
    """Lightweight, ultra-fast Greedy DIoU-NMS implementation."""
    candidates = [
        d for d in detections if d.conf >= conf_threshold and d.w > 0.0 and d.h > 0.0
    ]
    if not candidates:
        return []

    # Hard cap proposals to top-800 candidates sorted by confidence
    if len(candidates) > 800:
        candidates.sort(key=lambda x: x.conf, reverse=True)
        candidates = candidates[:800]

    # Group by class
    classes = sorted({d.class_id for d in candidates})
    result: List[GlobalDet] = []

    for cls in classes:
        cls_boxes = [d for d in candidates if d.class_id == cls]
        cls_boxes.sort(key=lambda x: x.conf, reverse=True)

        n = len(cls_boxes)
        if n == 1:
            result.append(cls_boxes[0])
            continue

        # In-place float32 coordinate arrays: [x1, y1, x2, y2]
        boxes = np.empty((n, 4), dtype=np.float32)
        for i, d in enumerate(cls_boxes):
            boxes[i, 0] = d.x
            boxes[i, 1] = d.y
            boxes[i, 2] = d.x + d.w
            boxes[i, 3] = d.y + d.h

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5

        suppressed = np.zeros(n, dtype=bool)

        for i in range(n):
            if suppressed[i]:
                continue
            result.append(cls_boxes[i])

            rem_mask = ~suppressed[i + 1:]
            if not np.any(rem_mask):
                break

            rem_idx = np.flatnonzero(rem_mask) + (i + 1)

            # Vectorized 1D intersection check with remaining candidates
            xx1 = np.maximum(boxes[i, 0], boxes[rem_idx, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rem_idx, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rem_idx, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rem_idx, 3])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter_area = inter_w * inter_h

            overlap_mask = inter_area > 0.0
            if not np.any(overlap_mask):
                continue

            sub_idx = rem_idx[overlap_mask]
            sub_inter = inter_area[overlap_mask]

            union = areas[i] + areas[sub_idx] - sub_inter
            iou = np.where(union > 1e-7, sub_inter / union, 0.0)

            d2 = (
                (centers_x[i] - centers_x[sub_idx]) ** 2
                + (centers_y[i] - centers_y[sub_idx]) ** 2
            )
            enc_x1 = np.minimum(boxes[i, 0], boxes[sub_idx, 0])
            enc_y1 = np.minimum(boxes[i, 1], boxes[sub_idx, 1])
            enc_x2 = np.maximum(boxes[i, 2], boxes[sub_idx, 2])
            enc_y2 = np.maximum(boxes[i, 3], boxes[sub_idx, 3])
            c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

            diou = np.where(c2 > 1e-7, iou - (d2 / c2), iou)
            to_suppress = sub_idx[diou >= diou_threshold]
            if to_suppress.size > 0:
                suppressed[to_suppress] = True

    return result


# Synthetic targets across the 8K frame for testing/mock mode (matches main_pipeline.cpp)
SYNTHETIC_TARGETS = [
    {"gx": 250.0, "gy": 180.0, "gw": 54.0, "gh": 32.0, "class_id": 2, "conf": 0.91},
    {"gx": 340.0, "gy": 290.0, "gw": 62.0, "gh": 38.0, "class_id": 2, "conf": 0.88},
    {"gx": 390.0, "gy": 370.0, "gw": 48.0, "gh": 28.0, "class_id": 2, "conf": 0.94},
    {"gx": 720.0, "gy": 410.0, "gw": 22.0, "gh": 44.0, "class_id": 0, "conf": 0.86},
    {"gx": 1250.0, "gy": 800.0, "gw": 110.0, "gh": 75.0, "class_id": 4, "conf": 0.95},
]


# =============================================================================
# Asynchronous Inference Worker
# =============================================================================
class InferenceWorker(QThread):
    """Background worker executing the complete asynchronous aerial detection cycle."""

    progress_changed = Signal(int, int, str)  # current, total, status_message
    detection_completed = Signal(list, float)  # list of detection dicts, execution_time_ms
    detection_finished = detection_completed  # alias for backwards compatibility
    error_occurred = Signal(str)  # error_message

    def __init__(
        self,
        image_source: Union[str, Path, np.ndarray],
        altitude: float = 120.0,
        vram_mb: int = 2048,
        conf_threshold: float = 0.20,
        diou_threshold: float = 0.50,
        detector: Optional[UnifiedDetector] = None,
        is_mock: bool = False,
        parent: Optional[QObject] = None,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        super().__init__(parent)
        self.image_source = image_source
        self.altitude = float(altitude)
        self.vram_mb = int(vram_mb)
        self.conf_threshold = float(conf_threshold)
        self.diou_threshold = float(diou_threshold)
        self.detector = detector
        self.is_mock = is_mock
        self.class_names = dict(class_names) if class_names else None

        self._is_cancelled: bool = False

    def get_class_names(self) -> Dict[int, str]:
        """Retrieve active class ID to name dictionary with fallback."""
        if self.class_names:
            return dict(self.class_names)
        if self.detector is not None and hasattr(self.detector, "get_class_names"):
            try:
                return self.detector.get_class_names()
            except Exception:
                pass
        return dict(DEFAULT_CLASS_NAMES)

    def cancel(self) -> None:
        """Request thread cancellation."""
        self._is_cancelled = True
        logger.info("InferenceWorker: Cancellation requested.")

    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def run(self) -> None:
        """Execute image loading, tiling, inference, offset remapping, and NMS."""
        start_time = time.perf_counter()
        try:
            # 1. Load image into memory
            self.progress_changed.emit(0, 100, "Завантаження знімка у пам'ять...")
            img = self._load_image_data()
            if img is None:
                self.error_occurred.emit("Не вдалося завантажити зображення для обробки.")
                return

            img_h, img_w = img.shape[:2]
            logger.info("InferenceWorker: Processing image (%dx%d px)...", img_w, img_h)

            if self._is_cancelled:
                self.progress_changed.emit(0, 100, "Скасовано користувачем.")
                return

            # 2. Dynamic Tiling Calculation
            tile_size, tiles = self._calculate_tiles(img_w, img_h)
            total_tiles = len(tiles)
            if total_tiles == 0:
                self.error_occurred.emit("Помилка генерації сітки плиток: отримано 0 тайлів.")
                return

            model_size = 640 if tile_size == 736 else tile_size

            logger.info(
                "InferenceWorker: Generated %d tiles with tile size %d (model size %d, Alt: %.1fm, VRAM: %dMB).",
                total_tiles,
                tile_size,
                model_size,
                self.altitude,
                self.vram_mb,
            )
            self.progress_changed.emit(
                0,
                total_tiles,
                f"Розраховано сітку: {total_tiles} тайлів (модель yolo_{model_size}). Початок детекції...",
            )

            # 3. Initialize Unified Detector if not provided
            if self.detector is None:
                self.detector = UnifiedDetector()

            # 4. Sequentially process tiles
            all_global_detections: List[Any] = []

            for idx, tile in enumerate(tiles):
                if self._is_cancelled:
                    logger.info("InferenceWorker: Process stopped at tile %d/%d.", idx, total_tiles)
                    self.progress_changed.emit(
                        idx, total_tiles, "Детекцію зупинено оператором."
                    )
                    return

                # Zero-Copy slice from image buffer
                tx, ty, tw, th = tile.x, tile.y, tile.w, tile.h
                tile_slice = img[ty : ty + th, tx : tx + tw]

                # Bilinear resize to 640x640 exclusively for 736px tiles
                if tw == 736 and th == 736:
                    tile_slice = cv2.resize(
                        tile_slice, (640, 640), interpolation=cv2.INTER_LINEAR
                    )

                # Run inference on tile
                tile_dets = []
                if self.is_mock:
                    # In mock testing mode, check intersection with synthetic targets
                    scale_x = float(model_size) / float(tile.w)
                    scale_y = float(model_size) / float(tile.h)
                    for st in SYNTHETIC_TARGETS:
                        if (
                            st["gx"] + st["gw"] > tile.x
                            and st["gx"] < tile.x + tile.w
                            and st["gy"] + st["gh"] > tile.y
                            and st["gy"] < tile.y + tile.h
                        ):
                            tile_dets.append(
                                Detection(
                                    x_local=(st["gx"] - float(tile.x)) * scale_x,
                                    y_local=(st["gy"] - float(tile.y)) * scale_y,
                                    w=st["gw"] * scale_x,
                                    h=st["gh"] * scale_y,
                                    conf=st["conf"],
                                    class_id=st["class_id"],
                                )
                            )
                    # Short sleep for UI animation pacing
                    time.sleep(0.002)
                else:
                    tile_dets = self.detector.predict_tile(
                        tile_slice,
                        tile_size=model_size,
                        conf_threshold=self.conf_threshold,
                    )

                # Immediately release tile view
                del tile_slice

                # Remap tile-local detections to global frame coordinates
                for d in tile_dets:
                    global_det = self._remap_detection(d, tile, model_size, idx)
                    all_global_detections.append(global_det)

                # Real-time progress signal
                status_msg = f"Обробка тайла {idx + 1}/{total_tiles} (знайдено: {len(all_global_detections)})..."
                self.progress_changed.emit(idx + 1, total_tiles, status_msg)

            if self._is_cancelled:
                return

            # 5. Post-Processing: Cluster-DIoU-NMS across tile boundaries
            self.progress_changed.emit(
                total_tiles, total_tiles, "Виконується Cluster-DIoU-NMS злиття меж..."
            )
            merged_detections = self._merge_boundary_detections(
                all_global_detections, tiles
            )

            # 6. Format final detections for UI
            final_json = self._format_results_for_gui(merged_detections)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "InferenceWorker: Completed in %.2f ms. Total merged targets: %d.",
                elapsed_ms,
                len(final_json),
            )

            self.detection_completed.emit(final_json, elapsed_ms)

        except Exception as exc:
            logger.exception("InferenceWorker: Exception occurred during inference: %s", exc)
            self.error_occurred.emit(f"Помилка інференсу: {exc}")

    def _load_image_data(self) -> Optional[np.ndarray]:
        """Load image as RGB NumPy uint8 array."""
        if isinstance(self.image_source, np.ndarray):
            return self.image_source

        file_str = str(self.image_source)
        if not os.path.exists(file_str):
            logger.error("InferenceWorker: Image file does not exist: %s", file_str)
            return None

        # Load with OpenCV
        bgr = cv2.imread(file_str, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.error("InferenceWorker: OpenCV failed to read: %s", file_str)
            return None

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _calculate_tiles(self, width: int, height: int) -> Tuple[int, List[Any]]:
        """Calculate dynamic tiling plan using pytiling_core or Python fallback."""
        if _pytiling_available:
            cfg = pytiling_core.calculate_tiling_params(
                width=width,
                height=height,
                altitude=self.altitude,
                vram_mb=self.vram_mb,
            )
            return cfg.tile_size, list(cfg.tiles)

        return py_calculate_tiling_params(
            width=width,
            height=height,
            altitude=self.altitude,
            vram_mb=self.vram_mb,
        )

    def _remap_detection(
        self, det: Detection, tile: Any, model_size: int, tile_id: int
    ) -> Any:
        """Remap tile-local detection to global frame coordinates."""
        if _pytiling_available and hasattr(pytiling_core, "remap_offsets"):
            return pytiling_core.remap_offsets(
                det.to_cpp(),
                tile,
                model_size,
                tile_id,
            )

        # Python fallback formula:
        scale_x = float(tile.w) / float(model_size)
        scale_y = float(tile.h) / float(model_size)
        gx = float(tile.x) + det.x_local * scale_x
        gy = float(tile.y) + det.y_local * scale_y
        gw = det.w * scale_x
        gh = det.h * scale_y

        return GlobalDet(
            x=gx,
            y=gy,
            w=gw,
            h=gh,
            conf=det.conf,
            class_id=det.class_id,
            tile_id=tile_id,
        )

    def _merge_boundary_detections(
        self, detections: List[Any], tiles: List[Any]
    ) -> List[Any]:
        """Perform Cluster-DIoU-NMS to merge overlapping detections."""
        if not detections:
            return []

        if _pytiling_available and hasattr(pytiling_core, "cluster_diou_nms"):
            try:
                return pytiling_core.cluster_diou_nms(
                    detections,
                    tiles,
                    self.diou_threshold,
                    self.conf_threshold,
                )
            except Exception as e:
                logger.warning("pytiling_core.cluster_diou_nms failed (%s), using fallback.", e)

        # Python fallback
        py_dets = []
        for d in detections:
            if isinstance(d, GlobalDet):
                py_dets.append(d)
            else:
                py_dets.append(
                    GlobalDet(
                        x=float(d.x),
                        y=float(d.y),
                        w=float(d.w),
                        h=float(d.h),
                        conf=float(d.conf),
                        class_id=int(d.class_id),
                        tile_id=int(getattr(d, "tile_id", -1)),
                    )
                )

        return py_cluster_diou_nms(
            py_dets,
            tiles,
            self.diou_threshold,
            self.conf_threshold,
        )

    def _format_results_for_gui(self, detections: List[Any]) -> List[Dict[str, Any]]:
        """Convert global detection objects to clean dictionary representations."""
        return format_detections_for_gui(detections, class_names=self.get_class_names())


def calculate_tiles_for_image(
    width: int, height: int, altitude: float, vram_mb: int
) -> Tuple[int, List[Any]]:
    """Calculate dynamic tiling plan using pytiling_core or Python fallback."""
    if _pytiling_available:
        cfg = pytiling_core.calculate_tiling_params(
            width=width,
            height=height,
            altitude=altitude,
            vram_mb=vram_mb,
        )
        return cfg.tile_size, list(cfg.tiles)

    return py_calculate_tiling_params(
        width=width,
        height=height,
        altitude=altitude,
        vram_mb=vram_mb,
    )


def remap_detection_to_global(
    det: Detection, tile: Any, model_size: int, tile_id: int
) -> Any:
    """Remap tile-local detection to global frame coordinates."""
    if _pytiling_available and hasattr(pytiling_core, "remap_offsets"):
        return pytiling_core.remap_offsets(
            det.to_cpp(),
            tile,
            model_size,
            tile_id,
        )

    # Python fallback formula:
    scale_x = float(tile.w) / float(model_size)
    scale_y = float(tile.h) / float(model_size)
    gx = float(tile.x) + det.x_local * scale_x
    gy = float(tile.y) + det.y_local * scale_y
    gw = det.w * scale_x
    gh = det.h * scale_y

    return GlobalDet(
        x=gx,
        y=gy,
        w=gw,
        h=gh,
        conf=det.conf,
        class_id=det.class_id,
        tile_id=tile_id,
    )


def merge_boundary_detections(
    detections: List[Any],
    tiles: List[Any],
    diou_threshold: float = 0.50,
    conf_threshold: float = 0.0,
) -> List[Any]:
    """Perform Cluster-DIoU-NMS to merge overlapping detections."""
    if not detections:
        return []

    if _pytiling_available and hasattr(pytiling_core, "cluster_diou_nms"):
        try:
            return pytiling_core.cluster_diou_nms(
                detections,
                tiles,
                diou_threshold,
                conf_threshold,
            )
        except Exception as e:
            logger.warning("pytiling_core.cluster_diou_nms failed (%s), using fallback.", e)

    # Python fallback
    py_dets = []
    for d in detections:
        if isinstance(d, GlobalDet):
            py_dets.append(d)
        else:
            py_dets.append(
                GlobalDet(
                    x=float(d.x),
                    y=float(d.y),
                    w=float(d.w),
                    h=float(d.h),
                    conf=float(d.conf),
                    class_id=int(d.class_id),
                    tile_id=int(getattr(d, "tile_id", -1)),
                )
            )

    return py_cluster_diou_nms(
        py_dets,
        tiles,
        diou_threshold,
        conf_threshold,
    )


def format_detections_for_gui(
    detections: List[Any],
    class_names: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Convert global detection objects to clean dictionary representations."""
    active_names = class_names if class_names is not None else DEFAULT_CLASS_NAMES
    results: List[Dict[str, Any]] = []
    for i, det in enumerate(detections):
        if isinstance(det, dict):
            class_id = int(det.get("class_id", 0))
            class_name = str(det.get("class_name", active_names.get(class_id, f"Клас {class_id}")))
            conf = float(det.get("confidence", det.get("conf", 0.0)))
            if "bbox" in det and isinstance(det["bbox"], (list, tuple)) and len(det["bbox"]) == 4:
                x = round(float(det["bbox"][0]), 2)
                y = round(float(det["bbox"][1]), 2)
                w = round(float(det["bbox"][2]), 2)
                h = round(float(det["bbox"][3]), 2)
            else:
                x = round(float(det.get("x", 0.0)), 2)
                y = round(float(det.get("y", 0.0)), 2)
                w = round(float(det.get("w", 0.0)), 2)
                h = round(float(det.get("h", 0.0)), 2)
        elif hasattr(det, "to_dict"):
            d_dict = det.to_dict()
            class_id = int(d_dict.get("class_id", 0))
            class_name = str(d_dict.get("class_name", active_names.get(class_id, f"Клас {class_id}")))
            conf = float(d_dict.get("confidence", d_dict.get("conf", 0.0)))
            x = round(float(d_dict.get("x", 0.0)), 2)
            y = round(float(d_dict.get("y", 0.0)), 2)
            w = round(float(d_dict.get("w", 0.0)), 2)
            h = round(float(d_dict.get("h", 0.0)), 2)
        elif isinstance(det, (list, tuple)) and len(det) >= 6:
            x1, y1, x2, y2, conf_val, cls_id = det[:6]
            class_id = int(cls_id)
            class_name = active_names.get(class_id, f"Клас {class_id}")
            x = round(float(x1), 2)
            y = round(float(y1), 2)
            w = round(float(x2 - x1), 2)
            h = round(float(y2 - y1), 2)
            conf = float(conf_val)
        else:
            class_id = int(det.class_id)
            class_name = active_names.get(class_id, f"Клас {class_id}")
            x = round(float(det.x), 2)
            y = round(float(det.y), 2)
            w = round(float(det.w), 2)
            h = round(float(det.h), 2)
            conf = round(float(det.conf), 4)

        results.append(
            {
                "id": i + 1,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": round(conf, 4),
                "confidence": round(conf, 4),
                "class_id": class_id,
                "class_name": class_name,
                "bbox": [x, y, w, h],
            }
        )
    return results


def run_tiled_inference(
    img_rgb: np.ndarray,
    altitude: float = 120.0,
    vram_mb: int = 2048,
    conf_threshold: float = 0.20,
    diou_threshold: float = 0.50,
    detector: Optional[UnifiedDetector] = None,
    is_mock: bool = False,
    is_cancelled_fn: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Execute complete tiled detection pipeline on an image buffer."""
    img_h, img_w = img_rgb.shape[:2]
    tile_size, tiles = calculate_tiles_for_image(img_w, img_h, altitude, vram_mb)
    total_tiles = len(tiles)
    if total_tiles == 0:
        return []

    model_size = 640 if tile_size == 736 else tile_size

    if detector is None:
        detector = UnifiedDetector()

    all_global_detections: List[Any] = []
    for idx, tile in enumerate(tiles):
        tx, ty, tw, th = tile.x, tile.y, tile.w, tile.h
        tile_slice = img_rgb[ty : ty + th, tx : tx + tw]

        # Bilinear resize to 640x640 exclusively for 736px tiles
        if tw == 736 and th == 736:
            tile_slice = cv2.resize(
                tile_slice, (640, 640), interpolation=cv2.INTER_LINEAR
            )

        tile_dets = []
        if is_mock:
            scale_x = float(model_size) / float(tile.w)
            scale_y = float(model_size) / float(tile.h)
            for st in SYNTHETIC_TARGETS:
                if (
                    st["gx"] + st["gw"] > tile.x
                    and st["gx"] < tile.x + tile.w
                    and st["gy"] + st["gh"] > tile.y
                    and st["gy"] < tile.y + tile.h
                ):
                    tile_dets.append(
                        Detection(
                            x_local=(st["gx"] - float(tile.x)) * scale_x,
                            y_local=(st["gy"] - float(tile.y)) * scale_y,
                            w=st["gw"] * scale_x,
                            h=st["gh"] * scale_y,
                            conf=st["conf"],
                            class_id=st["class_id"],
                        )
                    )
            time.sleep(0.002)
        else:
            tile_dets = detector.predict_tile(
                tile_slice,
                tile_size=model_size,
                conf_threshold=conf_threshold,
            )

        del tile_slice

        for d in tile_dets:
            global_det = remap_detection_to_global(d, tile, model_size, idx)
            all_global_detections.append(global_det)

        if progress_callback:
            status_msg = f"Обробка тайла {idx + 1}/{total_tiles} (знайдено: {len(all_global_detections)})..."
            progress_callback(idx + 1, total_tiles, status_msg)

    merged_detections = merge_boundary_detections(
        all_global_detections, tiles, diou_threshold=diou_threshold, conf_threshold=conf_threshold
    )
    cnames = class_names
    if cnames is None and detector is not None and hasattr(detector, "get_class_names"):
        try:
            cnames = detector.get_class_names()
        except Exception:
            pass
    return format_detections_for_gui(merged_detections, class_names=cnames)


# Re-export BatchTriageWorker for convenience
try:
    from gui.batch_triage_worker import BatchTriageWorker
except ImportError:
    pass
