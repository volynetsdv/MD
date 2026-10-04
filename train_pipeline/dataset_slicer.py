"""
Dataset Slicer for High-Resolution Aerial Imagery (YOLO / DOTA / VisDrone format).

Integrates with the compiled C++/pybind11 libtiling_core (pytiling_core) to apply
hardware-aware dynamic tiling, zero-loss boundary box remapping, fragment filtering,
and parallel multi-core slicing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

# =============================================================================
# Safe Import of pytiling_core
# =============================================================================
try:
    import pytiling_core
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    candidate_paths = [
        repo_root / "build",
        repo_root / "build" / "Release",
        repo_root / "build" / "bindings",
        repo_root / "bindings",
        Path.cwd() / "build",
        Path.cwd() / "build" / "Release",
    ]
    imported = False
    for p in candidate_paths:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            try:
                import pytiling_core
                imported = True
                break
            except ModuleNotFoundError:
                continue

    if not imported:
        raise ImportError(
            "Could not load 'pytiling_core'. Please make sure the C++/pybind11 module "
            "is built via CMake (e.g. 'cmake -B build && cmake --build build')."
        )

logger = logging.getLogger("AerialDatasetSlicer")


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class YOLOBox:
    """
    Representation of a single YOLO bounding box.

    Coordinates are normalized to [0.0, 1.0] relative to image or tile dimensions.
    """
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_xyxy_abs(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert normalized YOLO center-size box to absolute pixel (x1, y1, x2, y2)."""
        w_px = self.width * img_w
        h_px = self.height * img_h
        x_c_px = self.x_center * img_w
        y_c_px = self.y_center * img_h

        x1 = x_c_px - w_px / 2.0
        y1 = y_c_px - h_px / 2.0
        x2 = x_c_px + w_px / 2.0
        y2 = y_c_px + h_px / 2.0
        return x1, y1, x2, y2

    def to_yolo_line(self, precision: int = 6) -> str:
        """Format as a single YOLO annotation line."""
        return (
            f"{self.class_id} "
            f"{self.x_center:.{precision}f} "
            f"{self.y_center:.{precision}f} "
            f"{self.width:.{precision}f} "
            f"{self.height:.{precision}f}"
        )

    @classmethod
    def from_yolo_line(cls, line: str) -> "YOLOBox":
        """Parse a single line from a YOLO .txt annotation file."""
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO annotation line: '{line}'")
        class_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        return cls(class_id=class_id, x_center=xc, y_center=yc, width=w, height=h)


# =============================================================================
# Categories and Annotation Adapters
# =============================================================================
DOTA_V15_CLASSES: Dict[str, int] = {
    "plane": 0,
    "ship": 1,
    "storage-tank": 2,
    "baseball-diamond": 3,
    "tennis-court": 4,
    "basketball-court": 5,
    "ground-track-field": 6,
    "harbor": 7,
    "bridge": 8,
    "large-vehicle": 9,
    "small-vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer-ball-field": 13,
    "swimming-pool": 14,
    "container-crane": 15,
}

VISDRONE_CLASSES: Dict[str, int] = {
    "pedestrian": 0,
    "people": 1,
    "bicycle": 2,
    "car": 3,
    "van": 4,
    "truck": 5,
    "tricycle": 6,
    "awning-tricycle": 7,
    "bus": 8,
    "motor": 9,
}


class BaseAnnotationAdapter(ABC):
    """Abstract base class for dataset annotation adapters."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the unique format identifier."""
        pass

    @abstractmethod
    def get_class_names(self) -> Dict[int, str]:
        """Return dictionary mapping class IDs to class names."""
        pass

    @abstractmethod
    def parse_line(
        self,
        line: str,
        img_w: int = 0,
        img_h: int = 0,
        **kwargs: Any,
    ) -> Optional[YOLOBox]:
        """Parse a single line from an annotation file into a normalized YOLOBox."""
        pass

    def parse_file(
        self,
        file_path: Union[str, Path],
        img_w: int = 0,
        img_h: int = 0,
        **kwargs: Any,
    ) -> List[YOLOBox]:
        """Parse an entire annotation file into a list of normalized YOLOBox instances."""
        w = kwargs.get("img_width", img_w)
        h = kwargs.get("img_height", img_h)

        boxes: List[YOLOBox] = []
        path = Path(file_path)
        if not path.exists():
            return boxes

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                box = self.parse_line(line, img_w=w, img_h=h, **kwargs)
                if box is not None:
                    boxes.append(box)
        return boxes


class YoloHBBAdapter(BaseAnnotationAdapter):
    """
    Adapter for standard YOLO Horizontal Bounding Box (HBB) annotations.

    Expected format: class_id x_center y_center width height (normalized in [0.0, 1.0]).
    """

    def __init__(self, class_map: Optional[Dict[str, int]] = None):
        self.class_map = class_map

    @property
    def format_name(self) -> str:
        return "yolo"

    def get_class_names(self) -> Dict[int, str]:
        if self.class_map:
            return {int(v): str(k) for k, v in self.class_map.items()}
        return {int(v): str(k) for k, v in VISDRONE_CLASSES.items()}

    def parse_line(
        self,
        line: str,
        img_w: int = 0,
        img_h: int = 0,
        **kwargs: Any,
    ) -> Optional[YOLOBox]:
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            return None
        try:
            return YOLOBox.from_yolo_line(line_str)
        except Exception as e:
            logger.debug("YOLO parse error on line '%s': %s", line_str, e)
            return None


class YoloOBBAdapter(BaseAnnotationAdapter):
    """
    Adapter for YOLO-OBB format annotations (9 numerical tokens).

    Expected format:
    class_id x1 y1 x2 y2 x3 y3 x4 y4
    where coordinates are already normalized floats in [0.0, 1.0].
    Converts oriented bounding boxes into minimal axis-aligned bounding boxes (AABB).
    """

    def __init__(self, class_map: Optional[Dict[str, int]] = None):
        self.class_map = class_map

    @property
    def format_name(self) -> str:
        return "yolo_obb"

    def get_class_names(self) -> Dict[int, str]:
        if self.class_map:
            return {int(v): str(k) for k, v in self.class_map.items()}
        return {int(v): str(k) for k, v in DOTA_V15_CLASSES.items()}

    def parse_line(
        self,
        line: str,
        img_w: int = 0,
        img_h: int = 0,
        **kwargs: Any,
    ) -> Optional[YOLOBox]:
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            return None

        parts = line_str.split()
        if len(parts) != 9:
            return None

        try:
            class_id = int(float(parts[0]))
            coords = [float(p) for p in parts[1:9]]
        except ValueError:
            return None

        xs = coords[0::2]  # x1, x2, x3, x4
        ys = coords[1::2]  # y1, y2, y3, y4

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        # Clamping to normalized space [0.0, 1.0]
        x_min = max(0.0, min(1.0, x_min))
        x_max = max(0.0, min(1.0, x_max))
        y_min = max(0.0, min(1.0, y_min))
        y_max = max(0.0, min(1.0, y_max))

        w = x_max - x_min
        h = y_max - y_min
        if w <= 1e-6 or h <= 1e-6:
            return None

        xc = (x_min + x_max) / 2.0
        yc = (y_min + y_max) / 2.0

        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        return YOLOBox(class_id=class_id, x_center=xc, y_center=yc, width=w, height=h)


class DotaOBBAdapter(BaseAnnotationAdapter):
    """
    Adapter for DOTA v1.5 Oriented Bounding Box (OBB) annotations.

    Parses 8-point polygon coordinates: x1 y1 x2 y2 x3 y3 x4 y4 class_name [difficult]
    Ignores header metadata (imagesource:, gsd:, comments).
    Converts oriented bounding boxes into minimal axis-aligned bounding boxes (AABB)
    and normalizes coordinates to [0.0, 1.0] relative to (img_w, img_h).
    """

    def __init__(
        self,
        class_map: Optional[Dict[str, int]] = None,
        ignore_difficult: bool = False,
    ):
        self._custom_class_map = class_map
        if class_map is not None:
            self.class_map = {
                self._normalize_class_name(k): v for k, v in class_map.items()
            }
        else:
            self.class_map = self._build_default_class_map()
        self.ignore_difficult = ignore_difficult

    def get_class_names(self) -> Dict[int, str]:
        if self._custom_class_map is not None:
            return {int(v): str(k) for k, v in self._custom_class_map.items()}
        return {int(v): str(k) for k, v in DOTA_V15_CLASSES.items()}

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        """Normalize class names: lower-case, strip, replace spaces/underscores with hyphens."""
        return name.strip().lower().replace("_", "-").replace(" ", "-")

    @classmethod
    def _build_default_class_map(cls) -> Dict[str, int]:
        mapping = dict(DOTA_V15_CLASSES)
        aliases = {
            "storage tank": 2,
            "storagetank": 2,
            "baseball diamond": 3,
            "tennis court": 4,
            "basketball court": 5,
            "ground track field": 6,
            "harbour": 7,
            "large vehicle": 9,
            "small vehicle": 10,
            "soccer ball field": 13,
            "swimming pool": 14,
            "container crane": 15,
        }
        for alias, cid in aliases.items():
            mapping[cls._normalize_class_name(alias)] = cid
        return mapping

    @property
    def format_name(self) -> str:
        return "dota"

    def parse_line(
        self,
        line: str,
        img_w: int = 0,
        img_h: int = 0,
        **kwargs: Any,
    ) -> Optional[YOLOBox]:
        w_img = kwargs.get("img_width", img_w)
        h_img = kwargs.get("img_height", img_h)

        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            return None

        lower_str = line_str.lower()
        if lower_str.startswith("imagesource:") or lower_str.startswith("gsd:"):
            return None

        parts = line_str.split()
        if len(parts) < 9:
            return None

        try:
            coords = [float(p) for p in parts[:8]]
        except ValueError:
            return None

        trailing = parts[8:]
        difficult = 0
        if len(trailing) >= 2:
            try:
                difficult = int(trailing[-1])
                raw_class_name = " ".join(trailing[:-1])
            except ValueError:
                raw_class_name = " ".join(trailing)
        else:
            raw_class_name = trailing[0]

        if self.ignore_difficult and difficult == 1:
            return None

        norm_class = self._normalize_class_name(raw_class_name)
        if norm_class not in self.class_map:
            logger.debug("DOTA class '%s' not recognized in class_map, skipping.", norm_class)
            return None
        class_id = self.class_map[norm_class]

        if w_img <= 0 or h_img <= 0:
            return None

        xs = coords[0::2]
        ys = coords[1::2]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        # Clamping to image bounds
        x_min = max(0.0, min(float(w_img), x_min))
        x_max = max(0.0, min(float(w_img), x_max))
        y_min = max(0.0, min(float(h_img), y_min))
        y_max = max(0.0, min(float(h_img), y_max))

        w_px = x_max - x_min
        h_px = y_max - y_min
        if w_px <= 1e-6 or h_px <= 1e-6:
            return None

        # Normalized coordinates relative to full image
        xc = (x_min + x_max) / (2.0 * float(w_img))
        yc = (y_min + y_max) / (2.0 * float(h_img))
        w = w_px / float(w_img)
        h = h_px / float(h_img)

        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        return YOLOBox(class_id=class_id, x_center=xc, y_center=yc, width=w, height=h)


def detect_format(label_file_path: Union[str, Path]) -> str:
    """
    Automatically detect annotation format from a label file.

    Returns
    -------
    str
        'yolo' for normalized YOLO HBB (5 numerical columns)
        'yolo_obb' for normalized YOLO OBB (9 numerical columns: class_id x1 y1 x2 y2 x3 y3 x4 y4)
        'dota' for DOTA OBB (8 numerical coordinates + class name string [+ difficult])
    """
    path = Path(label_file_path)
    if not path.exists():
        return "yolo"

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                lower = line.lower()
                if lower.startswith("imagesource:") or lower.startswith("gsd:"):
                    continue

                parts = line.split()
                # 1. Check for standard YOLO HBB: exactly 5 tokens
                if len(parts) == 5:
                    try:
                        _ = int(float(parts[0]))
                        _ = [float(p) for p in parts[1:5]]
                        return "yolo"
                    except ValueError:
                        pass

                # 2. Check for YOLO-OBB: exactly 9 numerical tokens (first is int, rest are floats <= 1.05)
                if len(parts) == 9:
                    try:
                        _ = int(float(parts[0]))
                        coords = [float(p) for p in parts[1:9]]
                        if all(c <= 1.05 for c in coords):
                            return "yolo_obb"
                    except ValueError:
                        pass

                # 3. Check for DOTA: at least 9 tokens where first 8 are coordinates and trailing is class name
                if len(parts) >= 9:
                    try:
                        _ = [float(p) for p in parts[:8]]
                        return "dota"
                    except ValueError:
                        pass
    except Exception as err:
        logger.warning("Failed to detect format from '%s': %s", path, err)

    return "yolo"


def get_adapter(
    format_name: str = "auto",
    class_map: Optional[Dict[str, int]] = None,
    ignore_difficult: bool = False,
) -> BaseAnnotationAdapter:
    """
    Factory to instantiate the appropriate annotation adapter.
    """
    fmt = (format_name or "auto").strip().lower()
    if fmt in ("yolo", "yolo_hbb", "hbb", "visdrone"):
        return YoloHBBAdapter(class_map=class_map)
    elif fmt in ("yolo_obb", "yolo-obb", "obb_yolo", "yolobb"):
        return YoloOBBAdapter(class_map=class_map)
    elif fmt in ("dota", "dota_obb", "obb", "dota1.5", "dota_v15"):
        return DotaOBBAdapter(class_map=class_map, ignore_difficult=ignore_difficult)
    else:
        raise ValueError(
            f"Unsupported annotation format: '{format_name}'. Supported formats: 'yolo', 'yolo_obb', 'dota', 'auto'."
        )


# =============================================================================
# Mathematical Bounding Box Clipping and Normalization
# =============================================================================
def clip_and_normalize_bbox(
    box: YOLOBox,
    img_w: int,
    img_h: int,
    tile_rect: Any,
    min_area_ratio: float = 0.25,
) -> Optional[YOLOBox]:
    """
    Compute intersection of an absolute bounding box with a tile, filter fragments,
    and normalize coordinates to tile space [0.0, 1.0].

    Parameters
    ----------
    box : YOLOBox
        Original box in normalized coordinates [0.0, 1.0] relative to the source image.
    img_w : int
        Full source image width (pixels).
    img_h : int
        Full source image height (pixels).
    tile_rect : Any
        Tile rectangle with attributes x, y, w, h (e.g. pytiling_core.Rect).
    min_area_ratio : float
        Minimum retained area ratio threshold (default: 0.25 = 25%).
        If the clipped area in the tile is less than this fraction of the original box
        area, the fragment is discarded.

    Returns
    -------
    Optional[YOLOBox]
        Remapped and clamped YOLOBox in tile-relative coordinates [0.0, 1.0],
        or None if the box does not intersect the tile or was filtered out.
    """
    x1, y1, x2, y2 = box.to_xyxy_abs(img_w, img_h)

    # Validate input box dimensions
    orig_w = max(0.0, x2 - x1)
    orig_h = max(0.0, y2 - y1)
    orig_area = orig_w * orig_h
    if orig_area <= 1e-9:
        return None

    tile_x1 = float(tile_rect.x)
    tile_y1 = float(tile_rect.y)
    tile_x2 = float(tile_rect.x + tile_rect.w)
    tile_y2 = float(tile_rect.y + tile_rect.h)

    # Compute intersection with tile boundaries
    clip_x1 = max(x1, tile_x1)
    clip_y1 = max(y1, tile_y1)
    clip_x2 = min(x2, tile_x2)
    clip_y2 = min(y2, tile_y2)

    clip_w = max(0.0, clip_x2 - clip_x1)
    clip_h = max(0.0, clip_y2 - clip_y1)
    clip_area = clip_w * clip_h

    if clip_w <= 1e-6 or clip_h <= 1e-6 or clip_area <= 1e-9:
        return None

    # Filter out shards/fragments smaller than min_area_ratio threshold
    area_ratio = clip_area / orig_area
    if area_ratio < min_area_ratio:
        return None

    # Translate coordinates to tile-local space
    local_x1 = clip_x1 - tile_x1
    local_y1 = clip_y1 - tile_y1
    local_x2 = clip_x2 - tile_x1
    local_y2 = clip_y2 - tile_y1

    # Strict boundary clamping to [0, tile.w] and [0, tile.h]
    local_x1 = max(0.0, min(float(tile_rect.w), local_x1))
    local_y1 = max(0.0, min(float(tile_rect.h), local_y1))
    local_x2 = max(0.0, min(float(tile_rect.w), local_x2))
    local_y2 = max(0.0, min(float(tile_rect.h), local_y2))

    if local_x2 <= local_x1 or local_y2 <= local_y1:
        return None

    # Normalize to [0.0, 1.0] relative to the tile
    norm_x1 = local_x1 / float(tile_rect.w)
    norm_y1 = local_y1 / float(tile_rect.h)
    norm_x2 = local_x2 / float(tile_rect.w)
    norm_y2 = local_y2 / float(tile_rect.h)

    # Convert to center-width-height format
    norm_w = norm_x2 - norm_x1
    norm_h = norm_y2 - norm_y1
    norm_xc = norm_x1 + norm_w / 2.0
    norm_yc = norm_y1 + norm_h / 2.0

    # Invariant enforcement: strictly inside [0.0, 1.0]
    norm_xc = max(0.0, min(1.0, norm_xc))
    norm_yc = max(0.0, min(1.0, norm_yc))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    if norm_w <= 1e-6 or norm_h <= 1e-6:
        return None

    return YOLOBox(
        class_id=box.class_id,
        x_center=norm_xc,
        y_center=norm_yc,
        width=norm_w,
        height=norm_h,
    )


# =============================================================================
# Standalone Worker for Parallel Processing (Picklable)
# =============================================================================
def _process_image_task(task_params: Dict[str, Any]) -> Dict[str, Any]:
    """Worker task executed by ProcessPoolExecutor to slice a single image."""
    img_path = Path(task_params["img_path"])
    ann_path = Path(task_params["ann_path"]) if task_params.get("ann_path") else None
    out_images_dir = Path(task_params["out_images_dir"])
    out_labels_dir = Path(task_params["out_labels_dir"])
    altitude = float(task_params["altitude"])
    vram_mb = int(task_params["vram_mb"])
    min_area_ratio = float(task_params["min_area_ratio"])
    target_size_override = task_params.get("target_size_override")
    resize_to_target = bool(task_params.get("resize_to_target", True))
    save_empty_tiles = bool(task_params.get("save_empty_tiles", False))
    image_quality = int(task_params.get("image_quality", 95))
    format_name = task_params.get("format_name", "auto")
    class_map = task_params.get("class_map")
    ignore_difficult = bool(task_params.get("ignore_difficult", False))

    result = {
        "image_name": img_path.name,
        "success": False,
        "tiles_saved": 0,
        "annotations_saved": 0,
        "error": None,
    }

    try:
        img = cv2.imread(str(img_path))
        if img is None:
            result["error"] = f"Failed to read image at {img_path}"
            return result

        img_h, img_w = img.shape[:2]

        # Read and adapt annotations if available
        boxes: List[YOLOBox] = []
        if ann_path and ann_path.exists():
            if format_name == "auto":
                resolved_fmt = detect_format(ann_path)
            else:
                resolved_fmt = format_name

            adapter = get_adapter(
                format_name=resolved_fmt,
                class_map=class_map,
                ignore_difficult=ignore_difficult,
            )
            boxes = adapter.parse_file(ann_path, img_w=img_w, img_h=img_h)

        # Compute dynamic tiling parameters via libtiling_core
        tiling_config = pytiling_core.calculate_tiling_params(
            width=img_w,
            height=img_h,
            altitude=altitude,
            vram_mb=vram_mb,
        )

        effective_target_size = (
            target_size_override if target_size_override else tiling_config.tile_size
        )

        tiles_saved = 0
        annotations_saved = 0

        for tile_idx, tile in enumerate(tiling_config.tiles):
            # Clip and remap all bounding boxes intersecting this tile
            tile_boxes: List[YOLOBox] = []
            for b in boxes:
                clipped = clip_and_normalize_bbox(
                    box=b,
                    img_w=img_w,
                    img_h=img_h,
                    tile_rect=tile,
                    min_area_ratio=min_area_ratio,
                )
                if clipped is not None:
                    tile_boxes.append(clipped)

            # Check if tile should be skipped when empty
            if not tile_boxes and not save_empty_tiles:
                continue

            # Extract image crop from VRAM/RAM
            crop = img[tile.y : tile.y + tile.h, tile.x : tile.x + tile.w]

            # Resize to target square dimension if requested
            if resize_to_target and (
                crop.shape[1] != effective_target_size or crop.shape[0] != effective_target_size
            ):
                crop = cv2.resize(
                    crop,
                    (effective_target_size, effective_target_size),
                    interpolation=cv2.INTER_LINEAR,
                )

            stem = img_path.stem
            tile_filename_base = f"{stem}_tile_{tile_idx:04d}_{tile.x}_{tile.y}"
            out_img_file = out_images_dir / f"{tile_filename_base}.jpg"
            out_lbl_file = out_labels_dir / f"{tile_filename_base}.txt"

            # Save tile image
            cv2.imwrite(
                str(out_img_file),
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), image_quality],
            )

            # Save corresponding YOLO annotations
            with open(out_lbl_file, "w", encoding="utf-8") as f:
                for tb in tile_boxes:
                    f.write(tb.to_yolo_line() + "\n")

            tiles_saved += 1
            annotations_saved += len(tile_boxes)

        result["success"] = True
        result["tiles_saved"] = tiles_saved
        result["annotations_saved"] = annotations_saved
        result["tile_size"] = effective_target_size

    except Exception as exc:
        result["error"] = str(exc)

    return result


# =============================================================================
# AerialDatasetSlicer Class
# =============================================================================
class AerialDatasetSlicer:
    """
    Production-grade Dataset Slicer for high-resolution aerial imagery (VisDrone / DOTA).

    Features:
    - Dynamic grid calculation using pytiling_core for resolutions [320, 416, 512, 640].
    - Zero-loss and fragment-filtered bounding box transformation with strict boundary clamping.
    - Standard Ultralytics YOLO output layout: images/split, labels/split.
    - Multi-format annotation adapters: YOLO HBB, DOTA v1.5 OBB -> HBB.
    - Multi-process parallel slicing using ProcessPoolExecutor for massive 8K datasets.
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        annotation_dir: Union[str, Path],
        output_dir: Union[str, Path],
        altitude: float = 100.0,
        vram_mb: int = 2048,
        min_area_ratio: float = 0.25,
        target_size: Optional[int] = None,
        split: str = "train",
        save_empty_tiles: bool = False,
        resize_to_target: bool = True,
        num_workers: Optional[int] = None,
        image_extensions: Tuple[str, ...] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        ),
        format: str = "auto",
        class_map: Optional[Dict[str, int]] = None,
        ignore_difficult: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the AerialDatasetSlicer.

        Parameters
        ----------
        image_dir : Union[str, Path]
            Path to folder containing source aerial images.
        annotation_dir : Union[str, Path]
            Path to folder containing corresponding annotations (.txt).
        output_dir : Union[str, Path]
            Destination directory where Ultralytics structure (images/split, labels/split)
            will be created.
        altitude : float
            UAV altitude telemetry in meters (influences tile size and overlap).
        vram_mb : int
            Target GPU VRAM in MB used by libtiling_core for grid computation.
        min_area_ratio : float
            Threshold ratio (default: 0.25) to discard object fragments cut by tile seams.
        target_size : Optional[int]
            Optional target resolution override (320, 416, 512, or 640). If None,
            automatically picked by calculate_tiling_params.
        split : str
            Dataset partition name, e.g. 'train', 'val', or 'test'.
        save_empty_tiles : bool
            If True, saves tiles even if they contain zero object annotations.
        resize_to_target : bool
            If True, resizes edge tiles to the target square resolution (e.g. 640x640).
        num_workers : Optional[int]
            Number of worker processes for ProcessPoolExecutor. If None, defaults
            to os.cpu_count().
        image_extensions : Tuple[str, ...]
            Supported image file extensions (case-insensitive).
        format : str
            Annotation format: 'auto' (detect), 'yolo' (YOLO HBB), or 'dota' (DOTA v1.5 OBB).
        class_map : Optional[Dict[str, int]]
            Optional mapping of category names to integer class IDs for DOTA.
        ignore_difficult : bool
            If True, skips annotations marked with difficult=1 in DOTA.
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.altitude = float(altitude)
        self.vram_mb = int(vram_mb)
        self.min_area_ratio = float(min_area_ratio)
        self.target_size = target_size
        self.split = split
        self.save_empty_tiles = save_empty_tiles
        self.resize_to_target = resize_to_target
        self.num_workers = num_workers or os.cpu_count() or 1
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)
        self.format = kwargs.get("format_name", format)
        self.class_map = class_map
        self.ignore_difficult = ignore_difficult

        # Build directory paths according to Ultralytics YOLO format
        self.out_images_dir = self.output_dir / "images" / self.split
        self.out_labels_dir = self.output_dir / "labels" / self.split
        self.out_images_dir.mkdir(parents=True, exist_ok=True)
        self.out_labels_dir.mkdir(parents=True, exist_ok=True)

    def find_image_files(self) -> List[Path]:
        """Find all matching image files in image_dir (case-insensitive matching)."""
        if not self.image_dir.exists():
            return []
        files = [
            p
            for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.image_extensions
        ]
        files.sort()
        return files

    def slice_single_image(
        self, image_path: Union[str, Path], annotation_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Process and slice a single image synchronously.

        Parameters
        ----------
        image_path : Union[str, Path]
            Path to the input image file.
        annotation_path : Optional[Union[str, Path]]
            Path to the matching annotation file (.txt). If None, searches in self.annotation_dir.

        Returns
        -------
        Dict[str, Any]
            Execution status and statistics dictionary.
        """
        img_path = Path(image_path)
        if annotation_path is None:
            ann_path = self.annotation_dir / f"{img_path.stem}.txt"
        else:
            ann_path = Path(annotation_path)

        resolved_fmt = self.format
        if resolved_fmt == "auto" and ann_path.exists():
            resolved_fmt = detect_format(ann_path)

        task_params = {
            "img_path": str(img_path),
            "ann_path": str(ann_path) if ann_path.exists() else None,
            "out_images_dir": str(self.out_images_dir),
            "out_labels_dir": str(self.out_labels_dir),
            "altitude": self.altitude,
            "vram_mb": self.vram_mb,
            "min_area_ratio": self.min_area_ratio,
            "target_size_override": self.target_size,
            "resize_to_target": self.resize_to_target,
            "save_empty_tiles": self.save_empty_tiles,
            "image_quality": 95,
            "format_name": resolved_fmt,
            "class_map": self.class_map,
            "ignore_difficult": self.ignore_difficult,
        }

        res = _process_image_task(task_params)
        if res.get("success"):
            effective_tile_size = res.get("tile_size", self.target_size or 512)
            yaml_path, meta_path = self._generate_dataset_configs(
                resolved_fmt=resolved_fmt,
                effective_tile_size=effective_tile_size,
            )
            res["dataset_yaml"] = str(yaml_path)
            res["slicing_meta"] = str(meta_path)
            res["tile_size"] = effective_tile_size
        return res

    def _generate_dataset_configs(
        self,
        resolved_fmt: str,
        effective_tile_size: int,
    ) -> Tuple[Path, Path]:
        """
        Generate Ultralytics dataset.yaml and slicing_meta.json in output_dir.
        """
        adapter = get_adapter(
            format_name=resolved_fmt,
            class_map=self.class_map,
            ignore_difficult=self.ignore_difficult,
        )
        class_names_dict = adapter.get_class_names()
        sorted_names = {
            int(k): str(v)
            for k, v in sorted(class_names_dict.items(), key=lambda item: item[0])
        }

        # Ultralytics path resolution: relative to dataset path
        train_path = (
            "images/train"
            if (self.output_dir / "images" / "train").exists()
            else f"images/{self.split}"
        )
        val_path = (
            "images/val"
            if (self.output_dir / "images" / "val").exists()
            else train_path
        )

        yaml_content = {
            "path": str(self.output_dir.resolve()),
            "train": train_path,
            "val": val_path,
            "names": sorted_names,
        }

        dataset_yaml_path = self.output_dir / "dataset.yaml"
        with open(dataset_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False)

        slicing_meta = {
            "tile_size": int(effective_tile_size),
            "altitude": float(self.altitude),
            "vram_mb": int(self.vram_mb),
            "classes_count": len(sorted_names),
            "format": resolved_fmt,
            "split": self.split,
        }

        slicing_meta_path = self.output_dir / "slicing_meta.json"
        with open(slicing_meta_path, "w", encoding="utf-8") as f:
            json.dump(slicing_meta, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Generated dataset configurations: '{dataset_yaml_path}' and '{slicing_meta_path}' "
            f"(tile_size={effective_tile_size}, classes={len(sorted_names)})"
        )
        return dataset_yaml_path, slicing_meta_path

    def process_dataset(self) -> Dict[str, Any]:
        """
        Slice all images in image_dir using multi-process parallel execution.

        Returns
        -------
        Dict[str, Any]
            Overall dataset slicing summary report.
        """
        image_files = self.find_image_files()
        total_images = len(image_files)

        if total_images == 0:
            raise FileNotFoundError(
                f"No image files found in '{self.image_dir}' matching extensions: {self.image_extensions}"
            )

        # Pair images with annotations
        matching_tasks: List[Dict[str, Any]] = []
        found_labels_count = 0
        first_ann_file: Optional[Path] = None

        for img_path in image_files:
            ann_path = self.annotation_dir / f"{img_path.stem}.txt"
            if ann_path.exists():
                found_labels_count += 1
                if first_ann_file is None:
                    first_ann_file = ann_path
                actual_ann_str: Optional[str] = str(ann_path)
            else:
                actual_ann_str = None

            matching_tasks.append({
                "img_path": str(img_path),
                "ann_path": actual_ann_str,
            })

        if found_labels_count == 0:
            raise FileNotFoundError(
                f"No matching .txt annotation files found in '{self.annotation_dir}' "
                f"for {total_images} images located in '{self.image_dir}'."
            )

        # Resolve annotation format
        if self.format.lower() == "auto":
            resolved_fmt = detect_format(first_ann_file) if first_ann_file else "yolo"
        else:
            resolved_fmt = self.format.lower()

        if resolved_fmt == "dota":
            fmt_display = "DOTA v1.5 (OBB -> HBB conversion)"
        elif resolved_fmt in ("yolo_obb", "yolo-obb"):
            fmt_display = "YOLO-OBB (Normalized OBB -> HBB conversion)"
        elif resolved_fmt == "yolo":
            fmt_display = "YOLO HBB"
        else:
            fmt_display = resolved_fmt.upper()

        telemetry_lines = [
            f"[INFO] Знайдено зображень: {total_images}",
            f"[INFO] Знайдено відповідних файлів анотацій: {found_labels_count}",
            f"[INFO] Визначено формат розмітки: {fmt_display}",
        ]
        for t_line in telemetry_lines:
            print(t_line)
            logger.info(t_line)

        tasks: List[Dict[str, Any]] = []
        for mt in matching_tasks:
            tasks.append(
                {
                    "img_path": mt["img_path"],
                    "ann_path": mt["ann_path"],
                    "out_images_dir": str(self.out_images_dir),
                    "out_labels_dir": str(self.out_labels_dir),
                    "altitude": self.altitude,
                    "vram_mb": self.vram_mb,
                    "min_area_ratio": self.min_area_ratio,
                    "target_size_override": self.target_size,
                    "resize_to_target": self.resize_to_target,
                    "save_empty_tiles": self.save_empty_tiles,
                    "image_quality": 95,
                    "format_name": resolved_fmt,
                    "class_map": self.class_map,
                    "ignore_difficult": self.ignore_difficult,
                }
            )

        total_tiles = 0
        total_annotations = 0
        failed_count = 0
        errors: List[str] = []
        effective_tile_size = self.target_size

        if self.num_workers <= 1 or len(tasks) <= 1:
            for t in tasks:
                res = _process_image_task(t)
                if res["success"]:
                    total_tiles += res["tiles_saved"]
                    total_annotations += res["annotations_saved"]
                    if effective_tile_size is None and "tile_size" in res:
                        effective_tile_size = res["tile_size"]
                else:
                    failed_count += 1
                    errors.append(f"{res['image_name']}: {res['error']}")
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(_process_image_task, t) for t in tasks]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        res = fut.result()
                        if res["success"]:
                            total_tiles += res["tiles_saved"]
                            total_annotations += res["annotations_saved"]
                            if effective_tile_size is None and "tile_size" in res:
                                effective_tile_size = res["tile_size"]
                        else:
                            failed_count += 1
                            errors.append(f"{res['image_name']}: {res['error']}")
                    except Exception as exc:
                        failed_count += 1
                        errors.append(str(exc))

        if effective_tile_size is None:
            effective_tile_size = 512

        yaml_path, meta_path = self._generate_dataset_configs(
            resolved_fmt=resolved_fmt,
            effective_tile_size=effective_tile_size,
        )

        summary = {
            "total_images": total_images,
            "processed_successfully": total_images - failed_count,
            "failed_count": failed_count,
            "total_tiles_generated": total_tiles,
            "total_annotations_generated": total_annotations,
            "output_images_dir": str(self.out_images_dir),
            "output_labels_dir": str(self.out_labels_dir),
            "detected_format": resolved_fmt,
            "dataset_yaml": str(yaml_path),
            "slicing_meta": str(meta_path),
            "tile_size": effective_tile_size,
            "errors": errors,
        }

        logger.info(
            f"Dataset slicing complete: {summary['processed_successfully']}/{total_images} images processed, "
            f"{total_tiles} tiles generated, {total_annotations} annotations created."
        )

        return summary


# =============================================================================
# CLI Entry Point
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset Slicer for High-Resolution Aerial Imagery (DOTA / VisDrone / YOLO)"
    )
    parser.add_argument("--image-dir", "-i", type=str, required=True, help="Path to input images directory")
    parser.add_argument("--annotation-dir", "-a", type=str, required=True, help="Path to annotations directory")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Output directory for sliced dataset")
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="auto",
        choices=["auto", "yolo", "yolo_obb", "dota"],
        help="Dataset annotation format (default: auto)",
    )
    parser.add_argument("--altitude", type=float, default=100.0, help="Flight altitude in meters (default: 100.0)")
    parser.add_argument("--vram-mb", type=int, default=2048, help="Target VRAM in MB (default: 2048)")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.25,
        help="Minimum area ratio threshold to keep fragments (default: 0.25)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        choices=[320, 416, 512, 640],
        help="Target tile square resolution override",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split name (default: train)")
    parser.add_argument(
        "--save-empty-tiles",
        action="store_true",
        help="Save tiles that contain no annotations",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Do not resize crops to target size",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: cpu count)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    slicer = AerialDatasetSlicer(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        altitude=args.altitude,
        vram_mb=args.vram_mb,
        min_area_ratio=args.min_area_ratio,
        target_size=args.target_size,
        split=args.split,
        save_empty_tiles=args.save_empty_tiles,
        resize_to_target=not args.no_resize,
        num_workers=args.workers,
        format=args.format,
    )
    summary = slicer.process_dataset()
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
