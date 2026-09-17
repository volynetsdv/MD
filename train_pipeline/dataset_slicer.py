"""
Dataset Slicer for High-Resolution Aerial Imagery (YOLO / Ultralytics format).

Integrates with the compiled C++/pybind11 libtiling_core (pytiling_core) to apply
hardware-aware dynamic tiling, zero-loss boundary box remapping, fragment filtering,
and parallel multi-core slicing.
"""

import concurrent.futures
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# =============================================================================
# Safe Import of pytiling_core
# =============================================================================
try:
    import pytiling_core
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    candidate_paths = [
        repo_root / "build",
        repo_root / "build" / "bindings",
        repo_root / "bindings",
        Path.cwd() / "build",
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

        # Read annotations if available
        boxes: List[YOLOBox] = []
        if ann_path and ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        try:
                            boxes.append(YOLOBox.from_yolo_line(line_str))
                        except Exception as e:
                            logger.warning(f"Error parsing line '{line_str}' in {ann_path}: {e}")

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
    - Standard Ultralytics YOLO output layout: images/train, labels/train.
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
    ):
        """
        Initialize the AerialDatasetSlicer.

        Parameters
        ----------
        image_dir : Union[str, Path]
            Path to folder containing source aerial images.
        annotation_dir : Union[str, Path]
            Path to folder containing corresponding YOLO format .txt annotations.
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
            Supported image file extensions.
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

        # Build directory paths according to Ultralytics YOLO format
        self.out_images_dir = self.output_dir / "images" / self.split
        self.out_labels_dir = self.output_dir / "labels" / self.split
        self.out_images_dir.mkdir(parents=True, exist_ok=True)
        self.out_labels_dir.mkdir(parents=True, exist_ok=True)

    def find_image_files(self) -> List[Path]:
        """Find all matching image files in image_dir."""
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
        }

        return _process_image_task(task_params)

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

        logger.info(
            f"Starting dataset slicing: {total_images} images found in '{self.image_dir}' "
            f"using {self.num_workers} worker processes."
        )

        tasks: List[Dict[str, Any]] = []
        for img_path in image_files:
            ann_path = self.annotation_dir / f"{img_path.stem}.txt"
            tasks.append(
                {
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
                }
            )

        total_tiles = 0
        total_annotations = 0
        failed_count = 0
        errors: List[str] = []

        if self.num_workers <= 1 or len(tasks) <= 1:
            for t in tasks:
                res = _process_image_task(t)
                if res["success"]:
                    total_tiles += res["tiles_saved"]
                    total_annotations += res["annotations_saved"]
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
                        else:
                            failed_count += 1
                            errors.append(f"{res['image_name']}: {res['error']}")
                    except Exception as exc:
                        failed_count += 1
                        errors.append(str(exc))

        summary = {
            "total_images": total_images,
            "processed_successfully": total_images - failed_count,
            "failed_count": failed_count,
            "total_tiles_generated": total_tiles,
            "total_annotations_generated": total_annotations,
            "output_images_dir": str(self.out_images_dir),
            "output_labels_dir": str(self.out_labels_dir),
            "errors": errors,
        }

        logger.info(
            f"Dataset slicing complete: {summary['processed_successfully']}/{total_images} images processed, "
            f"{total_tiles} tiles generated, {total_annotations} annotations created."
        )

        return summary

