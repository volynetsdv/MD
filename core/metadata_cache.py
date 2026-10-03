"""Metadata Cache Module for Aerial Reconnaissance Workstation.

Handles JSON serialization, persistence, atomic file writing, and loading of target
detection metadata cache neighboring aerial image files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def get_metadata_cache_path(image_path: Union[str, Path]) -> Path:
    """Return the neighboring JSON metadata file path for an image.

    Example:
        /path/to/aerial_01.jpg -> /path/to/aerial_01.json
    """
    return Path(image_path).with_suffix(".json")


def has_metadata_cache(image_path: Union[str, Path]) -> bool:
    """Check if a valid JSON metadata file exists for the given image."""
    json_path = get_metadata_cache_path(image_path)
    return json_path.is_file()


def save_metadata_cache(
    image_path: Union[str, Path],
    image_size: Tuple[int, int],
    detections: List[Dict[str, Any]],
    output_json_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Atomically persist coordinate detections and metadata to a JSON file.

    Format:
    {
      "image": "image.jpg",
      "image_size": [width, height],
      "detections_count": N,
      "detections": [
        {"class_id": 9, "class_name": "large-vehicle", "confidence": 0.84, "bbox": [x, y, w, h]}
      ]
    }

    Args:
        image_path: Path to the original or destination image file.
        image_size: (width, height) in pixels.
        detections: List of detection dictionaries.
        output_json_path: Optional explicit output path; defaults to neighboring .json.

    Returns:
        Path of the persisted JSON file.
    """
    img_path = Path(image_path)
    target_json = Path(output_json_path) if output_json_path else get_metadata_cache_path(img_path)
    target_json.parent.mkdir(parents=True, exist_ok=True)

    formatted_detections: List[Dict[str, Any]] = []
    for det in detections:
        class_id = int(det.get("class_id", 0))
        class_name = str(det.get("class_name", f"Class {class_id}"))
        confidence = float(det.get("confidence", det.get("conf", 0.0)))

        # Extract bbox [x, y, w, h]
        if "bbox" in det and isinstance(det["bbox"], (list, tuple)) and len(det["bbox"]) == 4:
            bbox = [round(float(v), 2) for v in det["bbox"]]
        elif "x" in det and "y" in det and "w" in det and "h" in det:
            bbox = [
                round(float(det["x"]), 2),
                round(float(det["y"]), 2),
                round(float(det["w"]), 2),
                round(float(det["h"]), 2),
            ]
        elif "xmin" in det and "ymin" in det and "xmax" in det and "ymax" in det:
            x = float(det["xmin"])
            y = float(det["ymin"])
            bbox = [
                round(x, 2),
                round(y, 2),
                round(float(det["xmax"]) - x, 2),
                round(float(det["ymax"]) - y, 2),
            ]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]

        formatted_detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": bbox,
            }
        )

    data = {
        "image": img_path.name,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "detections_count": len(formatted_detections),
        "detections": formatted_detections,
    }

    # Atomic write to temporary file first
    tmp_path = target_json.with_suffix(f".tmp_{target_json.name}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(target_json)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise exc

    return target_json


def load_metadata_cache(json_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Safely load and parse detection metadata from a JSON cache file.

    Returns:
        Dict representing parsed metadata, or None if file does not exist or is invalid.
    """
    path = Path(json_path)
    if not path.is_file():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "detections" in data:
            return data
        logger.warning("Metadata file %s is missing 'detections' key.", path)
        return None
    except Exception as exc:
        logger.warning("Failed to load metadata cache from %s: %s", path, exc)
        return None


def metadata_to_gui_detections(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert JSON metadata structure into list of GUI detection dictionaries.

    Ensures compatibility with CanvasViewer and Target Table widgets by providing
    'id', 'x', 'y', 'w', 'h', 'conf', 'confidence', 'class_id', 'class_name', 'bbox'.
    """
    gui_detections: List[Dict[str, Any]] = []
    raw_detections = metadata.get("detections", [])

    for i, det in enumerate(raw_detections):
        class_id = int(det.get("class_id", 0))
        class_name = str(det.get("class_name", f"Class {class_id}"))
        confidence = float(det.get("confidence", det.get("conf", 0.0)))
        bbox = det.get("bbox", [0.0, 0.0, 0.0, 0.0])

        x = float(bbox[0]) if len(bbox) > 0 else 0.0
        y = float(bbox[1]) if len(bbox) > 1 else 0.0
        w = float(bbox[2]) if len(bbox) > 2 else 0.0
        h = float(bbox[3]) if len(bbox) > 3 else 0.0

        gui_detections.append(
            {
                "id": i + 1,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": confidence,
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
                "bbox": [x, y, w, h],
            }
        )

    return gui_detections
