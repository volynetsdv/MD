"""Background Batch Triage Worker for Aerial Reconnaissance Workstation.

Iterates over an imagery directory, runs detection on each frame, filters targets,
copies clean positive images to a designated output folder (Metadata-Driven, zero raster modification),
and persists coordinate vectors as JSON metadata caches.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from core.metadata_cache import (
    has_metadata_cache,
    load_metadata_cache,
    metadata_to_gui_detections,
    save_metadata_cache,
)
from gui.async_worker import (
    InferenceWorker,
    run_tiled_inference,
)
from src.detector_dispatcher import DEFAULT_CLASS_NAMES, UnifiedDetector

logger = logging.getLogger("BatchTriageWorker")


class BatchTriageWorker(QThread):
    """Background QThread for automated batch triage filtering of aerial imagery."""

    progress_changed = Signal(int, int, str)  # current_file_idx, total_files, status_message
    file_processed = Signal(str, int)  # file_path, detections_count
    triage_completed = Signal(int, int, float)  # total_files, positive_files_count, elapsed_seconds
    error_occurred = Signal(str)  # error_message

    def __init__(
        self,
        input_folder: Optional[Union[str, Path]] = None,
        output_folder: Optional[Union[str, Path]] = None,
        altitude: float = 150.0,
        vram_mb: int = 2048,
        conf_threshold: float = 0.25,
        diou_threshold: float = 0.45,
        detector: Optional[Any] = None,
        use_cache: bool = True,
        parent: Optional[QObject] = None,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        conf_thresh: Optional[float] = None,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        super().__init__(parent)
        in_dir = input_dir if input_dir is not None else input_folder
        if in_dir is None:
            raise ValueError("Input folder must be specified")
        self.input_folder = Path(in_dir)
        input_p = self.input_folder.resolve()

        out_dir = output_dir if output_dir is not None else output_folder
        out_p = Path(out_dir).resolve() if out_dir else None
        if out_p is None or out_p == input_p:
            self.output_folder = input_p.parent / f"{input_p.name}_detected"
        else:
            self.output_folder = Path(out_dir)

        self.altitude = float(altitude)
        self.vram_mb = int(vram_mb)
        c_th = conf_thresh if conf_thresh is not None else conf_threshold
        self.conf_threshold = float(c_th)
        self.diou_threshold = float(diou_threshold)
        self.detector = detector
        self.use_cache = bool(use_cache)
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
        """Signal thread to cancel batch processing gracefully."""
        self._is_cancelled = True
        self.requestInterruption()
        logger.info("BatchTriageWorker: Cancellation/Interruption requested.")

    def is_cancelled(self) -> bool:
        """Check if cancellation or interruption has been requested."""
        return self._is_cancelled or self.isInterruptionRequested()

    def run(self) -> None:
        """Execute batch triage scan over all supported imagery files."""
        start_time = time.perf_counter()
        try:
            input_p = self.input_folder.resolve()
            if not input_p.exists() or not input_p.is_dir():
                err_msg = f"Вхідна папка не існує або не є директорією: {self.input_folder}"
                logger.error(err_msg)
                self.error_occurred.emit(err_msg)
                return

            out_p = self.output_folder.resolve() if self.output_folder else None
            if out_p is None or out_p == input_p:
                self.output_folder = input_p.parent / f"{input_p.name}_detected"

            self.output_folder.mkdir(parents=True, exist_ok=True)

            valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
            image_files = [
                p
                for p in sorted(self.input_folder.iterdir())
                if p.is_file() and p.suffix.lower() in valid_extensions
            ]
            total_files = len(image_files)

            if total_files == 0:
                self.progress_changed.emit(0, 0, "У вхідній папці немає підтримуваних зображень.")
                self.triage_completed.emit(0, 0, 0.0)
                return

            # Autonomous detector in background worker thread: if not supplied, instantiate
            # a dedicated UnifiedDetector so this worker operates on its own session without
            # CUDA stream or lock contention with the main GUI thread.
            detector = self.detector
            if detector is None:
                detector = UnifiedDetector()
                self.detector = detector

            positive_count = 0

            for idx, img_path in enumerate(image_files):
                # 1. External interruption check before reading file (exactly once)
                if self.is_cancelled():
                    logger.info("BatchTriageWorker: Interrupted before file %d/%d.", idx, total_files)
                    break

                # 1. Check existing JSON cache in input folder if enabled
                detections: List[Dict[str, Any]] = []
                img_w, img_h = 0, 0

                if self.use_cache and has_metadata_cache(img_path):
                    cached_data = load_metadata_cache(img_path.with_suffix(".json"))
                    if cached_data is not None:
                        detections = metadata_to_gui_detections(cached_data)
                        size = cached_data.get("image_size", [0, 0])
                        img_w, img_h = int(size[0]), int(size[1])

                # 2. If no cached detections, run inference
                if not detections and not (self.use_cache and has_metadata_cache(img_path)):
                    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        logger.warning("BatchTriageWorker: Failed to read image: %s", img_path)
                        continue

                    img_h, img_w = bgr.shape[:2]
                    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    del bgr  # Free raw BGR buffer immediately

                    detections = self._perform_detection(img_path, img_rgb, detector)
                    del img_rgb  # Free RGB tensor immediately after inference

                # 2. External interruption check after detection (exactly once)
                if self.is_cancelled():
                    logger.info("BatchTriageWorker: Interrupted after detection %d/%d.", idx, total_files)
                    break

                # 3. Triage decision: if any targets found (len(detections) > 0)
                if len(detections) > 0:
                    positive_count += 1
                    dest_img_path = self.output_folder / img_path.name

                    src_file = Path(img_path).resolve()
                    dst_file = Path(dest_img_path).resolve()

                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    if src_file != dst_file:
                        shutil.copy2(src_file, dst_file)

                    if img_w <= 0 or img_h <= 0:
                        img_w, img_h = 1024, 1024

                    save_metadata_cache(dst_file, (img_w, img_h), detections)

                self.file_processed.emit(str(img_path), len(detections))
                if idx % 5 == 0 or idx == total_files - 1:
                    self.progress_changed.emit(
                        idx + 1,
                        total_files,
                        f"Оброблено {idx + 1} із {total_files} файлів (знайдено цілей у {positive_count})...",
                    )

                del detections

            elapsed_seconds = time.perf_counter() - start_time
            logger.info(
                "BatchTriageWorker: Completed. Processed %d/%d files, %d positive in %.2fs.",
                len(image_files),
                total_files,
                positive_count,
                elapsed_seconds,
            )
            self.triage_completed.emit(total_files, positive_count, elapsed_seconds)

        except Exception as exc:
            logger.exception("BatchTriageWorker: Exception occurred: %s", exc)
            self.error_occurred.emit(str(exc))

    def _perform_detection(
        self, img_path: Path, img_rgb: np.ndarray, detector: Any
    ) -> List[Dict[str, Any]]:
        """Run detection using custom mock detector or standard tiled UnifiedDetector."""
        cnames = self.get_class_names()
        dets: List[Dict[str, Any]] = []
        if callable(detector):
            res = detector(img_rgb)
            dets = res if isinstance(res, list) else []
        elif hasattr(detector, "detect_image"):
            res = detector.detect_image(img_rgb)
            dets = res if isinstance(res, list) else []
        elif hasattr(detector, "predict"):
            res = detector.predict(img_rgb)
            dets = res if isinstance(res, list) else []
        elif hasattr(detector, "predict_tile"):
            dets = run_tiled_inference(
                img_rgb=img_rgb,
                altitude=self.altitude,
                vram_mb=self.vram_mb,
                conf_threshold=self.conf_threshold,
                diou_threshold=self.diou_threshold,
                detector=detector,
                class_names=cnames,
            )

        for d in dets:
            if isinstance(d, dict) and "class_id" in d and "class_name" not in d:
                cid = int(d["class_id"])
                d["class_name"] = cnames.get(cid, f"Class {cid}")
        return dets
