"""Operator Workstation Main Window.

Implements modern tactical GUI for viewing ultra-high-resolution aerial imagery,
configuring flight altitude and VRAM limits, managing non-destructive detection
overlays, and inspecting target lists.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import QPoint, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from gui.async_worker import InferenceWorker
from gui.canvas_viewer import (
    DEFAULT_CLASS_NAMES,
    CanvasViewer,
    get_class_color,
)

logger = logging.getLogger(__name__)

# Dark Tactical Operator Workstation Theme QSS
DARK_TACTICAL_STYLE = """
QMainWindow {
    background-color: #0b0f19;
    color: #e2e8f0;
}

QWidget {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #cbd5e1;
}

/* Sidebar and containers */
#sidebar {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}

QGroupBox {
    background-color: #131d31;
    border: 1px solid #24344d;
    border-radius: 6px;
    margin-top: 1.2em;
    padding: 10px;
    font-weight: bold;
    color: #38bdf8;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}

/* Buttons */
QPushButton {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 5px;
    color: #f8fafc;
    padding: 7px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
}

QPushButton:pressed {
    background-color: #0ea5e9;
    border-color: #0284c7;
    color: #ffffff;
}

QPushButton#btn_detect {
    background-color: #0284c7;
    border: 1px solid #38bdf8;
    color: #ffffff;
    font-weight: bold;
    font-size: 14px;
    padding: 9px;
}

QPushButton#btn_detect:hover {
    background-color: #0ea5e9;
    border-color: #7dd3fc;
}

QPushButton#btn_detect:disabled {
    background-color: #1e293b;
    border-color: #334155;
    color: #64748b;
}

/* SpinBox, ComboBox */
QSpinBox, QComboBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 5px 8px;
    color: #f8fafc;
}

QSpinBox:focus, QComboBox:focus {
    border-color: #38bdf8;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #334155;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    border: 1px solid #475569;
    selection-background-color: #0284c7;
    selection-color: #ffffff;
    color: #f8fafc;
}

/* Checkboxes */
QCheckBox {
    color: #e2e8f0;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #475569;
    background-color: #1e293b;
}

QCheckBox::indicator:checked {
    background-color: #0284c7;
    border-color: #38bdf8;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #334155;
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #0284c7;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #38bdf8;
    border: 1px solid #0284c7;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

/* Target List Table */
QTableWidget {
    background-color: #0f172a;
    border: 1px solid #24344d;
    border-radius: 4px;
    gridline-color: #1e293b;
    color: #f1f5f9;
    selection-background-color: #1e3a5f;
    selection-color: #38bdf8;
}

QHeaderView::section {
    background-color: #131d31;
    color: #94a3b8;
    padding: 5px;
    border: none;
    border-right: 1px solid #1e293b;
    border-bottom: 1px solid #1e293b;
    font-weight: 600;
}

/* Progress bar */
QProgressBar {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
    height: 14px;
}

QProgressBar::chunk {
    background-color: #0284c7;
    border-radius: 3px;
}

/* Status Bar */
QStatusBar {
    background-color: #0f172a;
    border-top: 1px solid #1e293b;
    color: #94a3b8;
}

QStatusBar::item {
    border: none;
}
"""


class DetectionSimulatorWorker(QThread):
    """Asynchronous worker that simulates detection pipeline without blocking the UI.

    Generates 3 to 5 realistic synthetic detections in the coordinate space of the
    loaded image (up to 8K), with simulated progress steps.
    """

    progress_changed = Signal(int)  # 0 to 100%
    detection_finished = Signal(list, float)  # (detections_json, elapsed_ms)

    def __init__(
        self,
        img_width: int,
        img_height: int,
        altitude: int,
        vram_mb: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.img_width = max(100, img_width)
        self.img_height = max(100, img_height)
        self.altitude = altitude
        self.vram_mb = vram_mb

    def run(self) -> None:
        start_time = time.perf_counter()

        # Step 1: Simulate dynamic tiling computation
        for step in range(10, 50, 10):
            time.sleep(0.02)
            self.progress_changed.emit(step)

        # Step 2: Simulate inference forward pass on tiles
        for step in range(50, 90, 10):
            time.sleep(0.025)
            self.progress_changed.emit(step)

        # Step 3: Simulate cluster DIoU NMS and offset remapping
        time.sleep(0.02)
        self.progress_changed.emit(100)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Generate 3-5 realistic synthetic detections in the image's coordinate space
        num_dets = random.randint(3, 5)
        detections: List[Dict[str, Any]] = []

        target_classes = [
            (0, "Легкова техніка", (50, 80), (35, 55)),
            (1, "Вантажний транспорт", (100, 160), (50, 75)),
            (2, "Бронетехніка", (90, 140), (60, 90)),
            (3, "Артилерія / ППО", (80, 130), (55, 80)),
            (4, "Авіація / БПЛА", (120, 200), (90, 150)),
        ]

        # Strategic clusters across the image area
        margin_x = int(self.img_width * 0.1)
        margin_y = int(self.img_height * 0.1)
        max_x = max(margin_x + 10, self.img_width - margin_x)
        max_y = max(margin_y + 10, self.img_height - margin_y)

        for i in range(num_dets):
            cls_idx, cls_name, (w_min, w_max), (h_min, h_max) = random.choice(target_classes)
            box_w = random.randint(w_min, w_max)
            box_h = random.randint(h_min, h_max)
            box_x = random.randint(margin_x, max(margin_x + 1, max_x - box_w))
            box_y = random.randint(margin_y, max(margin_y + 1, max_y - box_h))
            conf = round(random.uniform(0.72, 0.98), 3)

            detections.append(
                {
                    "id": i + 1,
                    "x": float(box_x),
                    "y": float(box_y),
                    "w": float(box_w),
                    "h": float(box_h),
                    "conf": conf,
                    "class_id": cls_idx,
                    "class_name": cls_name,
                }
            )

        self.detection_finished.emit(detections, elapsed_ms)


class MainWindow(QMainWindow):
    """Main workstation window for aerial reconnaissance and target detection."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Aerial Reconnaissance Workstation — Ultra-HD Vector View")
        self.resize(1440, 900)
        self.setMinimumSize(1024, 650)
        self.setStyleSheet(DARK_TACTICAL_STYLE)

        # State
        self._current_image_path: Optional[str] = None
        self._current_image_array: Optional[np.ndarray] = None
        self._is_mock_image: bool = False
        self._worker: Optional[InferenceWorker] = None
        self._class_checkboxes: Dict[int, QCheckBox] = {}

        self._init_ui()
        self._init_signals()
        self._update_status_bar_info()

    # --------------------------------------------------------------------------
    # UI Setup
    # --------------------------------------------------------------------------
    def _init_ui(self) -> None:
        """Construct ergonomic layout with central canvas, left sidebar, and status bar."""
        # Central Canvas View
        self.canvas = CanvasViewer(self)

        # Side Control Panel
        sidebar_widget = self._create_sidebar()

        # Splitter to allow resizing sidebar
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(sidebar_widget)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)  # Sidebar fixed size
        splitter.setStretchFactor(1, 1)  # Canvas expands
        splitter.setSizes([380, 1060])

        self.setCentralWidget(splitter)

        # Bottom Status Bar
        self._init_status_bar()

    def _create_sidebar(self) -> QWidget:
        """Create scrollable ergonomic side panel with controls and target list."""
        sidebar = QWidget(self)
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(380)

        main_layout = QVBoxLayout(sidebar)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # 1. Header and Input Section
        grp_input = QGroupBox("Вхідні дані", sidebar)
        layout_input = QVBoxLayout(grp_input)
        layout_input.setSpacing(8)

        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("📁 Відкрити фото", grp_input)
        self.btn_open.setToolTip("Відкрити аерофотознімок з диска (Ctrl+O)")
        self.btn_open.clicked.connect(self._on_open_image_dialog)

        self.btn_load_mock_8k = QPushButton("⚡ Тест 8K", grp_input)
        self.btn_load_mock_8k.setToolTip("Згенерувати тестовий кадр 8K (7680×4320)")
        self.btn_load_mock_8k.clicked.connect(self.load_mock_8k_image)

        btn_layout.addWidget(self.btn_open, stretch=2)
        btn_layout.addWidget(self.btn_load_mock_8k, stretch=1)
        layout_input.addLayout(btn_layout)

        self.lbl_image_info = QLabel("Файл: не вибрано\nРоздільність: 0 × 0 px", grp_input)
        self.lbl_image_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout_input.addWidget(self.lbl_image_info)

        main_layout.addWidget(grp_input)

        # 2. Flight & Hardware Parameters Section
        grp_params = QGroupBox("Параметри польоту та пам'яті", sidebar)
        layout_params = QGridLayout(grp_params)
        layout_params.setSpacing(8)

        # Altitude selector
        lbl_alt = QLabel("Висота польоту:", grp_params)
        self.spin_altitude = QSpinBox(grp_params)
        self.spin_altitude.setRange(10, 500)
        self.spin_altitude.setValue(120)
        self.spin_altitude.setSingleStep(10)
        self.spin_altitude.setSuffix(" м")
        self.spin_altitude.setToolTip("Висота зйомки БПЛА (10 - 500 метрів)")

        layout_params.addWidget(lbl_alt, 0, 0)
        layout_params.addWidget(self.spin_altitude, 0, 1)

        # VRAM selection
        lbl_vram = QLabel("Ліміт VRAM:", grp_params)
        self.combo_vram = QComboBox(grp_params)
        self.combo_vram.addItem("Авто (визначити VRAM)", -1)
        self.combo_vram.addItem("1024 MB (1 GB)", 1024)
        self.combo_vram.addItem("2048 MB (2 GB)", 2048)
        self.combo_vram.addItem("4096 MB (4 GB)", 4096)
        self.combo_vram.addItem("8192 MB (8 GB)", 8192)
        self.combo_vram.addItem("16384 MB (16 GB)", 16384)
        self.combo_vram.setCurrentIndex(0)
        self.combo_vram.setToolTip("Обмеження пам'яті GPU для розрахунку сітки тайлів")

        layout_params.addWidget(lbl_vram, 1, 0)
        layout_params.addWidget(self.combo_vram, 1, 1)

        # Start and Cancel Detection Buttons
        detect_layout = QHBoxLayout()
        self.btn_detect = QPushButton("▶ Старт детекції", grp_params)
        self.btn_detect.setObjectName("btn_detect")
        self.btn_detect.setEnabled(False)
        self.btn_detect.setToolTip("Запустити детекцію цілей на знімку (F5)")
        self.btn_detect.clicked.connect(self.start_detection)

        self.btn_cancel = QPushButton("⏹ Зупинити", grp_params)
        self.btn_cancel.setObjectName("btn_cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setToolTip("Зупинити процес детекції")
        self.btn_cancel.clicked.connect(self.cancel_detection)

        detect_layout.addWidget(self.btn_detect, stretch=3)
        detect_layout.addWidget(self.btn_cancel, stretch=1)
        layout_params.addLayout(detect_layout, 2, 0, 1, 2)

        main_layout.addWidget(grp_params)

        # 3. Filtering Section
        grp_filter = QGroupBox("Фільтрація відображення", sidebar)
        layout_filter = QVBoxLayout(grp_filter)
        layout_filter.setSpacing(6)

        # Confidence Slider
        slider_row = QHBoxLayout()
        lbl_conf_title = QLabel("Мін. впевненість:", grp_filter)
        self.lbl_conf_val = QLabel("25%", grp_filter)
        self.lbl_conf_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        slider_row.addWidget(lbl_conf_title)
        slider_row.addStretch()
        slider_row.addWidget(self.lbl_conf_val)
        layout_filter.addLayout(slider_row)

        self.slider_conf = QSlider(Qt.Orientation.Horizontal, grp_filter)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(25)
        self.slider_conf.valueChanged.connect(self._on_confidence_slider_changed)
        layout_filter.addWidget(self.slider_conf)

        # Class Checkboxes
        lbl_classes = QLabel("Видимі класи цілей:", grp_filter)
        lbl_classes.setStyleSheet("margin-top: 4px; font-weight: 600;")
        layout_filter.addWidget(lbl_classes)

        for class_id, class_name in DEFAULT_CLASS_NAMES.items():
            color = get_class_color(class_id)
            chk = QCheckBox(f"{class_name}", grp_filter)
            chk.setChecked(True)
            chk.setStyleSheet(
                f"QCheckBox {{ color: {color.name()}; font-weight: 500; }}"
            )
            chk.toggled.connect(
                lambda checked, cid=class_id: self.canvas.set_class_visibility(cid, checked)
            )
            self._class_checkboxes[class_id] = chk
            layout_filter.addWidget(chk)

        main_layout.addWidget(grp_filter)

        # 4. Target List Section
        grp_targets = QGroupBox("Виявлені цілі", sidebar)
        layout_targets = QVBoxLayout(grp_targets)
        layout_targets.setContentsMargins(6, 12, 6, 6)

        self.table_targets = QTableWidget(0, 4, grp_targets)
        self.table_targets.setHorizontalHeaderLabels(["#", "Клас", "Впевн.", "Коорд. (X, Y)"])
        self.table_targets.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_targets.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_targets.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_targets.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_targets.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_targets.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table_targets.verticalHeader().setVisible(False)
        self.table_targets.itemSelectionChanged.connect(self._on_target_table_selection)

        layout_targets.addWidget(self.table_targets)
        main_layout.addWidget(grp_targets, stretch=1)

        return sidebar

    def _init_status_bar(self) -> None:
        """Construct bottom status bar with progress bar, FPS, and telemetry."""
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        # Permanent widgets in status bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedWidth(160)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.lbl_status_proc = QLabel("Час: — | FPS: —", self)
        self.lbl_status_proc.setStyleSheet("padding: 0 8px; color: #38bdf8;")
        self.status_bar.addPermanentWidget(self.lbl_status_proc)

        self.lbl_status_coords = QLabel("X: — | Y: —", self)
        self.lbl_status_coords.setStyleSheet("padding: 0 8px; color: #94a3b8;")
        self.status_bar.addPermanentWidget(self.lbl_status_coords)

        self.lbl_status_zoom = QLabel("Zoom: 100%", self)
        self.lbl_status_zoom.setStyleSheet("padding: 0 8px; color: #f8fafc;")
        self.status_bar.addPermanentWidget(self.lbl_status_zoom)

        # Quick zoom buttons
        btn_fit = QPushButton("Вписати", self)
        btn_fit.setFixedHeight(22)
        btn_fit.clicked.connect(self.canvas.fit_to_view)
        self.status_bar.addPermanentWidget(btn_fit)

        btn_100 = QPushButton("100%", self)
        btn_100.setFixedHeight(22)
        btn_100.clicked.connect(self.canvas.reset_zoom)
        self.status_bar.addPermanentWidget(btn_100)

        self.status_bar.showMessage("Готовий до роботи")

    def _init_signals(self) -> None:
        """Connect internal signals between CanvasViewer and MainWindow."""
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.canvas.cursor_position_changed.connect(self._on_cursor_position_changed)
        self.canvas.detection_selected.connect(self._on_canvas_detection_selected)
        self.canvas.detections_updated.connect(self._on_visible_detections_count_updated)

        # Initialize initial filter values in CanvasViewer
        self.canvas.set_min_confidence(self.slider_conf.value() / 100.0)

    # --------------------------------------------------------------------------
    # Image Operations
    # --------------------------------------------------------------------------
    def _on_open_image_dialog(self) -> None:
        """Open file picker to select an aerial image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Вибрати аерофотознімок",
            "",
            "Зображення (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;Всі файли (*.*)",
        )
        if file_path:
            self.load_image_file(file_path)

    def load_image_file(self, file_path: str) -> bool:
        """Load image file and prepare workspace."""
        success = self.canvas.load_image(file_path)
        if success:
            self._current_image_path = file_path
            self._current_image_array = None
            self._is_mock_image = False
            w, h = self.canvas.get_image_size()
            self.lbl_image_info.setText(
                f"Файл: {file_path.split('/')[-1]}\nРоздільність: {w} × {h} px"
            )
            self.btn_detect.setEnabled(True)
            self.status_bar.showMessage(f"Завантажено: {file_path} ({w}×{h})", 4000)
            self.table_targets.setRowCount(0)
        else:
            QMessageBox.critical(self, "Помилка", f"Не вдалося відкрити файл:\n{file_path}")
        return success

    def load_mock_8k_image(self) -> None:
        """Generate and load a synthetic 8K (7680x4320) aerial image pattern for testing."""
        w, h = 7680, 4320
        pixmap = QPixmap(w, h)

        # Paint an aerial terrain pattern with coordinate grid
        painter = QPainter(pixmap)
        painter.fillRect(0, 0, w, h, QColor("#1e293b"))  # Dark terrain base

        # Draw grid lines every 640px (tile boundaries)
        grid_pen = QPen(QColor("#334155"), 2.0, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        for x in range(0, w, 640):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, 640):
            painter.drawLine(0, y, w, y)

        # Text banner
        banner_font = QFont("SansSerif", 48, QFont.Weight.Bold)
        painter.setFont(banner_font)
        painter.setPen(QColor("#0ea5e9"))
        painter.drawText(
            QRectF(0, 0, float(w), float(h)),
            Qt.AlignmentFlag.AlignCenter,
            "AERIAL RECONNAISSANCE 8K SYNTHETIC PATTERN (7680 × 4320)",
        )
        painter.end()

        # Generate lightweight synthetic RGB NumPy array for zero-copy worker slicing
        synth_arr = np.zeros((h, w, 3), dtype=np.uint8)
        synth_arr[:] = [30, 41, 59]  # Dark slate base

        self.canvas.set_image(pixmap)
        self._current_image_path = "synthetic_8k_pattern.png"
        self._current_image_array = synth_arr
        self._is_mock_image = True

        self.lbl_image_info.setText(
            f"Файл: synthetic_8k_pattern.png\nРоздільність: {w} × {h} px (8K Ultra-HD)"
        )
        self.btn_detect.setEnabled(True)
        self.status_bar.showMessage("Згенеровано та завантажено тестовий 8K кадр (7680×4320)", 4000)
        self.table_targets.setRowCount(0)

    # --------------------------------------------------------------------------
    # Asynchronous Detection Pipeline Integration
    # --------------------------------------------------------------------------
    def start_detection(self) -> None:
        """Start asynchronous detection processing via InferenceWorker."""
        if not self.canvas.has_image():
            return

        alt = float(self.spin_altitude.value())
        vram_data = self.combo_vram.currentData()
        vram = 2048 if vram_data == -1 else int(vram_data)
        conf_thresh = self.slider_conf.value() / 100.0

        image_source = (
            self._current_image_array
            if self._current_image_array is not None
            else (self._current_image_path or "synthetic_8k_pattern.png")
        )

        # Update UI state
        self.btn_detect.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Ініціалізація інференсу...")

        # Asynchronous inference worker
        self._worker = InferenceWorker(
            image_source=image_source,
            altitude=alt,
            vram_mb=vram,
            conf_threshold=conf_thresh,
            is_mock=self._is_mock_image,
            parent=self,
        )
        self._worker.progress_changed.connect(self._on_worker_progress)
        self._worker.detection_completed.connect(self._on_detection_finished)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def cancel_detection(self) -> None:
        """Request cancellation of running inference worker."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.status_bar.showMessage("Зупинка детекції...")
            self.btn_cancel.setEnabled(False)

    def _on_worker_progress(self, current: int, total: int, status_msg: str) -> None:
        """Handle progress updates from InferenceWorker."""
        pct = int((current / max(1, total)) * 100)
        self.progress_bar.setValue(pct)
        self.status_bar.showMessage(status_msg)

    def _on_worker_error(self, err_msg: str) -> None:
        """Handle error message emitted by InferenceWorker."""
        self.status_bar.showMessage(f"Помилка: {err_msg}", 5000)
        QMessageBox.warning(self, "Помилка інференсу", err_msg)

    def _on_worker_finished(self) -> None:
        """Reset UI state after worker termination."""
        self.progress_bar.setVisible(False)
        self.btn_detect.setEnabled(True)
        self.btn_cancel.setVisible(False)

    def _on_detection_finished(self, detections: List[Dict[str, Any]], elapsed_ms: float) -> None:
        """Handle detection results received from worker."""
        self.progress_bar.setVisible(False)
        self.btn_detect.setEnabled(True)
        self.btn_cancel.setVisible(False)

        # Update canvas overlay (absolute 8K pixel coordinates)
        self.canvas.update_detections(detections)

        # Populate target list table
        self._populate_target_table(detections)

        # Update telemetry and status bar
        fps = (1000.0 / elapsed_ms) if elapsed_ms > 0 else 0.0
        self.lbl_status_proc.setText(f"Час: {elapsed_ms:.1f} мс | FPS: {fps:.2f}")
        self.status_bar.showMessage(
            f"Детекцію завершено. Знайдено {len(detections)} цілей за {elapsed_ms:.1f} мс.", 5000
        )

    def _populate_target_table(self, detections: List[Dict[str, Any]]) -> None:
        """Populate the detected targets table widget."""
        self.table_targets.setRowCount(0)
        self.table_targets.setRowCount(len(detections))

        for row, det in enumerate(detections):
            det_id = str(det.get("id", row + 1))
            class_name = det.get("class_name", f"Клас {det.get('class_id', 0)}")
            conf = float(det.get("conf", 0.0))
            x = int(det.get("x", 0))
            y = int(det.get("y", 0))

            color = get_class_color(int(det.get("class_id", 0)))

            item_id = QTableWidgetItem(det_id)
            item_id.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_class = QTableWidgetItem(class_name)
            item_class.setForeground(color)

            item_conf = QTableWidgetItem(f"{conf:.1%}")
            item_conf.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_coords = QTableWidgetItem(f"{x}, {y}")
            item_coords.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.table_targets.setItem(row, 0, item_id)
            self.table_targets.setItem(row, 1, item_class)
            self.table_targets.setItem(row, 2, item_conf)
            self.table_targets.setItem(row, 3, item_coords)

    # --------------------------------------------------------------------------
    # User Interactivity & Telemetry Handlers
    # --------------------------------------------------------------------------
    def _on_target_table_selection(self) -> None:
        """Focus and center canvas on selected target from table."""
        selected_rows = self.table_targets.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            self.canvas.focus_on_detection(row, target_zoom=1.5)

    def _on_canvas_detection_selected(self, det: Dict[str, Any]) -> None:
        """Sync table row selection when a detection box is clicked in canvas."""
        det_id = det.get("id")
        for row in range(self.table_targets.rowCount()):
            item = self.table_targets.item(row, 0)
            if item and item.text() == str(det_id):
                self.table_targets.selectRow(row)
                break

    def _on_confidence_slider_changed(self, val: int) -> None:
        """Filter detections in canvas by minimum confidence."""
        self.lbl_conf_val.setText(f"{val}%")
        self.canvas.set_min_confidence(val / 100.0)

    def _on_visible_detections_count_updated(self, visible_count: int) -> None:
        """Update status bar with count of visible detections."""
        total = len(self.canvas.get_detections())
        if total > 0:
            self.status_bar.showMessage(
                f"Відображається цілей: {visible_count} з {total}", 3000
            )

    def _on_zoom_changed(self, zoom: float) -> None:
        """Update zoom telemetry indicator."""
        self.lbl_status_zoom.setText(f"Zoom: {int(zoom * 100)}%")

    def _on_cursor_position_changed(self, x: int, y: int) -> None:
        """Update cursor coordinate tracker in status bar."""
        self.lbl_status_coords.setText(f"X: {x} | Y: {y}")

    def _update_status_bar_info(self) -> None:
        """Set initial status bar text."""
        self.lbl_status_coords.setText("X: — | Y: —")
        self.lbl_status_zoom.setText("Zoom: 100%")

