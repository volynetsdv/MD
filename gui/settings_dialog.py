"""Settings and preferences dialog for Aerial Reconnaissance Workstation."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import AppConfig, get_config
from gui.styles import DARK_TACTICAL_STYLE

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Application preferences and configuration dialog."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Параметри застосунку")
        self.resize(520, 520)
        self.setStyleSheet(DARK_TACTICAL_STYLE)

        self.cfg = get_config()
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # 1. Detection Group
        grp_det = QGroupBox("Параметри детекції та інференсу", self)
        form_det = QFormLayout(grp_det)
        form_det.setSpacing(8)

        self.spin_conf = QDoubleSpinBox(grp_det)
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(self.cfg.default_confidence)
        self.spin_conf.setToolTip("Поріг впевненості детекції за замовчуванням [0.01 - 1.00]")
        form_det.addRow("Поріг впевненості (Confidence):", self.spin_conf)

        self.spin_iou = QDoubleSpinBox(grp_det)
        self.spin_iou.setRange(0.01, 1.0)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setValue(self.cfg.default_iou)
        self.spin_iou.setToolTip("Поріг IoU для NMS придушення дублікатів [0.01 - 1.00]")
        form_det.addRow("Поріг перекриття (IoU / NMS):", self.spin_iou)

        self.spin_alt = QDoubleSpinBox(grp_det)
        self.spin_alt.setRange(10.0, 1000.0)
        self.spin_alt.setSingleStep(10.0)
        self.spin_alt.setSuffix(" м")
        self.spin_alt.setValue(self.cfg.default_altitude)
        form_det.addRow("Базова висота польоту:", self.spin_alt)

        self.combo_vram = QComboBox(grp_det)
        vram_options = [
            (1024, "1024 MB (1 GB)"),
            (2048, "2048 MB (2 GB)"),
            (4096, "4096 MB (4 GB)"),
            (8192, "8192 MB (8 GB)"),
            (16384, "16384 MB (16 GB)"),
        ]
        for mb, name in vram_options:
            self.combo_vram.addItem(name, mb)
            if self.cfg.default_vram_limit_mb == mb:
                self.combo_vram.setCurrentIndex(self.combo_vram.count() - 1)
        form_det.addRow("Ліміт VRAM за замовчуванням:", self.combo_vram)

        main_layout.addWidget(grp_det)

        # 2. Visualization Group
        grp_vis = QGroupBox("Візуалізація та оверлей цілей", self)
        form_vis = QFormLayout(grp_vis)
        form_vis.setSpacing(8)

        self.chk_ids_only = QCheckBox("Відображати ID класу замість назви (#ID)", grp_vis)
        self.chk_ids_only.setChecked(self.cfg.show_class_ids_only)
        self.chk_ids_only.setToolTip("Компактні мітки виду #09 замість повної назви класу")
        form_vis.addRow(self.chk_ids_only)

        self.chk_scale_zoom = QCheckBox("Масштабувати шрифт разом із зумом", grp_vis)
        self.chk_scale_zoom.setChecked(self.cfg.scale_overlay_text_with_zoom)
        self.chk_scale_zoom.setToolTip(
            "Якщо вимкнено — розмір шрифту фіксований на екрані незалежно від зуму"
        )
        form_vis.addRow(self.chk_scale_zoom)

        self.spin_font_size = QSpinBox(grp_vis)
        self.spin_font_size.setRange(6, 36)
        self.spin_font_size.setValue(self.cfg.base_font_size)
        self.spin_font_size.setSuffix(" pt")
        form_vis.addRow("Розмір базового шрифту підписів:", self.spin_font_size)

        self.spin_box_width = QSpinBox(grp_vis)
        self.spin_box_width.setRange(1, 10)
        self.spin_box_width.setValue(self.cfg.box_border_width)
        self.spin_box_width.setSuffix(" px")
        form_vis.addRow("Товщина лінії рамок (border width):", self.spin_box_width)

        main_layout.addWidget(grp_vis)

        # 3. Storage Paths Group
        grp_paths = QGroupBox("Шляхи до файлів та моделей", self)
        form_paths = QFormLayout(grp_paths)
        form_paths.setSpacing(8)

        row_models = QHBoxLayout()
        self.txt_models_dir = QLineEdit(self.cfg.models_dir, grp_paths)
        btn_browse_models = QPushButton("Огляд...", grp_paths)
        btn_browse_models.clicked.connect(self._browse_models_dir)
        row_models.addWidget(self.txt_models_dir)
        row_models.addWidget(btn_browse_models)
        form_paths.addRow("Каталог моделей:", row_models)

        row_recent = QHBoxLayout()
        self.txt_recent_dir = QLineEdit(self.cfg.recent_dir, grp_paths)
        btn_browse_recent = QPushButton("Огляд...", grp_paths)
        btn_browse_recent.clicked.connect(self._browse_recent_dir)
        row_recent.addWidget(self.txt_recent_dir)
        row_recent.addWidget(btn_browse_recent)
        form_paths.addRow("Робоча папка знімків:", row_recent)

        self.btn_download_models = QPushButton("Завантажити / Оновити ваги з релізу...", grp_paths)
        self.btn_download_models.clicked.connect(self._download_models)
        form_paths.addRow("Автозавантаження ваг:", self.btn_download_models)

        main_layout.addWidget(grp_paths)

        # 4. Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Скинути до стандартних (Reset to Defaults)", self)
        self.btn_reset.setStyleSheet("color: #f87171; border-color: #ef4444;")
        self.btn_reset.clicked.connect(self._on_reset_to_defaults)
        btn_layout.addWidget(self.btn_reset)

        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Скасувати", self)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Зберегти", self)
        self.btn_save.setObjectName("btn_detect")
        self.btn_save.clicked.connect(self._on_save)
        btn_layout.addWidget(self.btn_save)

        main_layout.addLayout(btn_layout)

    def _browse_models_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(
            self, "Обрати каталог моделей", self.txt_models_dir.text()
        )
        if p:
            self.txt_models_dir.setText(p)

    def _browse_recent_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(
            self, "Обрати робочу папку знімків", self.txt_recent_dir.text()
        )
        if p:
            self.txt_recent_dir.setText(p)

    def _download_models(self) -> None:
        """Download missing or updated weights from configured URLs."""
        from PySide6.QtWidgets import QMessageBox

        models_dir = self.txt_models_dir.text().strip() or "models"
        self.btn_download_models.setEnabled(False)
        self.btn_download_models.setText("Завантаження ваг...")
        try:
            from scripts.download_weights import ensure_models

            success = ensure_models(models_dir=models_dir)
            if success:
                QMessageBox.information(
                    self,
                    "Завантаження ваг",
                    f"Усі моделі успішно перевірено та готові до роботи в каталозі '{models_dir}'.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Завантаження ваг",
                    "Не вдалося завантажити всі файли моделей. Перевірте підключення до мережі або URL-адреси.",
                )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Помилка завантаження",
                f"Виникла помилка під час завантаження ваг:\n{exc}",
            )
        finally:
            self.btn_download_models.setEnabled(True)
            self.btn_download_models.setText("Завантажити / Оновити ваги з релізу...")

    def _on_reset_to_defaults(self) -> None:
        """Reset form controls to standard defaults."""
        defaults = AppConfig()
        self.spin_conf.setValue(defaults.default_confidence)
        self.spin_iou.setValue(defaults.default_iou)
        self.spin_alt.setValue(defaults.default_altitude)
        for i in range(self.combo_vram.count()):
            if self.combo_vram.itemData(i) == defaults.default_vram_limit_mb:
                self.combo_vram.setCurrentIndex(i)
                break
        self.chk_ids_only.setChecked(defaults.show_class_ids_only)
        self.chk_scale_zoom.setChecked(defaults.scale_overlay_text_with_zoom)
        self.spin_font_size.setValue(defaults.base_font_size)
        self.spin_box_width.setValue(defaults.box_border_width)
        self.txt_models_dir.setText(defaults.models_dir)
        self.txt_recent_dir.setText(defaults.recent_dir)

    def _on_save(self) -> None:
        """Save settings to AppConfig and persist to disk."""
        self.cfg.default_confidence = self.spin_conf.value()
        self.cfg.default_iou = self.spin_iou.value()
        self.cfg.default_altitude = self.spin_alt.value()
        self.cfg.default_vram_limit_mb = int(self.combo_vram.currentData())
        self.cfg.show_class_ids_only = self.chk_ids_only.isChecked()
        self.cfg.scale_overlay_text_with_zoom = self.chk_scale_zoom.isChecked()
        self.cfg.base_font_size = self.spin_font_size.value()
        self.cfg.box_border_width = self.spin_box_width.value()
        self.cfg.models_dir = self.txt_models_dir.text().strip()
        self.cfg.recent_dir = self.txt_recent_dir.text().strip()
        self.cfg.save()
        self.accept()
