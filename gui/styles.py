"""Unified Tactical Dark Military Theme stylesheet for Aerial Reconnaissance Workstation."""

DARK_TACTICAL_STYLE = """
/* ========================================================================= */
/*                   DARK MILITARY / TACTICAL THEME (QSS)                    */
/* ========================================================================= */

QMainWindow, QDialog {
    background-color: #0b1326;
    color: #e2e8f0;
}

QWidget {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #cbd5e1;
}

/* Sidebar and panel containers */
#sidebar, QWidget#sidebar {
    background-color: #0b1326;
    border-right: 1px solid #1e3563;
}

QGroupBox {
    background-color: #0f1d3a;
    border: 1px solid #1e3563;
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
    padding: 0 6px;
    color: #38bdf8;
    background-color: #0f1d3a;
    border-radius: 3px;
}

/* Top Menu Bar */
QMenuBar {
    background-color: #0b1326;
    color: #e2e8f0;
    border-bottom: 1px solid #1e3563;
    padding: 2px;
}

QMenuBar::item {
    background: transparent;
    padding: 5px 10px;
    color: #e2e8f0;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #1e3563;
    color: #38bdf8;
}

QMenuBar::item:pressed {
    background-color: #0284c7;
    color: #ffffff;
}

/* Dropdown Menus */
QMenu {
    background-color: #0f1d3a;
    color: #e2e8f0;
    border: 1px solid #1e3563;
    border-radius: 6px;
    padding: 5px;
}

QMenu::item {
    padding: 6px 24px;
    color: #e2e8f0;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #1e3563;
    color: #38bdf8;
}

QMenu::separator {
    height: 1px;
    background-color: #1e3563;
    margin: 4px 6px;
}

/* Top Toolbar */
QToolBar {
    background-color: #0b1326;
    border-bottom: 1px solid #1e3563;
    spacing: 8px;
    padding: 4px 8px;
}

QToolBar::separator {
    width: 1px;
    background-color: #1e3563;
    margin: 4px 6px;
}

QToolButton {
    color: #e2e8f0;
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 5px 10px;
    font-weight: 500;
}

QToolButton:hover {
    background-color: #1e3563;
    border-color: #38bdf8;
    color: #38bdf8;
}

QToolButton:pressed {
    background-color: #0284c7;
    border-color: #38bdf8;
    color: #ffffff;
}

QToolButton:checked {
    background-color: #1e3563;
    border-color: #0284c7;
    color: #38bdf8;
}

/* Buttons */
QPushButton {
    background-color: #1e293b;
    border: 1px solid #1e3563;
    border-radius: 5px;
    color: #f8fafc;
    padding: 7px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #1e3563;
    border-color: #38bdf8;
    color: #38bdf8;
}

QPushButton:pressed {
    background-color: #0284c7;
    border-color: #38bdf8;
    color: #ffffff;
}

QPushButton:disabled {
    background-color: #0f172a;
    border-color: #1e293b;
    color: #475569;
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

/* Inputs: SpinBox, ComboBox, LineEdit */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: #080f1e;
    border: 1px solid #1e3563;
    border-radius: 4px;
    padding: 5px 8px;
    color: #f8fafc;
}

QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
    border-color: #38bdf8;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #1e3563;
}

QComboBox QAbstractItemView {
    background-color: #0f1d3a;
    border: 1px solid #1e3563;
    selection-background-color: #1e3563;
    selection-color: #38bdf8;
    color: #f8fafc;
    outline: none;
}

/* Checkboxes */
QCheckBox {
    color: #e2e8f0;
    spacing: 6px;
    background: transparent;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 3px;
    border: 1px solid #1e3563;
    background-color: #080f1e;
}

QCheckBox::indicator:hover {
    border-color: #38bdf8;
}

QCheckBox::indicator:checked {
    background-color: #0284c7;
    border-color: #38bdf8;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #1e3563;
    height: 6px;
    background: #080f1e;
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

/* Target List Table & File List */
QTableWidget, QListWidget {
    background-color: #080f1e;
    border: 1px solid #1e3563;
    border-radius: 4px;
    gridline-color: #1e3563;
    color: #f1f5f9;
    selection-background-color: #1e3563;
    selection-color: #38bdf8;
    outline: none;
}

QListWidget::item {
    padding: 4px 6px;
    border-bottom: 1px solid #0f1d3a;
}

QListWidget::item:hover {
    background-color: #0f1d3a;
    color: #ffffff;
}

QListWidget::item:selected {
    background-color: #1e3563;
    color: #38bdf8;
}

QHeaderView::section {
    background-color: #0f1d3a;
    color: #94a3b8;
    padding: 5px;
    border: none;
    border-right: 1px solid #1e3563;
    border-bottom: 1px solid #1e3563;
    font-weight: 600;
}

/* Progress bar */
QProgressBar {
    background-color: #080f1e;
    border: 1px solid #1e3563;
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
    background-color: #0b1326;
    border-top: 1px solid #1e3563;
    color: #94a3b8;
}

QStatusBar::item {
    border: none;
}

/* Splitter */
QSplitter {
    background-color: #0b1326;
}

QSplitter::handle {
    background-color: #1e3563;
    width: 2px;
    height: 2px;
}

QSplitter::handle:hover {
    background-color: #38bdf8;
}

/* Scroll Area & Bars */
QScrollArea, QScrollArea > QWidget > QWidget, #scroll_classes_content {
    background-color: #080f1e;
    border: 1px solid #1e3563;
    border-radius: 4px;
}

QScrollBar:vertical {
    border: none;
    background: #0b1326;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #1e3563;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #38bdf8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #0b1326;
    height: 8px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #1e3563;
    min-width: 20px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background: #38bdf8;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Tooltips */
QToolTip {
    background-color: #0f1d3a;
    color: #f8fafc;
    border: 1px solid #38bdf8;
    padding: 4px 8px;
    border-radius: 4px;
}
"""
