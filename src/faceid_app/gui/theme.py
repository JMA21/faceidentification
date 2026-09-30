"""Shared visual styling for the main window and its settings dialog."""

APP_STYLESHEET = """
QMainWindow, QDialog { background: #f3f6fb; }
QWidget { color: #243247; font-family: "Segoe UI"; font-size: 10pt; }
QMenuBar, QMenu, QStatusBar { background: #f3f6fb; }
QLabel#appTitle { font-size: 23pt; font-weight: 700; color: #172a46; }
QLabel#subtitle, QLabel#searchInfo { color: #617189; }
QLabel#sectionTitle { font-size: 12pt; font-weight: 600; }
QLabel#versionBadge, QLabel#resultCount {
    background: #e6edf8; color: #315a96; border-radius: 8px; padding: 6px 12px;
}
QFrame#card, QGroupBox {
    background: white; border: 1px solid #dde5f0; border-radius: 12px;
}
QGroupBox { margin-top: 14px; padding: 16px 12px 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QPushButton {
    background: white; border: 1px solid #ccd7e6; border-radius: 7px;
    padding: 8px 12px; font-weight: 600;
}
QPushButton:hover { background: #edf3fc; border-color: #7d9dcd; }
QPushButton:pressed { background: #dde8f8; }
QPushButton[primary="true"] { background: #2563eb; color: white; border-color: #2563eb; }
QPushButton[primary="true"]:hover { background: #1d4ed8; }
QPushButton:disabled { background: #edf1f6; color: #8c99ab; border-color: #e0e6ef; }
QLineEdit, QComboBox, QListWidget, QPlainTextEdit {
    background: white; border: 1px solid #d5dfec; border-radius: 7px;
    padding: 7px; selection-background-color: #2563eb; selection-color: white;
}
QLineEdit:focus, QComboBox:focus { border-color: #2563eb; }
QComboBox { min-width: 160px; }
QComboBox QAbstractItemView { background: white; color: #243247; }
QListView#results { background: white; border: none; padding: 4px; }
QLabel#emptyState { color: #617189; padding: 28px; font-size: 12pt; }
QPlainTextEdit#journal { font-family: "Consolas"; font-size: 9pt; background: #f8fafd; }
QProgressBar { border: none; background: #e1e8f3; border-radius: 5px; text-align: center; }
QProgressBar::chunk { background: #2563eb; border-radius: 5px; }
QSplitter::handle { background: #e1e8f3; height: 4px; }
QSlider::groove:horizontal { height: 6px; background: #dde5f0; border-radius: 3px; }
QSlider::handle:horizontal { background: #2563eb; width: 16px; margin: -5px 0; border-radius: 8px; }
"""