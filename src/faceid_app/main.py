from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from faceid_app.gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Face ID Local App")
    app.setOrganizationName("Local")

    window = MainWindow()
    window.show()
    return app.exec()
