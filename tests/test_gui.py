from pathlib import Path
import sqlite3

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEventLoop, QSize, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from faceid_app.config.settings import AppSettings
from faceid_app.gui.main_window import MainWindow, SearchResultsModel
from faceid_app.gui.search_worker import SearchWorker
from faceid_app.gui.settings_dialog import SettingsDialog
from faceid_app.models import FaceDetection, MatchCandidate
from faceid_app.storage.database import connect, initialize_database, resolve_database_path
from faceid_app.storage.repositories import FaceIndexRepository


@pytest.fixture(scope="module")
def app():
    import os

    previous = os.environ.get("QT_QPA_PLATFORM")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    application = QApplication.instance() or QApplication([])
    yield application
    if previous is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = previous


def candidate(path: Path, face_id: int = 1) -> MatchCandidate:
    return MatchCandidate(1, "Alice", face_id, path, 0.9, (0, 0, 10, 10))


def test_thumbnail_dimensions_cache_limit_and_missing_image(app, tmp_path):
    model = SearchResultsModel(QSize(160, 120))
    model._thumbnail_cache_limit = 2
    paths = [tmp_path / f"{index}.jpg" for index in range(3)]
    for path in paths:
        Image.new("RGB", (2400, 1800), "#aaccee").save(path)
    model.set_matches([candidate(path, index) for index, path in enumerate(paths)])
    thumbnail = model.index(0).data(model.THUMBNAIL_ROLE)
    assert thumbnail.size() == QSize(160, 120)
    assert model.index(0).data(Qt.ItemDataRole.ToolTipRole) == str(paths[0])
    assert model.index(0).data(model.THUMBNAIL_ROLE).cacheKey() == thumbnail.cacheKey()
    model.index(1).data(model.THUMBNAIL_ROLE)
    model.index(0).data(model.THUMBNAIL_ROLE)
    model.index(2).data(model.THUMBNAIL_ROLE)
    assert len(model._thumbnail_cache) == 2
    assert str(paths[1]) not in model._thumbnail_cache
    missing = tmp_path / "missing.jpg"
    assert model._get_thumbnail(missing).isNull()
    assert model._get_thumbnail(missing).isNull()
    model.clear()
    assert model.rowCount() == 0
    assert not model._thumbnail_cache


def test_thumbnail_respects_exif_rotation(app, tmp_path):
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (800, 400), "#aaccee")
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    thumbnail = SearchResultsModel(QSize(160, 120))._get_thumbnail(path)
    assert thumbnail.height() > thumbnail.width()
    assert thumbnail.height() <= 120


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    settings = AppSettings(storage_directory=str(tmp_path), last_selected_person="Alice")
    database_path = initialize_database(resolve_database_path(tmp_path))
    connection = connect(database_path)
    repository = FaceIndexRepository(connection)
    detection = FaceDetection((0, 0, 10, 10), np.array([1, 0], dtype=np.float32), 0.99)
    ref_id, _ = repository.upsert_reference_image(tmp_path / "Alice.png", "Alice", 1, 1, "1:1")
    repository.replace_reference_faces(ref_id, [detection])
    photo_path = tmp_path / "photo.jpg"
    Image.new("RGB", (800, 600), "#aaccee").save(photo_path)
    photo_id, _ = repository.upsert_photo(tmp_path, photo_path, 1, 1, "1:1")
    repository.replace_photo_faces(photo_id, [detection])
    repository.mark_photo_completed(photo_id)
    repository.commit()
    connection.close()
    monkeypatch.setattr(AppSettings, "load", classmethod(lambda cls, settings_path=None: settings))
    monkeypatch.setattr(AppSettings, "save", lambda self, settings_path=None: tmp_path / "settings.json")
    widget = MainWindow()
    yield widget
    if widget._search_thread is not None:
        wait_for_search(widget)
    widget.close()
    widget.deleteLater()
    app.processEvents()


def wait_for_search(window):
    thread = window._search_thread
    assert thread is not None
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    thread.finished.connect(loop.quit)
    timer.start(5000)
    loop.exec()
    assert window._search_thread is None, "Search thread did not finish"


def test_background_search_and_result_actions(app, window):
    assert window.person_combo.currentText() == "Alice"
    assert not window.export_photos_button.isEnabled()
    assert not window.validate_button.isEnabled()
    window._run_search()
    assert not window.search_button.isEnabled()
    assert not window.settings_button.isEnabled()
    wait_for_search(window)
    assert window.search_button.isEnabled()
    assert window.export_photos_button.isEnabled()
    assert window._search_results_model.rowCount() == 1
    assert window.results_count_label.text() == "1 résultat"
    window.results_list.setCurrentIndex(window._search_results_model.index(0))
    assert window.validate_button.isEnabled()
    window.show()
    app.processEvents()
    assert not window.grab().isNull()
    window._apply_validation("rejected")
    wait_for_search(window)
    assert window._search_results_model.rowCount() == 0
    assert not window.export_photos_button.isEnabled()
    assert not window.validate_button.isEnabled()
    assert not window.empty_results_label.isHidden()


def test_search_failure_restores_controls(window, monkeypatch):
    monkeypatch.setattr("faceid_app.gui.search_worker.FaceMatcher.search_person", lambda *args: 1 / 0)
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[-1]))
    window._run_search()
    wait_for_search(window)
    assert errors
    assert window.search_button.isEnabled()
    assert window.settings_button.isEnabled()
    assert window._search_worker is None


def test_log_is_plain_text_and_bounded(window):
    window._log("<b>Texte brut</b>")
    assert "<b>Texte brut</b>" in window.log_output.toPlainText()
    window._log("\n".join(f"Line {index}" for index in range(1600)))
    assert window.log_output.document().blockCount() == 1500


def test_settings_dialog_renders(app, window):
    dialog = SettingsDialog(window.settings, window)
    dialog.show()
    app.processEvents()
    assert not dialog.grab().isNull()
    assert dialog.performance_mode_combo.count() == 3
    dialog.close()


def test_search_worker_closes_connection_on_error(app, monkeypatch, tmp_path):
    connection = connect(initialize_database(tmp_path / "index.sqlite3"))
    monkeypatch.setattr("faceid_app.gui.search_worker.connect", lambda path: connection)
    monkeypatch.setattr("faceid_app.gui.search_worker.FaceMatcher.search_person", lambda *args: 1 / 0)
    worker = SearchWorker(tmp_path / "index.sqlite3", "Alice", 0.5)
    errors = []
    worker.failed.connect(errors.append)
    worker.run()
    assert errors
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")