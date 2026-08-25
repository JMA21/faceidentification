from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import QAbstractListModel, QEvent, QModelIndex, QRect, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import QCompleter

from faceid_app import __version__
from PySide6.QtGui import QMouseEvent, QPainter, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from faceid_app.config.settings import AppSettings
from faceid_app.core.face_engine import detect_engine_version
from faceid_app.core.matcher import FaceMatcher
from faceid_app.models import MatchCandidate
from faceid_app.storage.database import initialize_database, resolve_database_path, connect
from faceid_app.storage.repositories import FaceIndexRepository
from faceid_app.gui.indexing_worker import IndexingWorker
from faceid_app.gui.settings_dialog import SettingsDialog


class SearchResultsModel(QAbstractListModel):
    MATCH_ROLE = Qt.ItemDataRole.UserRole + 1
    FILE_NAME_ROLE = Qt.ItemDataRole.UserRole + 2
    PATH_ROLE = Qt.ItemDataRole.UserRole + 3
    SIMILARITY_ROLE = Qt.ItemDataRole.UserRole + 4
    THUMBNAIL_ROLE = Qt.ItemDataRole.UserRole + 5

    def __init__(self, icon_size: QSize, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._matches: list[MatchCandidate] = []
        self._icon_size = QSize(icon_size)
        self._thumbnail_cache: dict[str, QPixmap] = {}

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        if parent.isValid():
            return 0
        return len(self._matches)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:  # type: ignore[override]
        if not index.isValid() or not (0 <= index.row() < len(self._matches)):
            return None
        match = self._matches[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return f"{match.person_name} | similarité={match.similarity:.3f}"
        if role == self.MATCH_ROLE:
            return match
        if role == self.FILE_NAME_ROLE:
            return match.photo_path.name
        if role == self.PATH_ROLE:
            return str(match.photo_path)
        if role == self.SIMILARITY_ROLE:
            return match.similarity
        if role == self.THUMBNAIL_ROLE:
            return self._get_thumbnail(match.photo_path)
        return None

    def set_matches(self, matches: list[MatchCandidate]) -> None:
        self.beginResetModel()
        self._matches = list(matches)
        self._thumbnail_cache.clear()
        self.endResetModel()

    def clear(self) -> None:
        self.set_matches([])

    def match_at(self, index: QModelIndex) -> MatchCandidate | None:
        if not index.isValid() or not (0 <= index.row() < len(self._matches)):
            return None
        return self._matches[index.row()]

    def _get_thumbnail(self, photo_path: Path) -> QPixmap | None:
        key = str(photo_path)
        if key in self._thumbnail_cache:
            return self._thumbnail_cache[key]
        pixmap = QPixmap(key)
        if pixmap.isNull():
            self._thumbnail_cache[key] = QPixmap()
            return None
        scaled = pixmap.scaled(
            self._icon_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumbnail_cache[key] = scaled
        return scaled


class SearchResultDelegate(QStyledItemDelegate):
    pathClicked = Signal(str)

    def __init__(self, icon_size: QSize, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icon_size = QSize(icon_size)
        self._outer_padding = 6
        self._line_spacing = 3

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:  # type: ignore[override]
        painter.save()

        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        if is_selected:
            painter.fillRect(option.rect, option.palette.highlight())

        content_rect = option.rect.adjusted(
            self._outer_padding,
            self._outer_padding,
            -self._outer_padding,
            -self._outer_padding,
        )

        thumb_rect = QRect(content_rect.topLeft(), self._icon_size)
        thumb_rect.moveTop(content_rect.top() + max(0, (content_rect.height() - self._icon_size.height()) // 2))

        thumbnail = index.data(SearchResultsModel.THUMBNAIL_ROLE)
        if isinstance(thumbnail, QPixmap) and not thumbnail.isNull():
            draw_x = thumb_rect.x() + (thumb_rect.width() - thumbnail.width()) // 2
            draw_y = thumb_rect.y() + (thumb_rect.height() - thumbnail.height()) // 2
            painter.drawPixmap(draw_x, draw_y, thumbnail)

        text_rect = QRect(
            thumb_rect.right() + 10,
            content_rect.top(),
            max(10, content_rect.width() - thumb_rect.width() - 10),
            content_rect.height(),
        )

        normal_pen = option.palette.highlightedText().color() if is_selected else option.palette.text().color()
        link_pen = option.palette.link().color()

        title = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        file_name = str(index.data(SearchResultsModel.FILE_NAME_ROLE) or "")
        full_path = str(index.data(SearchResultsModel.PATH_ROLE) or "")

        line_height = option.fontMetrics.height()
        y = text_rect.top() + line_height

        painter.setPen(normal_pen)
        painter.drawText(text_rect.left(), y, title)
        y += line_height + self._line_spacing
        painter.drawText(text_rect.left(), y, f"Fichier: {file_name}")
        y += line_height + self._line_spacing

        painter.setPen(link_pen)
        painter.drawText(text_rect.left(), y, f"Chemin: {full_path}")

        painter.restore()

    def sizeHint(self, option, index: QModelIndex) -> QSize:  # type: ignore[override]
        line_height = option.fontMetrics.height()
        text_height = (line_height * 3) + (self._line_spacing * 2)
        height = max(self._icon_size.height(), text_height) + (self._outer_padding * 2)
        return QSize(100, height)

    def editorEvent(self, event, model, option, index: QModelIndex) -> bool:  # type: ignore[override]
        if event.type() != QEvent.Type.MouseButtonRelease:
            return super().editorEvent(event, model, option, index)
        if not isinstance(event, QMouseEvent):
            return super().editorEvent(event, model, option, index)

        content_rect = option.rect.adjusted(
            self._outer_padding,
            self._outer_padding,
            -self._outer_padding,
            -self._outer_padding,
        )
        thumb_rect = QRect(content_rect.topLeft(), self._icon_size)
        thumb_rect.moveTop(content_rect.top() + max(0, (content_rect.height() - self._icon_size.height()) // 2))
        text_rect = QRect(
            thumb_rect.right() + 10,
            content_rect.top(),
            max(10, content_rect.width() - thumb_rect.width() - 10),
            content_rect.height(),
        )

        line_height = option.fontMetrics.height()
        path_top = text_rect.top() + (line_height * 2) + (self._line_spacing * 2)
        path_rect = QRect(text_rect.left(), path_top, text_rect.width(), line_height + 4)

        if path_rect.contains(event.position().toPoint()):
            full_path = str(index.data(SearchResultsModel.PATH_ROLE) or "")
            if full_path:
                self.pathClicked.emit(full_path)
                return True

        return super().editorEvent(event, model, option, index)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self.setWindowTitle("Face ID Local App")
        self.resize(self.settings.window.width, self.settings.window.height)

        self.person_combo = QComboBox()
        self.person_combo.setEditable(True)
        self.person_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.person_combo.lineEdit().setPlaceholderText("Saisir un prénom ou un nom…")
        self._person_completer_model = QStandardItemModel()
        completer = QCompleter(self._person_completer_model, self.person_combo)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionRole(Qt.ItemDataRole.DisplayRole)
        self.person_combo.setCompleter(completer)
        self.results_list = QListView()
        self._result_icon_size = QSize(160, 120)
        self.results_list.setIconSize(self._result_icon_size)
        self.results_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results_list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._search_results_model = SearchResultsModel(icon_size=self._result_icon_size, parent=self)
        self.results_list.setModel(self._search_results_model)
        self._search_result_delegate = SearchResultDelegate(icon_size=self._result_icon_size, parent=self.results_list)
        self.results_list.setItemDelegate(self._search_result_delegate)
        self._search_result_delegate.pathClicked.connect(self._open_file_in_explorer_from_string)
        self.results_count_label = QLabel("0 match")
        self.threshold_display_label = QLabel()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.status_label = QLabel("Prêt")
        self.face_count_label = QLabel("Visages détectés: 0")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(240)
        self.progress_bar.setVisible(False)
        self._indexing_thread: QThread | None = None
        self._indexing_worker: IndexingWorker | None = None
        self._last_search_matches: list[object] = []
        self._last_search_person: str = ""
        self._settings_dialog: SettingsDialog | None = None

        self._build_ui()
        self._refresh_threshold_display()
        self._load_people_from_storage()
        self.person_combo.setCurrentIndex(-1)
        self.person_combo.setEditText("")
        self._refresh_engine_info()
        self._open_settings_if_required()

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Vertical)
        root_layout.addWidget(splitter)

        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        splitter.addWidget(top_widget)

        button_row = QHBoxLayout()
        self.settings_button = QPushButton("Paramètres…")
        self.settings_button.clicked.connect(self._open_settings_dialog)
        self.index_button = QPushButton("Indexer photos + références")
        self.index_button.clicked.connect(self._run_indexing)
        self.rerun_incremental_button = QPushButton("Rerun incrémental")
        self.rerun_incremental_button.clicked.connect(self._run_incremental_indexing)
        self.cancel_index_button = QPushButton("Annuler l'indexation")
        self.cancel_index_button.clicked.connect(self._cancel_indexing)
        self.cancel_index_button.setEnabled(False)
        self.version_button = QPushButton("Tester la version moteur")
        self.version_button.clicked.connect(self._show_engine_version_details)
        button_row.addWidget(self.settings_button)
        button_row.addWidget(self.index_button)
        button_row.addWidget(self.rerun_incremental_button)
        button_row.addWidget(self.cancel_index_button)
        button_row.addWidget(self.version_button)
        top_layout.addLayout(button_row)

        menu_tools = self.menuBar().addMenu("Outils")
        self.settings_menu_action = menu_tools.addAction("Paramètres…", self._open_settings_dialog)

        search_layout = QVBoxLayout()
        top_layout.addLayout(search_layout)

        search_controls = QHBoxLayout()
        search_controls.addWidget(QLabel("Personne"))
        search_controls.addWidget(self.person_combo)
        self.search_button = QPushButton("Rechercher")
        self.search_button.clicked.connect(self._run_search)
        self.export_photos_button = QPushButton("Exporter photos")
        self.export_photos_button.clicked.connect(self._export_search_photos)
        self.validate_button = QPushButton("Valider")
        self.validate_button.clicked.connect(lambda: self._apply_validation("confirmed"))
        self.reject_button = QPushButton("Dévalider")
        self.reject_button.clicked.connect(lambda: self._apply_validation("rejected"))
        self.manual_button = QPushButton("Marquer manuel")
        self.manual_button.clicked.connect(lambda: self._apply_validation("manual"))
        search_controls.addWidget(self.search_button)
        search_controls.addWidget(self.export_photos_button)
        search_controls.addWidget(self.validate_button)
        search_controls.addWidget(self.reject_button)
        search_controls.addWidget(self.manual_button)
        search_layout.addLayout(search_controls)
        search_layout.addWidget(self.threshold_display_label)
        search_layout.addWidget(self.results_count_label)
        search_layout.addWidget(self.results_list)

        self.results_list.selectionModel().selectionChanged.connect(self._show_selected_result_details)

        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        splitter.addWidget(bottom_widget)
        bottom_layout.addWidget(QLabel("Journal"))
        bottom_layout.addWidget(self.log_output)
        splitter.setSizes([620, 240])

        status_bar = self.statusBar()
        status_bar.addWidget(self.status_label, 1)
        status_bar.addPermanentWidget(self.face_count_label)
        status_bar.addPermanentWidget(self.progress_bar)

    def _refresh_threshold_display(self) -> None:
        preset_labels = {
            "rapide": "Rapide",
            "equilibre": "Équilibré",
            "precis": "Précis",
        }
        preset_name = self.settings.indexing_preset_name()
        preset_label = preset_labels.get(preset_name, "Précis")
        self.threshold_display_label.setText(
            f"Seuil courant de similarité: {self.settings.similarity_threshold:.2f} | "
            f"Mode indexation: {preset_label} (modifier via Paramètres…)"
        )

    def _open_settings_if_required(self) -> None:
        if self.settings.storage_directory:
            return
        QTimer.singleShot(0, self._open_settings_dialog)

    def _open_settings_dialog(self) -> None:
        if self._indexing_thread is not None:
            QMessageBox.information(self, "Indexation en cours", "Modifie les paramètres après la fin de l'indexation.")
            return

        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(self.settings, self)
            self._settings_dialog.settingsSaved.connect(self._on_settings_saved)
        else:
            self._settings_dialog.update_settings(self.settings)

        self._settings_dialog.exec()

    def _on_settings_saved(self, saved_settings: AppSettings) -> None:
        previous_storage = self.settings.storage_directory
        previous_reference = self.settings.reference_directory
        self.settings = saved_settings
        self._refresh_threshold_display()
        self._log("Paramètres mis à jour via la boîte de dialogue.")

        if (
            self.settings.storage_directory != previous_storage
            or self.settings.reference_directory != previous_reference
        ):
            self._load_people_from_storage()

    def _persist_window_and_selection_state(self) -> None:
        self.settings.window.width = self.width()
        self.settings.window.height = self.height()
        self.settings.last_selected_person = self.person_combo.currentText().strip()
        self.settings.save()

    def _load_people_from_storage(self) -> None:
        self.person_combo.clear()
        self._person_completer_model.clear()
        repository = self._open_repository_if_possible()
        if repository is None:
            return
        for row in repository.list_people():
            name = str(row["display_name"])
            self.person_combo.addItem(name)
            self._person_completer_model.appendRow(QStandardItem(name))
        if self.settings.last_selected_person:
            index = self.person_combo.findText(self.settings.last_selected_person)
            if index >= 0:
                self.person_combo.setCurrentIndex(index)

    def _refresh_engine_info(self) -> None:
        info = detect_engine_version()
        providers = ", ".join(info.providers)
        status = "disponible" if info.available else "indisponible"
        self._log(
            f"App v{__version__} | {info.engine_name} {status} | installé: insightface={info.package_version or 'non installé'}, "
            f"onnxruntime={info.runtime_version or 'non installé'} | compatible: insightface={info.latest_compatible_package_version or 'inconnue'}, "
            f"onnxruntime={info.latest_compatible_runtime_version or 'inconnue'} | PyPI: insightface={info.latest_package_version or 'inconnue'}, "
            f"onnxruntime={info.latest_runtime_version or 'inconnue'} | providers={providers}"
        )
        self._set_status(f"App {__version__} / {info.engine_name} {info.package_version}")

    def _show_engine_version_details(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.version_button.setEnabled(False)
        QApplication.processEvents()
        try:
            info = detect_engine_version()
            providers = ", ".join(info.providers)

            installed_pkg = info.package_version or "non installé"
            installed_runtime = info.runtime_version or "non installé"
            available_pkg = info.latest_package_version or "inconnue"
            available_runtime = info.latest_runtime_version or "inconnue"
            compatible_pkg = info.latest_compatible_package_version or "inconnue"
            compatible_runtime = info.latest_compatible_runtime_version or "inconnue"

            details = (
                f"Moteur: {info.engine_name}\n"
                f"Installé:\n"
                f"- insightface: {installed_pkg}\n"
                f"- onnxruntime: {installed_runtime}\n\n"
                f"Dernière version COMPATIBLE (ton Python/OS):\n"
                f"- insightface: {compatible_pkg}\n"
                f"- onnxruntime: {compatible_runtime}\n\n"
                f"Dernière version sur PyPI (globale):\n"
                f"- insightface: {available_pkg}\n"
                f"- onnxruntime: {available_runtime}\n\n"
                f"Liens:\n"
                f"- insightface: {info.package_url}\n"
                f"- onnxruntime: {info.runtime_url}\n\n"
                f"Procédure d'installation / mise à jour (venv):\n"
                f"1) Activer le venv: .\\.venv\\Scripts\\Activate.ps1\n"
                f"2) Installer/mettre à jour: {info.install_command}\n"
                f"3) Redémarrer l'application\n\n"
                f"Note: si la version PyPI est plus récente que la version compatible, pip peut afficher 'already satisfied'.\n\n"
                f"Providers: {providers}"
            )

            status = "disponible" if info.available else "indisponible"
            self._log(
                f"App v{__version__} | {info.engine_name} {status} | installé: insightface={info.package_version or 'non installé'}, "
                f"onnxruntime={info.runtime_version or 'non installé'} | compatible: insightface={info.latest_compatible_package_version or 'inconnue'}, "
                f"onnxruntime={info.latest_compatible_runtime_version or 'inconnue'} | PyPI: insightface={info.latest_package_version or 'inconnue'}, "
                f"onnxruntime={info.latest_runtime_version or 'inconnue'} | providers={providers}"
            )
            self._set_status(f"App {__version__} / {info.engine_name} {info.package_version}")

            self._log("===== Test version moteur =====")
            for line in details.splitlines():
                self._log(line)
            QMessageBox.information(self, "Version du moteur", details)
        finally:
            QApplication.restoreOverrideCursor()
            self.version_button.setEnabled(True)

    def _run_indexing(self) -> None:
        if self._indexing_thread is not None:
            self._log("Une indexation est déjà en cours.")
            self._set_status("Indexation déjà en cours…")
            return

        storage_directory = self.settings.storage_directory.strip()
        if not storage_directory:
            QMessageBox.warning(self, "Stockage manquant", "Choisissez un dossier de stockage pour l'index SQLite.")
            self._open_settings_dialog()
            return

        photo_roots = [Path(path) for path in self.settings.photo_directories if path]
        reference_dir = Path(self.settings.reference_directory) if self.settings.reference_directory else None
        if not photo_roots and reference_dir is None:
            QMessageBox.information(
                self,
                "Aucune source à traiter",
                "Ajoutez au moins un répertoire photo ou un répertoire de références avant de lancer l'indexation.",
            )
            return

        self._indexing_thread = QThread(self)
        self._indexing_worker = IndexingWorker(
            photo_directories=[str(path) for path in photo_roots],
            reference_directory=str(reference_dir) if reference_dir is not None else "",
            storage_directory=storage_directory,
            model_name=self.settings.indexing_model_name,
            detection_size=self.settings.indexing_detection_size,
        )
        self._indexing_worker.moveToThread(self._indexing_thread)
        self._indexing_thread.started.connect(self._indexing_worker.run)
        self._indexing_worker.log_message.connect(self._log)
        self._indexing_worker.status_message.connect(self._set_status)
        self._indexing_worker.progress_changed.connect(self._update_progress)
        self._indexing_worker.face_count_changed.connect(self._update_face_count)
        self._indexing_worker.completed.connect(self._on_indexing_completed)
        self._indexing_worker.cancelled.connect(self._on_indexing_cancelled)
        self._indexing_worker.failed.connect(self._on_indexing_failed)
        self._indexing_worker.completed.connect(self._indexing_thread.quit)
        self._indexing_worker.cancelled.connect(self._indexing_thread.quit)
        self._indexing_worker.failed.connect(self._indexing_thread.quit)
        self._indexing_thread.finished.connect(self._cleanup_indexing_thread)

        self._set_controls_enabled(False)
        self.cancel_index_button.setEnabled(True)
        self._set_status("Préparation de l'indexation…")
        self._update_face_count(0)
        self._update_progress(0, 0)
        self._log("Indexation lancée en arrière-plan.")
        self._indexing_thread.start()

    def _run_incremental_indexing(self) -> None:
        self._log("Rerun incrémental demandé (seules les nouveautés/modifications seront traitées).")
        self._run_indexing()

    def _cancel_indexing(self) -> None:
        if self._indexing_worker is None:
            return
        self.cancel_index_button.setEnabled(False)
        self._log("Demande d'annulation envoyée.")
        self._set_status("Annulation demandée…")
        self._indexing_worker.cancel()

    def _on_indexing_completed(self, summary: object) -> None:
        self._load_people_from_storage()
        self._set_controls_enabled(True)
        self.cancel_index_button.setEnabled(False)
        self._reset_progress()
        self._set_status("Indexation terminée.")
        self._log(self._format_summary(summary, cancelled=False))
        self._export_indexing_summary(summary)

    def _on_indexing_cancelled(self, summary: object) -> None:
        self._load_people_from_storage()
        self._set_controls_enabled(True)
        self.cancel_index_button.setEnabled(False)
        self._reset_progress()
        self._set_status("Indexation annulée.")
        self._log(self._format_summary(summary, cancelled=True))
        self._export_indexing_summary(summary)

    def _on_indexing_failed(self, error_message: str) -> None:
        self._set_controls_enabled(True)
        self.cancel_index_button.setEnabled(False)
        self._reset_progress()
        self._set_status("Indexation en erreur.")
        self._log(f"Erreur d'indexation: {error_message}")
        QMessageBox.critical(self, "Indexation impossible", error_message)

    def _cleanup_indexing_thread(self) -> None:
        if self._indexing_worker is not None:
            self._indexing_worker.deleteLater()
            self._indexing_worker = None
        if self._indexing_thread is not None:
            self._indexing_thread.deleteLater()
            self._indexing_thread = None

    def _set_controls_enabled(self, enabled: bool) -> None:
        widgets = (
            self.person_combo,
            self.settings_button,
            self.index_button,
            self.rerun_incremental_button,
            self.version_button,
            self.search_button,
            self.export_photos_button,
            self.validate_button,
            self.reject_button,
            self.manual_button,
            self.results_list,
        )
        for widget in widgets:
            widget.setEnabled(enabled)
        self.settings_menu_action.setEnabled(enabled)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _update_progress(self, current: int, total: int) -> None:
        self.progress_bar.setVisible(True)
        if total <= 0:
            self.progress_bar.setRange(0, 0)
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def _reset_progress(self) -> None:
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

    def _update_face_count(self, total_faces: int) -> None:
        self.face_count_label.setText(f"Visages détectés: {total_faces}")

    def _format_summary(self, summary: object, cancelled: bool) -> str:
        if not isinstance(summary, dict):
            return f"Indexation {'annulée' if cancelled else 'terminée'}: {summary}"

        photos = summary.get("photos") if isinstance(summary.get("photos"), dict) else {}
        refs_wrapper = summary.get("references") if isinstance(summary.get("references"), dict) else {}
        refs_process = refs_wrapper.get("process") if isinstance(refs_wrapper.get("process"), dict) else {}
        elapsed = summary.get("elapsed_seconds", "n/a")
        total_faces = summary.get("detected_faces_total", 0)

        return (
            f"Indexation {'annulée' if cancelled else 'terminée'} | durée={elapsed}s | "
            f"photos traitées={photos.get('processed', 0)} | photos en échec={photos.get('failed', 0)} | "
            f"références traitées={refs_process.get('processed', 0)} | références en échec={refs_process.get('failed', 0)} | "
            f"visages détectés={total_faces}"
        )

    def _export_indexing_summary(self, summary: object) -> None:
        storage_directory = self.settings.storage_directory.strip()
        if not storage_directory:
            return
        if not isinstance(summary, dict):
            return
        summary_path = Path(storage_directory) / "last-indexing-summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log(f"Résumé d'indexation exporté: {summary_path}")

    def _run_search(self) -> None:
        repository = self._open_repository_if_possible()
        if repository is None:
            QMessageBox.warning(self, "Stockage manquant", "Choisissez d'abord un emplacement de stockage.")
            return
        person_name = self.person_combo.currentText().strip()
        if not person_name:
            QMessageBox.information(self, "Aucune personne", "Aucune personne de référence n'est encore indexée.")
            return

        matcher = FaceMatcher(repository)
        result = matcher.search_person(person_name, self.settings.similarity_threshold)
        self._last_search_matches = list(result.matches)
        self._last_search_person = person_name
        self._search_results_model.set_matches(result.matches)
        match_count = len(result.matches)
        self.results_count_label.setText(f"{match_count} match{'es' if match_count > 1 else ''}")
        self._log(f"Recherche '{person_name}': {len(result.matches)} résultat(s) au seuil {result.threshold:.2f}")
        self._set_status(f"Recherche terminée pour {person_name}: {len(result.matches)} résultat(s).")

    def _open_file_in_explorer_from_string(self, file_path: str) -> None:
        self._open_file_in_explorer(Path(file_path))

    def _open_file_in_explorer(self, file_path: Path) -> None:
        path = Path(file_path)
        if not path.exists():
            QMessageBox.warning(self, "Fichier introuvable", f"Le fichier n'existe plus:\n{path}")
            return
        try:
            if path.anchor:
                subprocess.Popen(["explorer", f"/select,{path}"], shell=False)
            else:
                subprocess.Popen(["explorer", str(path.parent)], shell=False)
        except OSError as error:
            QMessageBox.warning(self, "Ouverture impossible", f"Impossible d'ouvrir l'explorateur:\n{error}")

    def _export_search_photos(self) -> None:
        if not self._last_search_matches:
            QMessageBox.information(self, "Aucun résultat", "Lancez une recherche avant d'exporter les photos.")
            return

        destination_root = QFileDialog.getExistingDirectory(self, "Choisir le répertoire d'export")
        if not destination_root:
            return

        person_name = self._last_search_person or self.person_combo.currentText().strip() or "personne"
        person_dir = Path(destination_root) / self._sanitize_export_folder_name(person_name)
        person_dir.mkdir(parents=True, exist_ok=True)

        unique_sources: list[Path] = []
        seen_paths: set[str] = set()
        for match in self._last_search_matches:
            source = Path(match.photo_path)
            source_key = str(source.resolve()) if source.exists() else str(source)
            if source_key in seen_paths:
                continue
            seen_paths.add(source_key)
            unique_sources.append(source)

        copied = 0
        failed = 0
        for source in unique_sources:
            if not source.exists():
                failed += 1
                continue
            destination_file = self._next_available_path(person_dir, source.name)
            try:
                shutil.copy2(source, destination_file)
                copied += 1
            except OSError:
                failed += 1

        self._log(f"Export photos '{person_name}': {copied} copiée(s), {failed} échec(s) -> {person_dir}")
        self._set_status(f"Export terminé: {copied} fichier(s) copié(s).")
        QMessageBox.information(
            self,
            "Export terminé",
            f"Dossier: {person_dir}\nFichiers copiés: {copied}\nÉchecs: {failed}",
        )

    def _sanitize_export_folder_name(self, name: str) -> str:
        invalid_chars = '<>:"/\\|?*'
        cleaned = "".join("_" if char in invalid_chars else char for char in name.strip())
        return cleaned or "personne"

    def _next_available_path(self, directory: Path, filename: str) -> Path:
        candidate = directory / filename
        if not candidate.exists():
            return candidate
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        index = 1
        while True:
            candidate = directory / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def _apply_validation(self, state: str) -> None:
        repository = self._open_repository_if_possible()
        selected_index = self.results_list.currentIndex()
        if repository is None or not selected_index.isValid():
            return
        match = self._search_results_model.match_at(selected_index)
        if match is None:
            return
        repository.set_validation(match.person_id, match.photo_face_id, state)
        repository.commit()
        self._log(f"Validation mise à jour: {match.person_name} / face #{match.photo_face_id} -> {state}")

    def _show_selected_result_details(self, *_args) -> None:
        selected_index = self.results_list.currentIndex()
        if not selected_index.isValid():
            return
        match = self._search_results_model.match_at(selected_index)
        if match is None:
            return
        self._log(
            f"Sélection: {match.person_name} | photo={match.photo_path.name} | bbox={match.bbox} | score={match.similarity:.3f} | source={match.source}"
        )

    def _ensure_repository(self) -> FaceIndexRepository | None:
        storage_directory = self.settings.storage_directory.strip()
        if not storage_directory:
            QMessageBox.warning(self, "Stockage manquant", "Choisissez un dossier de stockage pour l'index SQLite.")
            return None
        db_path = initialize_database(resolve_database_path(storage_directory))
        repository = FaceIndexRepository(connect(db_path))
        return repository

    def _open_repository_if_possible(self) -> FaceIndexRepository | None:
        storage_directory = self.settings.storage_directory.strip()
        if not storage_directory:
            return None
        db_path = resolve_database_path(storage_directory)
        if not db_path.exists():
            return None
        return FaceIndexRepository(connect(db_path))

    def _log(self, message: str) -> None:
        safe_message = str(message).encode("utf-8", errors="replace").decode("utf-8")
        self.log_output.append(safe_message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._indexing_thread is not None:
            QMessageBox.information(self, "Indexation en cours", "Patientez jusqu'à la fin de l'indexation avant de fermer l'application.")
            event.ignore()
            return
        self._persist_window_and_selection_state()
        super().closeEvent(event)
