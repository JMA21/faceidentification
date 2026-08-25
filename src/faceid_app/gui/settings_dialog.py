from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QDialog,
    QSlider,
    QVBoxLayout,
)

from faceid_app.config.settings import AppSettings

PERFORMANCE_PRESET_OPTIONS: list[tuple[str, str]] = [
    ("rapide", "Rapide (buffalo_s, 416x416)"),
    ("equilibre", "Équilibré (buffalo_l, 512x512)"),
    ("precis", "Précis (buffalo_l, 640x640)"),
]


class SettingsDialog(QDialog):
    settingsSaved = Signal(object)

    def __init__(self, settings: AppSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Paramètres")
        self.setModal(True)
        self.resize(760, 520)

        self._settings = settings

        self.photo_dirs_list = QListWidget()
        self.reference_dir_edit = QLineEdit()
        self.storage_dir_edit = QLineEdit()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_value_label = QLabel()
        self.performance_mode_combo = QComboBox()

        self._build_ui()
        self._load_settings_into_form()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)

        help_text = QLabel(
            "Configure les sources et le stockage. Le stockage est obligatoire. "
            "Il faut au moins un dossier photos ou un dossier références."
        )
        help_text.setWordWrap(True)
        root_layout.addWidget(help_text)

        config_box = QGroupBox("Paramètres")
        config_layout = QGridLayout(config_box)
        root_layout.addWidget(config_box)

        config_layout.addWidget(QLabel("Répertoires photos"), 0, 0)
        config_layout.addWidget(self.photo_dirs_list, 1, 0, 3, 2)
        self.add_photo_dir_button = QPushButton("Ajouter…")
        self.add_photo_dir_button.clicked.connect(self._add_photo_directory)
        self.remove_photo_dir_button = QPushButton("Retirer")
        self.remove_photo_dir_button.clicked.connect(self._remove_selected_photo_directory)
        config_layout.addWidget(self.add_photo_dir_button, 1, 2)
        config_layout.addWidget(self.remove_photo_dir_button, 2, 2)

        config_layout.addWidget(QLabel("Répertoire des références"), 4, 0)
        config_layout.addWidget(self.reference_dir_edit, 4, 1)
        self.browse_reference_button = QPushButton("Parcourir…")
        self.browse_reference_button.clicked.connect(self._browse_reference_directory)
        config_layout.addWidget(self.browse_reference_button, 4, 2)

        config_layout.addWidget(QLabel("Emplacement du stockage"), 5, 0)
        config_layout.addWidget(self.storage_dir_edit, 5, 1)
        self.browse_storage_button = QPushButton("Parcourir…")
        self.browse_storage_button.clicked.connect(self._browse_storage_directory)
        config_layout.addWidget(self.browse_storage_button, 5, 2)

        config_layout.addWidget(QLabel("Finesse de ressemblance"), 6, 0)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.valueChanged.connect(self._update_threshold_label)
        config_layout.addWidget(self.threshold_slider, 6, 1)
        config_layout.addWidget(self.threshold_value_label, 6, 2)

        config_layout.addWidget(QLabel("Mode performance indexation"), 7, 0)
        for key, label in PERFORMANCE_PRESET_OPTIONS:
            self.performance_mode_combo.addItem(label, userData=key)
        config_layout.addWidget(self.performance_mode_combo, 7, 1, 1, 2)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.apply_button = QPushButton("Appliquer")
        self.apply_button.clicked.connect(self._apply)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._accept)
        self.cancel_button = QPushButton("Annuler")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.ok_button)
        button_row.addWidget(self.cancel_button)
        root_layout.addLayout(button_row)

    def _load_settings_into_form(self) -> None:
        self.photo_dirs_list.clear()
        for directory in self._settings.photo_directories:
            self.photo_dirs_list.addItem(directory)
        self.reference_dir_edit.setText(self._settings.reference_directory)
        self.storage_dir_edit.setText(self._settings.storage_directory)
        self.threshold_slider.setValue(max(1, min(100, int(round(self._settings.similarity_threshold * 100)))))
        preset_name = self._settings.indexing_preset_name()
        index = self.performance_mode_combo.findData(preset_name)
        if index >= 0:
            self.performance_mode_combo.setCurrentIndex(index)
        self._update_threshold_label()

    def _collect_settings(self) -> AppSettings:
        settings = AppSettings(
            photo_directories=[self.photo_dirs_list.item(index).text() for index in range(self.photo_dirs_list.count())],
            reference_directory=self.reference_dir_edit.text().strip(),
            storage_directory=self.storage_dir_edit.text().strip(),
            similarity_threshold=self.threshold_slider.value() / 100.0,
            indexing_model_name=self._settings.indexing_model_name,
            indexing_detection_size=self._settings.indexing_detection_size,
            window=self._settings.window,
            last_selected_person=self._settings.last_selected_person,
        )
        selected_preset = str(self.performance_mode_combo.currentData() or "precis")
        settings.apply_indexing_preset(selected_preset)
        return settings

    def _validate_settings(self, settings: AppSettings) -> str | None:
        if not settings.storage_directory:
            return "Choisissez un dossier de stockage pour l'index SQLite."
        if not settings.photo_directories and not settings.reference_directory:
            return "Ajoutez au moins un répertoire photo ou un répertoire de références."
        if not (0.01 <= settings.similarity_threshold <= 1.0):
            return "Le seuil de similarité doit être compris entre 0.01 et 1.00."
        return None

    def _apply(self) -> bool:
        new_settings = self._collect_settings()
        validation_error = self._validate_settings(new_settings)
        if validation_error is not None:
            QMessageBox.warning(self, "Paramètres invalides", validation_error)
            return False

        overlaps = new_settings.overlapping_photo_directories()
        if overlaps:
            sample = "\n".join(f"- {parent}\n  ⮑ {child}" for parent, child in overlaps[:3])
            suffix = "\n..." if len(overlaps) > 3 else ""
            QMessageBox.information(
                self,
                "Répertoires imbriqués détectés",
                "Des répertoires photos se recouvrent. L'indexation appliquera une déduplication automatique.\n\n"
                f"Exemples:\n{sample}{suffix}",
            )

        settings_path = new_settings.save()
        self._settings = new_settings
        self.settingsSaved.emit(new_settings)
        QMessageBox.information(self, "Paramètres enregistrés", f"Paramètres enregistrés dans {settings_path}")
        return True

    def _accept(self) -> None:
        if self._apply():
            self.accept()

    def _add_photo_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choisir un répertoire de photos")
        if directory:
            existing = [self.photo_dirs_list.item(index).text() for index in range(self.photo_dirs_list.count())]
            if directory not in existing:
                self.photo_dirs_list.addItem(directory)

    def _remove_selected_photo_directory(self) -> None:
        row = self.photo_dirs_list.currentRow()
        if row >= 0:
            self.photo_dirs_list.takeItem(row)

    def _browse_reference_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choisir le répertoire des références")
        if directory:
            self.reference_dir_edit.setText(directory)

    def _browse_storage_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choisir le répertoire de stockage")
        if directory:
            self.storage_dir_edit.setText(directory)

    def _update_threshold_label(self) -> None:
        self.threshold_value_label.setText(f"{self.threshold_slider.value() / 100.0:.2f}")

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings
        self._load_settings_into_form()
