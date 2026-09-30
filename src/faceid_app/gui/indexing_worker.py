from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from faceid_app.core.face_engine import InsightFaceEngine, detect_engine_version
from faceid_app.core.photo_indexer import PhotoIndexer
from faceid_app.core.reference_indexer import ReferenceIndexer
from faceid_app.storage.database import connect, initialize_database, resolve_database_path
from faceid_app.storage.repositories import FaceIndexRepository


class IndexingWorker(QObject):
    log_message = Signal(str)
    status_message = Signal(str)
    progress_changed = Signal(int, int)
    face_count_changed = Signal(int)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal(object)

    def __init__(
        self,
        photo_directories: list[str],
        reference_directory: str,
        storage_directory: str,
        model_name: str,
        detection_size: int,
    ) -> None:
        super().__init__()
        self.photo_directories = photo_directories
        self.reference_directory = reference_directory
        self.storage_directory = storage_directory
        self.model_name = model_name
        self.detection_size = detection_size
        self._cancel_requested = False
        self._detected_faces_total = 0

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True
        self.status_message.emit("Annulation demandée… fin du fichier en cours puis arrêt.")

    @Slot()
    def run(self) -> None:
        connection = None
        started = time.perf_counter()
        try:
            self.status_message.emit("Initialisation du stockage et du moteur…")
            self.progress_changed.emit(0, 0)
            self.face_count_changed.emit(0)

            db_path = initialize_database(resolve_database_path(self.storage_directory))
            connection = connect(db_path)
            repository = FaceIndexRepository(connection)
            safe_detection_size = max(320, int(self.detection_size))
            engine = InsightFaceEngine(
                model_name=self.model_name,
                det_size=(safe_detection_size, safe_detection_size),
            )
            self.log_message.emit(
                "Moteur indexation: "
                f"modèle={engine.model_name}, det_size={engine.det_size[0]}x{engine.det_size[1]}, "
                f"providers={', '.join(engine.providers)}"
            )
            photo_indexer = PhotoIndexer(repository, engine)
            reference_indexer = ReferenceIndexer(repository, engine)

            photo_roots = [Path(path) for path in self.photo_directories if path]
            reference_dir = Path(self.reference_directory) if self.reference_directory else None

            self.status_message.emit("Inventaire des photos…")
            photo_sync = photo_indexer.synchronize_inventory(photo_roots)
            self.log_message.emit(
                f"Inventaire photos: découvertes={photo_sync.discovered}, nouvelles/modifiées={photo_sync.new_or_changed}, supprimées={photo_sync.deleted}"
            )

            pending_in_scope = photo_indexer.count_pending(photo_roots)
            self.log_message.emit(
                f"Photos en attente dans le périmètre courant avant traitement: {pending_in_scope}"
            )

            if photo_sync.new_or_changed:
                self.status_message.emit(f"Traitement des photos ({photo_sync.new_or_changed} à analyser)…")
            else:
                self.status_message.emit("Aucune nouvelle photo à traiter.")
                self.progress_changed.emit(0, 1)
            photo_process = photo_indexer.process_pending(
                root_directories=photo_roots,
                progress_callback=self._emit_photo_progress,
                should_stop=self._should_stop,
            )
            self._detected_faces_total += photo_process.detected_faces
            self.face_count_changed.emit(self._detected_faces_total)
            self.log_message.emit(
                f"Traitement photos: traitées={photo_process.processed}, visages détectés={photo_process.detected_faces}, échecs={photo_process.failed}"
            )

            if photo_process.cancelled:
                summary = {
                    "cancelled": True,
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "photos": {
                        "discovered": photo_sync.discovered,
                        "new_or_changed": photo_sync.new_or_changed,
                        "deleted": photo_sync.deleted,
                        "processed": photo_process.processed,
                        "failed": photo_process.failed,
                        "detected_faces": photo_process.detected_faces,
                    },
                    "references": None,
                    "detected_faces_total": self._detected_faces_total,
                }
                self.status_message.emit("Indexation annulée pendant le traitement des photos.")
                self.cancelled.emit(summary)
                return

            reference_sync_dict: dict[str, int] | None = None
            reference_process_dict: dict[str, int | bool] | None = None
            if reference_dir is not None:
                self.status_message.emit("Inventaire des références…")
                self.progress_changed.emit(0, 0)
                reference_sync = reference_indexer.synchronize_inventory(reference_dir)
                reference_sync_dict = {
                    "discovered": reference_sync.discovered,
                    "new_or_changed": reference_sync.new_or_changed,
                    "deleted": reference_sync.deleted,
                }
                self.log_message.emit(
                    f"Inventaire références: découvertes={reference_sync.discovered}, nouvelles/modifiées={reference_sync.new_or_changed}, supprimées={reference_sync.deleted}"
                )

                if reference_sync.new_or_changed:
                    self.status_message.emit(f"Traitement des références ({reference_sync.new_or_changed} à analyser)…")
                else:
                    self.status_message.emit("Aucune nouvelle référence à traiter.")
                    self.progress_changed.emit(0, 1)
                reference_process = reference_indexer.process_pending(
                    progress_callback=self._emit_reference_progress,
                    should_stop=self._should_stop,
                )
                self._detected_faces_total += reference_process.detected_faces
                self.face_count_changed.emit(self._detected_faces_total)
                reference_process_dict = {
                    "processed": reference_process.processed,
                    "failed": reference_process.failed,
                    "detected_faces": reference_process.detected_faces,
                    "cancelled": reference_process.cancelled,
                }
                self.log_message.emit(
                    f"Traitement références: traitées={reference_process.processed}, visages détectés={reference_process.detected_faces}, échecs={reference_process.failed}"
                )

                if reference_process.cancelled:
                    summary = {
                        "cancelled": True,
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                        "photos": {
                            "discovered": photo_sync.discovered,
                            "new_or_changed": photo_sync.new_or_changed,
                            "deleted": photo_sync.deleted,
                            "processed": photo_process.processed,
                            "failed": photo_process.failed,
                            "detected_faces": photo_process.detected_faces,
                        },
                        "references": {
                            "sync": reference_sync_dict,
                            "process": reference_process_dict,
                        },
                        "detected_faces_total": self._detected_faces_total,
                    }
                    self.status_message.emit("Indexation annulée pendant le traitement des références.")
                    self.cancelled.emit(summary)
                    return

            version_info = detect_engine_version(check_updates=False)
            repository.save_engine_metadata(
                {
                    "engine_name": version_info.engine_name,
                    "engine_version": version_info.package_version or "",
                    "runtime_version": version_info.runtime_version or "",
                }
            )
            repository.commit()

            summary = {
                "cancelled": False,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "photos": {
                    "discovered": photo_sync.discovered,
                    "new_or_changed": photo_sync.new_or_changed,
                    "deleted": photo_sync.deleted,
                    "processed": photo_process.processed,
                    "failed": photo_process.failed,
                    "detected_faces": photo_process.detected_faces,
                },
                "references": {
                    "sync": reference_sync_dict,
                    "process": reference_process_dict,
                },
                "detected_faces_total": self._detected_faces_total,
            }
            self.status_message.emit("Indexation terminée.")
            self.progress_changed.emit(1, 1)
            self.completed.emit(summary)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if connection is not None:
                connection.close()

    def _emit_photo_progress(self, current: int, total: int, path: Path, detected_faces_phase: int) -> None:
        total_faces = self._detected_faces_total + detected_faces_phase
        self.face_count_changed.emit(total_faces)
        self.status_message.emit(f"Photo {current}/{total} traitée: {path.name} | visages détectés: {total_faces}")
        self.progress_changed.emit(current, total)

    def _emit_reference_progress(self, current: int, total: int, path: Path, detected_faces_phase: int) -> None:
        total_faces = self._detected_faces_total + detected_faces_phase
        self.face_count_changed.emit(total_faces)
        self.status_message.emit(f"Référence {current}/{total} traitée: {path.name} | visages détectés: {total_faces}")
        self.progress_changed.emit(current, total)

    def _should_stop(self) -> bool:
        return self._cancel_requested