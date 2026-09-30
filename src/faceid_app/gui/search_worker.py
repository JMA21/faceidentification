from __future__ import annotations

from contextlib import closing
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from faceid_app.core.matcher import FaceMatcher
from faceid_app.storage.database import connect
from faceid_app.storage.repositories import FaceIndexRepository


class SearchWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, database_path: Path, person_name: str, threshold: float) -> None:
        super().__init__()
        self.database_path = database_path
        self.person_name = person_name
        self.threshold = threshold

    @Slot()
    def run(self) -> None:
        try:
            # SQLite connections must be opened and closed in their owning thread.
            with closing(connect(self.database_path)) as connection:
                result = FaceMatcher(FaceIndexRepository(connection)).search_person(self.person_name, self.threshold)
            self.completed.emit(result)
        except Exception as error:
            self.failed.emit(str(error))