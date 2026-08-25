from pathlib import Path

import numpy as np

from faceid_app.core.reference_indexer import ReferenceIndexer
from faceid_app.models import FaceDetection
from faceid_app.storage.database import connect, initialize_database
from faceid_app.storage.repositories import FaceIndexRepository


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def extract_faces(self, image_path: Path):
        self.calls.append(image_path)
        return [FaceDetection(bbox=(0, 0, 10, 10), embedding=np.array([0.0, 1.0], dtype=np.float32), score=0.98)]


def test_only_new_reference_is_processed_on_second_pass(tmp_path: Path) -> None:
    ref_dir = tmp_path / "Ref"
    ref_dir.mkdir()
    (ref_dir / "Alice Example.png").write_bytes(b"1")

    db_path = initialize_database(tmp_path / "index.sqlite3")
    repository = FaceIndexRepository(connect(db_path))
    engine = FakeEngine()
    indexer = ReferenceIndexer(repository, engine)

    first_sync = indexer.synchronize_inventory(ref_dir)
    assert first_sync.discovered == 1
    assert first_sync.new_or_changed == 1
    first_process = indexer.process_pending()
    assert first_process.processed == 1
    assert len(engine.calls) == 1

    (ref_dir / "Bob Example.png").write_bytes(b"2")
    second_sync = indexer.synchronize_inventory(ref_dir)
    assert second_sync.discovered == 2
    assert second_sync.new_or_changed == 1
    second_process = indexer.process_pending()
    assert second_process.processed == 1
    assert len(engine.calls) == 2


def test_incremental_sync_deindexes_deleted_references(tmp_path: Path) -> None:
    ref_dir = tmp_path / "Ref"
    ref_dir.mkdir()
    first = ref_dir / "Alice Example.png"
    second = ref_dir / "Bob Example.png"
    first.write_bytes(b"1")
    second.write_bytes(b"2")

    db_path = initialize_database(tmp_path / "index.sqlite3")
    repository = FaceIndexRepository(connect(db_path))
    indexer = ReferenceIndexer(repository, FakeEngine())

    first_sync = indexer.synchronize_inventory(ref_dir)
    assert first_sync.discovered == 2
    assert first_sync.deleted == 0

    second.unlink()
    second_sync = indexer.synchronize_inventory(ref_dir)
    assert second_sync.discovered == 1
    assert second_sync.deleted == 1
