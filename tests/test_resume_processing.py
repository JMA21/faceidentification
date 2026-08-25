from pathlib import Path

import numpy as np

from faceid_app.core.photo_indexer import PhotoIndexer
from faceid_app.models import FaceDetection
from faceid_app.storage.database import connect, initialize_database
from faceid_app.storage.repositories import FaceIndexRepository


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def extract_faces(self, image_path: Path):
        self.calls.append(image_path)
        return [FaceDetection(bbox=(0, 0, 10, 10), embedding=np.array([1.0, 0.0], dtype=np.float32), score=0.99)]


def test_resume_does_not_reprocess_completed_photos(tmp_path: Path) -> None:
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "a.jpg").write_bytes(b"a")
    (photos_dir / "b.jpg").write_bytes(b"b")

    db_path = initialize_database(tmp_path / "index.sqlite3")
    repository = FaceIndexRepository(connect(db_path))
    engine = FakeEngine()
    indexer = PhotoIndexer(repository, engine)

    sync_summary = indexer.synchronize_inventory([photos_dir])
    assert sync_summary.discovered == 2
    assert sync_summary.new_or_changed == 2

    first_run = indexer.process_pending(limit=1)
    assert first_run.processed == 1
    assert len(engine.calls) == 1

    second_run = indexer.process_pending()
    assert second_run.processed == 1
    assert len(engine.calls) == 2

    third_run = indexer.process_pending()
    assert third_run.processed == 0
    assert len(engine.calls) == 2


def test_incremental_sync_deindexes_deleted_photos(tmp_path: Path) -> None:
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    first = photos_dir / "a.jpg"
    second = photos_dir / "b.jpg"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    db_path = initialize_database(tmp_path / "index.sqlite3")
    repository = FaceIndexRepository(connect(db_path))
    indexer = PhotoIndexer(repository, FakeEngine())

    first_sync = indexer.synchronize_inventory([photos_dir])
    assert first_sync.discovered == 2
    assert first_sync.deleted == 0

    second.unlink()
    second_sync = indexer.synchronize_inventory([photos_dir])
    assert second_sync.discovered == 1
    assert second_sync.deleted == 1


def test_incremental_sync_deduplicates_overlapping_roots(tmp_path: Path) -> None:
    photos_dir = tmp_path / "photos"
    nested_dir = photos_dir / "nested"
    nested_dir.mkdir(parents=True)
    (photos_dir / "a.jpg").write_bytes(b"a")
    (nested_dir / "b.jpg").write_bytes(b"b")
    (nested_dir / "c.jpg").write_bytes(b"c")

    db_path = initialize_database(tmp_path / "index.sqlite3")
    repository = FaceIndexRepository(connect(db_path))
    engine = FakeEngine()
    indexer = PhotoIndexer(repository, engine)

    first_sync = indexer.synchronize_inventory([photos_dir, nested_dir])
    assert first_sync.discovered == 3
    assert first_sync.new_or_changed == 3

    first_process = indexer.process_pending()
    assert first_process.processed == 3

    second_sync = indexer.synchronize_inventory([photos_dir, nested_dir])
    assert second_sync.discovered == 3
    assert second_sync.new_or_changed == 0
    assert second_sync.deleted == 0

    photos_count = repository.connection.execute("SELECT COUNT(*) AS total FROM photos").fetchone()["total"]
    faces_count = repository.connection.execute("SELECT COUNT(*) AS total FROM photo_faces").fetchone()["total"]
    assert int(photos_count) == 3
    assert int(faces_count) == 3


def test_process_pending_is_scoped_to_selected_roots(tmp_path: Path) -> None:
    photos_a = tmp_path / "photos_a"
    photos_b = tmp_path / "photos_b"
    photos_a.mkdir()
    photos_b.mkdir()
    (photos_a / "a.jpg").write_bytes(b"a")
    (photos_b / "b.jpg").write_bytes(b"b")

    db_path = initialize_database(tmp_path / "index.sqlite3")
    repository = FaceIndexRepository(connect(db_path))
    engine = FakeEngine()
    indexer = PhotoIndexer(repository, engine)

    # First run indexed with root A only and then cancelled before processing,
    # leaving pending rows for root A in storage.
    sync_a = indexer.synchronize_inventory([photos_a])
    assert sync_a.new_or_changed == 1

    # New run is configured on root B only.
    sync_b = indexer.synchronize_inventory([photos_b])
    assert sync_b.discovered == 1
    assert sync_b.new_or_changed == 1

    process_b = indexer.process_pending(root_directories=[photos_b])
    assert process_b.processed == 1
    assert [path.name for path in engine.calls] == ["b.jpg"]
