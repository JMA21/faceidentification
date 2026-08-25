from pathlib import Path

import numpy as np

from faceid_app.core.photo_indexer import PhotoIndexer
from faceid_app.core.reference_indexer import ReferenceIndexer
from faceid_app.models import FaceDetection
from faceid_app.storage.database import connect, initialize_database
from faceid_app.storage.repositories import FaceIndexRepository


class FakeEngine:
    def extract_faces(self, image_path: Path):
        return [FaceDetection(bbox=(0, 0, 10, 10), embedding=np.array([1.0, 0.0], dtype=np.float32), score=0.99)]


def test_photo_indexer_emits_progress_for_each_pending_file(tmp_path: Path) -> None:
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "a.jpg").write_bytes(b"a")
    (photos_dir / "b.jpg").write_bytes(b"b")

    db_path = initialize_database(tmp_path / "index.sqlite3")
    repository = FaceIndexRepository(connect(db_path))
    indexer = PhotoIndexer(repository, FakeEngine())
    indexer.synchronize_inventory([photos_dir])

    progress_events: list[tuple[int, int, str, int]] = []
    summary = indexer.process_pending(
        progress_callback=lambda current, total, path, faces: progress_events.append((current, total, path.name, faces))
    )

    assert progress_events == [(1, 2, "a.jpg", 1), (2, 2, "b.jpg", 2)]
    assert summary.detected_faces == 2
    assert summary.cancelled is False


def test_reference_indexer_emits_progress_for_each_pending_file(tmp_path: Path) -> None:
    ref_dir = tmp_path / "Ref"
    ref_dir.mkdir()
    (ref_dir / "Alice.png").write_bytes(b"1")
    (ref_dir / "Bob.png").write_bytes(b"2")

    db_path = initialize_database(tmp_path / "index.sqlite3")
    repository = FaceIndexRepository(connect(db_path))
    indexer = ReferenceIndexer(repository, FakeEngine())
    indexer.synchronize_inventory(ref_dir)

    progress_events: list[tuple[int, int, str, int]] = []
    summary = indexer.process_pending(
        progress_callback=lambda current, total, path, faces: progress_events.append((current, total, path.name, faces))
    )

    assert progress_events == [(1, 2, "Alice.png", 1), (2, 2, "Bob.png", 2)]
    assert summary.detected_faces == 2
    assert summary.cancelled is False


def test_photo_indexer_can_stop_between_files(tmp_path: Path) -> None:
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "a.jpg").write_bytes(b"a")
    (photos_dir / "b.jpg").write_bytes(b"b")

    db_path = initialize_database(tmp_path / "index.sqlite3")
    repository = FaceIndexRepository(connect(db_path))
    indexer = PhotoIndexer(repository, FakeEngine())
    indexer.synchronize_inventory([photos_dir])

    calls = 0

    def should_stop() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    summary = indexer.process_pending(should_stop=should_stop)

    assert summary.processed == 1
    assert summary.detected_faces == 1
    assert summary.cancelled is True