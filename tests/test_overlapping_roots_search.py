from pathlib import Path

import numpy as np

from faceid_app.core.matcher import FaceMatcher
from faceid_app.core.photo_indexer import PhotoIndexer
from faceid_app.models import FaceDetection
from faceid_app.storage.database import connect, initialize_database
from faceid_app.storage.repositories import FaceIndexRepository


class FakeEngine:
    def extract_faces(self, image_path: Path):
        return [FaceDetection(bbox=(0, 0, 10, 10), embedding=np.array([1.0, 0.0], dtype=np.float32), score=0.99)]


def test_search_results_are_not_duplicated_with_overlapping_photo_roots(tmp_path: Path) -> None:
    photos_dir = tmp_path / "photos"
    nested_dir = photos_dir / "nested"
    nested_dir.mkdir(parents=True)
    photo_path = nested_dir / "img.jpg"
    photo_path.write_bytes(b"x")

    db_path = initialize_database(tmp_path / "index.sqlite3")
    repository = FaceIndexRepository(connect(db_path))

    ref_image = tmp_path / "Alice.png"
    ref_image.write_bytes(b"r")
    reference_id, _ = repository.upsert_reference_image(ref_image, "Alice", 1, 1, "1:1")
    repository.replace_reference_faces(
        reference_id,
        [FaceDetection(bbox=(0, 0, 10, 10), embedding=np.array([1.0, 0.0], dtype=np.float32), score=0.99)],
    )
    repository.mark_reference_completed(reference_id)
    repository.commit()

    indexer = PhotoIndexer(repository, FakeEngine())
    first_sync = indexer.synchronize_inventory([photos_dir, nested_dir])
    assert first_sync.discovered == 1
    indexer.process_pending()

    second_sync = indexer.synchronize_inventory([photos_dir, nested_dir])
    assert second_sync.discovered == 1
    assert second_sync.new_or_changed == 0

    matcher = FaceMatcher(repository)
    result = matcher.search_person("Alice", 0.5)

    assert len(result.matches) == 1
    assert len({match.photo_face_id for match in result.matches}) == 1
