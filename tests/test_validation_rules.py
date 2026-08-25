from pathlib import Path

import numpy as np

from faceid_app.core.matcher import FaceMatcher
from faceid_app.models import FaceDetection
from faceid_app.storage.database import connect, initialize_database
from faceid_app.storage.repositories import FaceIndexRepository


def test_rejected_match_is_excluded_from_search(tmp_path: Path) -> None:
    db_path = initialize_database(tmp_path / "index.sqlite3")
    repository = FaceIndexRepository(connect(db_path))

    person_id = repository.ensure_person("Alice")
    reference_id, _ = repository.upsert_reference_image(tmp_path / "Alice.png", "Alice", 1, 1, "1:1")
    repository.replace_reference_faces(
        reference_id,
        [FaceDetection(bbox=(0, 0, 10, 10), embedding=np.array([1.0, 0.0], dtype=np.float32), score=0.99)],
    )
    repository.mark_reference_completed(reference_id)

    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    photo_path = photo_dir / "img.jpg"
    photo_path.write_bytes(b"x")
    photo_id, _ = repository.upsert_photo(photo_dir, photo_path, 1, 1, "1:1")
    repository.replace_photo_faces(
        photo_id,
        [FaceDetection(bbox=(0, 0, 10, 10), embedding=np.array([1.0, 0.0], dtype=np.float32), score=0.99)],
    )
    repository.mark_photo_completed(photo_id)
    repository.commit()

    matcher = FaceMatcher(repository)
    first_result = matcher.search_person("Alice", 0.5)
    assert len(first_result.matches) == 1

    repository.set_validation(person_id, first_result.matches[0].photo_face_id, "rejected")
    repository.commit()

    second_result = matcher.search_person("Alice", 0.5)
    assert second_result.matches == []
