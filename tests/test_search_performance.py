import sqlite3

import numpy as np
import pytest

from faceid_app.core.matcher import FaceMatcher, cosine_similarity
from faceid_app.models import FaceDetection
from faceid_app.storage.database import connect, initialize_database
from faceid_app.storage.repositories import FaceIndexRepository


@pytest.fixture
def repository(tmp_path):
    connection = connect(initialize_database(tmp_path / "index.sqlite3"))
    yield FaceIndexRepository(connection)
    connection.close()


def detection(embedding):
    return FaceDetection((0, 0, 10, 10), np.asarray(embedding, dtype=np.float32), 0.99)


def test_batch_search_matches_scalar_scores_and_preserves_validation(repository, tmp_path):
    rng = np.random.default_rng(42)
    references = rng.normal(size=(4, 32)).astype(np.float32)
    references[0] = 0
    reference_id, _ = repository.upsert_reference_image(tmp_path / "Alice.png", "Alice", 1, 1, "1:1")
    repository.replace_reference_faces(reference_id, [detection(vector) for vector in references])
    person_id = repository.ensure_person("Alice")
    embeddings = rng.normal(size=(601, 32)).astype(np.float32)
    embeddings[0] = 0
    photo_id, _ = repository.upsert_photo(tmp_path, tmp_path / "photo.jpg", 1, 1, "1:1")
    repository.replace_photo_faces(photo_id, [detection(vector) for vector in embeddings])
    repository.mark_photo_completed(photo_id)
    faces = list(repository.iter_photo_faces())
    for face, state in zip(faces, ("confirmed", "rejected", "manual")):
        repository.set_validation(person_id, face[0], state)
    repository.commit()

    expected = {}
    for index, face in enumerate(faces):
        score = max(cosine_similarity(vector, face[3]) for vector in references)
        if index != 1 and (score >= 0.2 or index == 0):
            expected[face[0]] = score

    queries = []
    repository.connection.set_trace_callback(queries.append)
    result = FaceMatcher(repository).search_person("Alice", 0.2)
    repository.connection.set_trace_callback(None)
    assert {match.photo_face_id for match in result.matches} == set(expected)
    for match in result.matches:
        assert match.similarity == pytest.approx(expected[match.photo_face_id], abs=1e-6)
    confirmed = next(match for match in result.matches if match.photo_face_id == faces[0][0])
    assert confirmed.similarity == 0
    assert confirmed.source == "manual"
    scores = [match.similarity for match in result.matches]
    assert scores == sorted(scores, reverse=True)
    assert len([query for query in queries if "SELECT" in query and "validation_rules" in query]) == 1


def test_zero_norm_and_negative_similarity(repository, tmp_path):
    ref_id, _ = repository.upsert_reference_image(tmp_path / "Alice.png", "Alice", 1, 1, "1:1")
    repository.replace_reference_faces(ref_id, [detection([1, 0])])
    photo_id, _ = repository.upsert_photo(tmp_path, tmp_path / "photo.jpg", 1, 1, "1:1")
    repository.replace_photo_faces(photo_id, [detection([0, 0]), detection([-1, 0]), detection([1, 0])])
    repository.mark_photo_completed(photo_id)
    repository.commit()
    result = FaceMatcher(repository).search_person("Alice", -1)
    assert [match.similarity for match in result.matches] == [1, 0, -1]


def test_unknown_person_and_empty_references(repository):
    matcher = FaceMatcher(repository)
    assert matcher.search_person("Unknown", 0.5).matches == []
    repository.ensure_person("Alice")
    assert matcher.search_person("Alice", 0.5).matches == []


def test_decision_at_exact_threshold_matches_scalar_calculation(repository, tmp_path):
    rng = np.random.default_rng(19)
    reference = rng.normal(size=512).astype(np.float32)
    embeddings = rng.normal(size=(20, 512)).astype(np.float32)
    ref_id, _ = repository.upsert_reference_image(tmp_path / "Alice.png", "Alice", 1, 1, "1:1")
    repository.replace_reference_faces(ref_id, [detection(reference)])
    photo_id, _ = repository.upsert_photo(tmp_path, tmp_path / "photo.jpg", 1, 1, "1:1")
    repository.replace_photo_faces(photo_id, [detection(vector) for vector in embeddings])
    repository.mark_photo_completed(photo_id)
    repository.commit()
    matcher = FaceMatcher(repository)
    faces = list(repository.iter_photo_faces())
    for face_id, _, _, embedding in faces:
        threshold = cosine_similarity(reference, embedding)
        assert face_id in {match.photo_face_id for match in matcher.search_person("Alice", threshold).matches}
        just_above = float(np.nextafter(threshold, np.inf))
        assert face_id not in {match.photo_face_id for match in matcher.search_person("Alice", just_above).matches}


@pytest.mark.parametrize("kind", ["photos", "references"])
def test_inventory_cleanup_works_with_low_sqlite_variable_limit(repository, tmp_path, kind):
    paths = [tmp_path / f"{index}.jpg" for index in range(50)]
    for path in paths:
        if kind == "photos":
            repository.upsert_photo(tmp_path, path, 1, 1, "1:1")
        else:
            repository.upsert_reference_image(path, path.stem, 1, 1, "1:1")
    repository.commit()
    # Python 3.10 has no Connection.setlimit; a large inventory still exercises the path.
    if hasattr(repository.connection, "setlimit"):
        repository.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 10)
    if kind == "photos":
        deleted = repository.delete_photos_missing_from_inventory([tmp_path], paths[:-1])
    else:
        deleted = repository.delete_reference_images_missing_from_inventory(paths[:-1])
    assert deleted == 1


def test_unchanged_inventory_does_not_write_photo_or_reference(repository, tmp_path):
    repository.upsert_photo(tmp_path, tmp_path / "photo.jpg", 1, 1, "1:1")
    repository.upsert_reference_image(tmp_path / "Alice.png", "Alice", 1, 1, "1:1")
    repository.commit()
    before = repository.connection.total_changes
    assert repository.upsert_photo(tmp_path, tmp_path / "photo.jpg", 1, 1, "1:1")[1] is False
    assert repository.upsert_reference_image(tmp_path / "Alice.png", "Alice", 1, 1, "1:1")[1] is False
    assert repository.connection.total_changes == before