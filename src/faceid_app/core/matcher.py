from __future__ import annotations

from typing import Iterable

import numpy as np

from faceid_app.models import MatchCandidate, SearchResult
from faceid_app.storage.repositories import FaceIndexRepository


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


class FaceMatcher:
    def __init__(self, repository: FaceIndexRepository) -> None:
        self.repository = repository

    def search_person(self, person_name: str, threshold: float) -> SearchResult:
        person = self.repository.get_person(person_name)
        if person is None:
            return SearchResult(person_name=person_name, threshold=threshold, matches=[])

        person_id = int(person["id"])
        reference_embeddings = self.repository.get_reference_embeddings(person_id)
        if not reference_embeddings:
            return SearchResult(person_name=person_name, threshold=threshold, matches=[])

        matches: list[MatchCandidate] = []
        for photo_face_id, photo_path, bbox, photo_embedding in self.repository.iter_photo_faces():
            validation_state = self.repository.get_validation_state(person_id, photo_face_id)
            if validation_state == "rejected":
                continue

            best_similarity = max(
                cosine_similarity(reference_embedding, photo_embedding)
                for reference_embedding in reference_embeddings
            )
            if best_similarity < threshold and validation_state != "confirmed":
                continue

            matches.append(
                MatchCandidate(
                    person_id=person_id,
                    person_name=person_name,
                    photo_face_id=photo_face_id,
                    photo_path=photo_path,
                    similarity=best_similarity,
                    bbox=bbox,
                    source="manual" if validation_state in {"confirmed", "manual"} else "automatic",
                )
            )

        matches.sort(key=lambda item: item.similarity, reverse=True)
        search_run_id = self.repository.create_search_run(person_id, threshold)
        self.repository.replace_match_suggestions(search_run_id, person_id, matches)
        self.repository.commit()
        return SearchResult(person_name=person_name, threshold=threshold, matches=matches)
