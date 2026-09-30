from __future__ import annotations

from itertools import islice

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

        references = np.stack(reference_embeddings)
        reference_norms = np.linalg.norm(references, axis=1)
        validation_states = self.repository.get_validation_states(person_id)
        matches: list[MatchCandidate] = []
        faces = iter(self.repository.iter_photo_faces())
        while batch := list(islice(faces, 256)):
            embeddings = np.stack([face[3] for face in batch])
            denominators = np.linalg.norm(embeddings, axis=1)[:, None] * reference_norms[None, :]
            similarities = np.zeros((len(batch), len(references)), dtype=np.float32)
            np.divide(embeddings @ references.T, denominators, out=similarities, where=denominators != 0)
            best_similarities = similarities.max(axis=1)
            near_threshold = np.isclose(best_similarities, threshold, rtol=1e-6, atol=1e-6)

            for (photo_face_id, photo_path, bbox, embedding), similarity, needs_exact_score in zip(
                batch, best_similarities, near_threshold
            ):
                validation_state = validation_states.get(photo_face_id)
                if validation_state == "rejected":
                    continue
                best_similarity = float(similarity)
                # BLAS reductions can round differently from the original scalar dot product.
                # Preserve the exact inclusion decision at the threshold, not just close scores.
                if needs_exact_score:
                    best_similarity = max(cosine_similarity(reference, embedding) for reference in reference_embeddings)
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
