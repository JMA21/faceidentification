from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from faceid_app.models import FaceDetection, MatchCandidate


def pack_embedding(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack_embedding(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.float32)


class FaceIndexRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def upsert_photo_root(self, root_path: Path) -> int:
        self.connection.execute(
            "INSERT OR IGNORE INTO photo_roots(path) VALUES (?)",
            (str(root_path),),
        )
        row = self.connection.execute(
            "SELECT id FROM photo_roots WHERE path = ?",
            (str(root_path),),
        ).fetchone()
        assert row is not None
        return int(row["id"])

    def upsert_photo(self, root_path: Path, photo_path: Path, size_bytes: int, modified_time_ns: int, fingerprint: str) -> tuple[int, bool]:
        root_id = self.upsert_photo_root(root_path)
        row = self.connection.execute(
            "SELECT id, fingerprint FROM photos WHERE path = ?",
            (str(photo_path),),
        ).fetchone()
        if row is None:
            cursor = self.connection.execute(
                """
                INSERT INTO photos(root_id, path, size_bytes, modified_time_ns, fingerprint, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (root_id, str(photo_path), size_bytes, modified_time_ns, fingerprint),
            )
            return int(cursor.lastrowid), True

        changed = str(row["fingerprint"]) != fingerprint
        status = "pending" if changed else self.connection.execute(
            "SELECT status FROM photos WHERE id = ?",
            (int(row["id"]),),
        ).fetchone()["status"]
        self.connection.execute(
            """
            UPDATE photos
            SET root_id = ?, size_bytes = ?, modified_time_ns = ?, fingerprint = ?, status = ?,
                updated_at = CURRENT_TIMESTAMP,
                error_message = CASE WHEN ? THEN NULL ELSE error_message END
            WHERE id = ?
            """,
            (root_id, size_bytes, modified_time_ns, fingerprint, status, 1 if changed else 0, int(row["id"])),
        )
        return int(row["id"]), changed

    def list_pending_photos(self, limit: int | None = None, root_paths: Sequence[Path] | None = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM photos WHERE status IN ('pending', 'processing')"
        params_list: list[object] = []
        if root_paths is not None:
            root_path_values = list(dict.fromkeys(str(path) for path in root_paths))
            if not root_path_values:
                return []
            root_placeholders = ",".join("?" for _ in root_path_values)
            query += f" AND root_id IN (SELECT id FROM photo_roots WHERE path IN ({root_placeholders}))"
            params_list.extend(root_path_values)

        query += " ORDER BY id"
        params: tuple[object, ...] = tuple(params_list)
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)
        return list(self.connection.execute(query, params).fetchall())

    def count_pending_photos(self, root_paths: Sequence[Path] | None = None) -> int:
        query = "SELECT COUNT(*) AS total FROM photos WHERE status IN ('pending', 'processing')"
        params_list: list[object] = []
        if root_paths is not None:
            root_path_values = list(dict.fromkeys(str(path) for path in root_paths))
            if not root_path_values:
                return 0
            root_placeholders = ",".join("?" for _ in root_path_values)
            query += f" AND root_id IN (SELECT id FROM photo_roots WHERE path IN ({root_placeholders}))"
            params_list.extend(root_path_values)

        row = self.connection.execute(query, tuple(params_list)).fetchone()
        return 0 if row is None else int(row["total"])

    def delete_photos_missing_from_inventory(self, root_paths: Sequence[Path], discovered_paths: Sequence[Path]) -> int:
        if not root_paths:
            return 0

        root_path_values = list(dict.fromkeys(str(path) for path in root_paths))
        root_placeholders = ",".join("?" for _ in root_path_values)
        params: list[object] = [*root_path_values]
        query = (
            "DELETE FROM photos "
            "WHERE root_id IN (SELECT id FROM photo_roots WHERE path IN (" + root_placeholders + "))"
        )
        if discovered_paths:
            discovered_values = list(dict.fromkeys(str(path) for path in discovered_paths))
            discovered_placeholders = ",".join("?" for _ in discovered_values)
            query += f" AND path NOT IN ({discovered_placeholders})"
            params.extend(discovered_values)

        cursor = self.connection.execute(query, tuple(params))
        return int(cursor.rowcount)

    def mark_photo_processing(self, photo_id: int) -> None:
        self.connection.execute(
            "UPDATE photos SET status = 'processing', updated_at = CURRENT_TIMESTAMP, error_message = NULL WHERE id = ?",
            (photo_id,),
        )

    def mark_photo_completed(self, photo_id: int) -> None:
        self.connection.execute(
            "UPDATE photos SET status = 'completed', last_indexed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (photo_id,),
        )

    def mark_photo_failed(self, photo_id: int, error_message: str) -> None:
        self.connection.execute(
            "UPDATE photos SET status = 'failed', error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (error_message, photo_id),
        )

    def replace_photo_faces(self, photo_id: int, detections: Sequence[FaceDetection]) -> None:
        self.connection.execute("DELETE FROM photo_faces WHERE photo_id = ?", (photo_id,))
        self.connection.executemany(
            """
            INSERT INTO photo_faces(photo_id, bbox_left, bbox_top, bbox_right, bbox_bottom, score, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    photo_id,
                    float(d.bbox[0]),
                    float(d.bbox[1]),
                    float(d.bbox[2]),
                    float(d.bbox[3]),
                    float(d.score),
                    pack_embedding(d.embedding),
                )
                for d in detections
            ],
        )

    def photo_status_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*) AS total FROM photos GROUP BY status"
        ).fetchall()
        return {str(row["status"]): int(row["total"]) for row in rows}

    def ensure_person(self, display_name: str) -> int:
        self.connection.execute(
            "INSERT OR IGNORE INTO reference_people(display_name) VALUES (?)",
            (display_name,),
        )
        row = self.connection.execute(
            "SELECT id FROM reference_people WHERE display_name = ?",
            (display_name,),
        ).fetchone()
        assert row is not None
        return int(row["id"])

    def upsert_reference_image(self, image_path: Path, person_name: str, size_bytes: int, modified_time_ns: int, fingerprint: str) -> tuple[int, bool]:
        person_id = self.ensure_person(person_name)
        row = self.connection.execute(
            "SELECT id, fingerprint FROM reference_images WHERE path = ?",
            (str(image_path),),
        ).fetchone()
        if row is None:
            cursor = self.connection.execute(
                """
                INSERT INTO reference_images(person_id, path, size_bytes, modified_time_ns, fingerprint, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (person_id, str(image_path), size_bytes, modified_time_ns, fingerprint),
            )
            return int(cursor.lastrowid), True

        changed = str(row["fingerprint"]) != fingerprint
        status = "pending" if changed else self.connection.execute(
            "SELECT status FROM reference_images WHERE id = ?",
            (int(row["id"]),),
        ).fetchone()["status"]
        self.connection.execute(
            """
            UPDATE reference_images
            SET person_id = ?, size_bytes = ?, modified_time_ns = ?, fingerprint = ?, status = ?,
                updated_at = CURRENT_TIMESTAMP,
                error_message = CASE WHEN ? THEN NULL ELSE error_message END
            WHERE id = ?
            """,
            (person_id, size_bytes, modified_time_ns, fingerprint, status, 1 if changed else 0, int(row["id"])),
        )
        return int(row["id"]), changed

    def list_pending_reference_images(self, limit: int | None = None) -> list[sqlite3.Row]:
        query = "SELECT reference_images.*, reference_people.display_name AS person_name FROM reference_images JOIN reference_people ON reference_people.id = reference_images.person_id WHERE status IN ('pending', 'processing') ORDER BY reference_images.id"
        params: tuple[object, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        return list(self.connection.execute(query, params).fetchall())

    def delete_reference_images_missing_from_inventory(self, discovered_paths: Sequence[Path]) -> int:
        if discovered_paths:
            discovered_values = [str(path) for path in discovered_paths]
            placeholders = ",".join("?" for _ in discovered_values)
            cursor = self.connection.execute(
                f"DELETE FROM reference_images WHERE path NOT IN ({placeholders})",
                tuple(discovered_values),
            )
        else:
            cursor = self.connection.execute("DELETE FROM reference_images")

        deleted = int(cursor.rowcount)
        self.connection.execute(
            "DELETE FROM reference_people WHERE id NOT IN (SELECT DISTINCT person_id FROM reference_images)"
        )
        return deleted

    def mark_reference_processing(self, reference_image_id: int) -> None:
        self.connection.execute(
            "UPDATE reference_images SET status = 'processing', updated_at = CURRENT_TIMESTAMP, error_message = NULL WHERE id = ?",
            (reference_image_id,),
        )

    def mark_reference_completed(self, reference_image_id: int) -> None:
        self.connection.execute(
            "UPDATE reference_images SET status = 'completed', last_indexed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (reference_image_id,),
        )

    def mark_reference_failed(self, reference_image_id: int, error_message: str) -> None:
        self.connection.execute(
            "UPDATE reference_images SET status = 'failed', error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (error_message, reference_image_id),
        )

    def replace_reference_faces(self, reference_image_id: int, detections: Sequence[FaceDetection]) -> None:
        self.connection.execute("DELETE FROM reference_faces WHERE reference_image_id = ?", (reference_image_id,))
        self.connection.executemany(
            """
            INSERT INTO reference_faces(reference_image_id, bbox_left, bbox_top, bbox_right, bbox_bottom, score, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    reference_image_id,
                    float(d.bbox[0]),
                    float(d.bbox[1]),
                    float(d.bbox[2]),
                    float(d.bbox[3]),
                    float(d.score),
                    pack_embedding(d.embedding),
                )
                for d in detections
            ],
        )

    def reference_status_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*) AS total FROM reference_images GROUP BY status"
        ).fetchall()
        return {str(row["status"]): int(row["total"]) for row in rows}

    def list_people(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM reference_people ORDER BY display_name").fetchall())

    def get_person(self, person_name: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM reference_people WHERE display_name = ?",
            (person_name,),
        ).fetchone()

    def get_reference_embeddings(self, person_id: int) -> list[np.ndarray]:
        rows = self.connection.execute(
            """
            SELECT reference_faces.embedding
            FROM reference_faces
            JOIN reference_images ON reference_images.id = reference_faces.reference_image_id
            WHERE reference_images.person_id = ?
            """,
            (person_id,),
        ).fetchall()
        return [unpack_embedding(row["embedding"]) for row in rows]

    def iter_photo_faces(self) -> Iterable[tuple[int, Path, tuple[float, float, float, float], np.ndarray]]:
        rows = self.connection.execute(
            """
            SELECT photo_faces.id, photos.path, photo_faces.bbox_left, photo_faces.bbox_top,
                   photo_faces.bbox_right, photo_faces.bbox_bottom, photo_faces.embedding
            FROM photo_faces
            JOIN photos ON photos.id = photo_faces.photo_id
            WHERE photos.status = 'completed'
            ORDER BY photo_faces.id
            """
        ).fetchall()
        for row in rows:
            yield (
                int(row["id"]),
                Path(str(row["path"])),
                (
                    float(row["bbox_left"]),
                    float(row["bbox_top"]),
                    float(row["bbox_right"]),
                    float(row["bbox_bottom"]),
                ),
                unpack_embedding(row["embedding"]),
            )

    def create_search_run(self, person_id: int, threshold: float) -> int:
        cursor = self.connection.execute(
            "INSERT INTO search_runs(person_id, threshold) VALUES (?, ?)",
            (person_id, threshold),
        )
        return int(cursor.lastrowid)

    def replace_match_suggestions(self, search_run_id: int, person_id: int, matches: Sequence[MatchCandidate]) -> None:
        self.connection.executemany(
            """
            INSERT INTO match_suggestions(search_run_id, person_id, photo_face_id, similarity)
            VALUES (?, ?, ?, ?)
            """,
            [
                (search_run_id, person_id, match.photo_face_id, float(match.similarity))
                for match in matches
            ],
        )

    def set_validation(self, person_id: int, photo_face_id: int, state: str, source: str = "user") -> None:
        self.connection.execute(
            """
            INSERT INTO validation_rules(person_id, photo_face_id, state, source)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(person_id, photo_face_id) DO UPDATE SET
                state = excluded.state,
                source = excluded.source,
                updated_at = CURRENT_TIMESTAMP
            """,
            (person_id, photo_face_id, state, source),
        )

    def get_validation_state(self, person_id: int, photo_face_id: int) -> str | None:
        row = self.connection.execute(
            "SELECT state FROM validation_rules WHERE person_id = ? AND photo_face_id = ?",
            (person_id, photo_face_id),
        ).fetchone()
        return None if row is None else str(row["state"])

    def save_engine_metadata(self, metadata: dict[str, str]) -> None:
        self.connection.executemany(
            """
            INSERT INTO engine_metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
            """,
            list(metadata.items()),
        )

    def commit(self) -> None:
        self.connection.commit()
