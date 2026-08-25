from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from faceid_app.models import FaceExtractorProtocol
from faceid_app.storage.repositories import FaceIndexRepository

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ProgressCallback = Callable[[int, int, Path, int], None]
ShouldStopCallback = Callable[[], bool]


@dataclass(slots=True)
class SyncSummary:
    discovered: int = 0
    new_or_changed: int = 0
    deleted: int = 0
    processed: int = 0
    failed: int = 0
    detected_faces: int = 0
    cancelled: bool = False


class PhotoIndexer:
    def __init__(self, repository: FaceIndexRepository, extractor: FaceExtractorProtocol) -> None:
        self.repository = repository
        self.extractor = extractor

    @staticmethod
    def _normalized_path_key(path: Path) -> str:
        return os.path.normcase(str(path))

    def _normalize_roots(self, root_directories: Sequence[Path]) -> list[Path]:
        normalized_roots: list[Path] = []
        seen_keys: set[str] = set()
        for root in root_directories:
            normalized_root = root.expanduser().resolve(strict=False)
            if not normalized_root.exists():
                continue
            key = self._normalized_path_key(normalized_root)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            normalized_roots.append(normalized_root)
        return normalized_roots

    def discover(self, root_directories: Sequence[Path]) -> Iterable[tuple[Path, Path]]:
        for root in self._normalize_roots(root_directories):
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield root, path.resolve(strict=False)

    def synchronize_inventory(self, root_directories: Sequence[Path]) -> SyncSummary:
        summary = SyncSummary()
        normalized_roots = self._normalize_roots(root_directories)
        discovered_paths_by_key: dict[str, Path] = {}
        for root, photo_path in self.discover(root_directories):
            discovered_key = self._normalized_path_key(photo_path)
            if discovered_key in discovered_paths_by_key:
                continue
            discovered_paths_by_key[discovered_key] = photo_path

            stats = photo_path.stat()
            summary.discovered += 1
            _, changed = self.repository.upsert_photo(
                root_path=root,
                photo_path=photo_path,
                size_bytes=stats.st_size,
                modified_time_ns=stats.st_mtime_ns,
                fingerprint=self._fingerprint(stats.st_size, stats.st_mtime_ns),
            )
            if changed:
                summary.new_or_changed += 1
        summary.deleted = self.repository.delete_photos_missing_from_inventory(
            normalized_roots,
            list(discovered_paths_by_key.values()),
        )
        self.repository.commit()
        return summary

    def count_pending(self, root_directories: Sequence[Path] | None = None) -> int:
        pending_roots = None if root_directories is None else self._normalize_roots(root_directories)
        return self.repository.count_pending_photos(root_paths=pending_roots)

    def process_pending(
        self,
        limit: int | None = None,
        root_directories: Sequence[Path] | None = None,
        progress_callback: ProgressCallback | None = None,
        should_stop: ShouldStopCallback | None = None,
    ) -> SyncSummary:
        summary = SyncSummary()
        pending_roots = None if root_directories is None else self._normalize_roots(root_directories)
        pending_rows = self.repository.list_pending_photos(limit=limit, root_paths=pending_roots)
        total = len(pending_rows)
        for index, row in enumerate(pending_rows, start=1):
            if should_stop is not None and should_stop():
                summary.cancelled = True
                break
            photo_id = int(row["id"])
            photo_path = Path(str(row["path"]))
            self.repository.mark_photo_processing(photo_id)
            self.repository.commit()
            try:
                detections = self.extractor.extract_faces(photo_path)
                self.repository.replace_photo_faces(photo_id, detections)
                self.repository.mark_photo_completed(photo_id)
                summary.processed += 1
                summary.detected_faces += len(detections)
            except Exception as exc:  # pragma: no cover - defensive for runtime engine failures
                self.repository.mark_photo_failed(photo_id, str(exc))
                summary.failed += 1
            if progress_callback is not None:
                progress_callback(index, total, photo_path, summary.detected_faces)
            self.repository.commit()
        return summary

    @staticmethod
    def _fingerprint(size_bytes: int, modified_time_ns: int) -> str:
        return f"{size_bytes}:{modified_time_ns}"
