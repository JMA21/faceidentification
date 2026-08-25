from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from faceid_app.models import FaceExtractorProtocol
from faceid_app.storage.repositories import FaceIndexRepository

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ProgressCallback = Callable[[int, int, Path, int], None]
ShouldStopCallback = Callable[[], bool]


@dataclass(slots=True)
class ReferenceSyncSummary:
    discovered: int = 0
    new_or_changed: int = 0
    deleted: int = 0
    processed: int = 0
    failed: int = 0
    detected_faces: int = 0
    cancelled: bool = False


class ReferenceIndexer:
    def __init__(self, repository: FaceIndexRepository, extractor: FaceExtractorProtocol) -> None:
        self.repository = repository
        self.extractor = extractor

    def synchronize_inventory(self, reference_directory: Path) -> ReferenceSyncSummary:
        summary = ReferenceSyncSummary()
        discovered_paths: list[Path] = []
        if not reference_directory.exists():
            summary.deleted = self.repository.delete_reference_images_missing_from_inventory(discovered_paths)
            self.repository.commit()
            return summary

        for image_path in sorted(reference_directory.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stats = image_path.stat()
            discovered_paths.append(image_path)
            summary.discovered += 1
            _, changed = self.repository.upsert_reference_image(
                image_path=image_path,
                person_name=image_path.stem,
                size_bytes=stats.st_size,
                modified_time_ns=stats.st_mtime_ns,
                fingerprint=self._fingerprint(stats.st_size, stats.st_mtime_ns),
            )
            if changed:
                summary.new_or_changed += 1
        summary.deleted = self.repository.delete_reference_images_missing_from_inventory(discovered_paths)
        self.repository.commit()
        return summary

    def process_pending(
        self,
        limit: int | None = None,
        progress_callback: ProgressCallback | None = None,
        should_stop: ShouldStopCallback | None = None,
    ) -> ReferenceSyncSummary:
        summary = ReferenceSyncSummary()
        pending_rows = self.repository.list_pending_reference_images(limit=limit)
        total = len(pending_rows)
        for index, row in enumerate(pending_rows, start=1):
            if should_stop is not None and should_stop():
                summary.cancelled = True
                break
            reference_image_id = int(row["id"])
            image_path = Path(str(row["path"]))
            self.repository.mark_reference_processing(reference_image_id)
            self.repository.commit()
            try:
                detections = self.extractor.extract_faces(image_path)
                self.repository.replace_reference_faces(reference_image_id, detections)
                self.repository.mark_reference_completed(reference_image_id)
                summary.processed += 1
                summary.detected_faces += len(detections)
            except Exception as exc:  # pragma: no cover - defensive for runtime engine failures
                self.repository.mark_reference_failed(reference_image_id, str(exc))
                summary.failed += 1
            if progress_callback is not None:
                progress_callback(index, total, image_path, summary.detected_faces)
            self.repository.commit()
        return summary

    @staticmethod
    def _fingerprint(size_bytes: int, modified_time_ns: int) -> str:
        return f"{size_bytes}:{modified_time_ns}"
