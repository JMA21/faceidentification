from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


APP_FOLDER_NAME = "FaceIdLocalApp"
SETTINGS_FILE_NAME = "settings.json"

INDEXING_PRESETS: dict[str, tuple[str, int]] = {
    "rapide": ("buffalo_s", 416),
    "equilibre": ("buffalo_l", 512),
    "precis": ("buffalo_l", 640),
}


@dataclass(slots=True)
class WindowState:
    width: int = 1280
    height: int = 860


@dataclass(slots=True)
class AppSettings:
    photo_directories: list[str] = field(default_factory=list)
    reference_directory: str = ""
    storage_directory: str = ""
    similarity_threshold: float = 0.55
    indexing_model_name: str = "buffalo_l"
    indexing_detection_size: int = 640
    window: WindowState = field(default_factory=WindowState)
    last_selected_person: str = ""

    @staticmethod
    def _normalize_path_text(path_text: str) -> str:
        path = Path(path_text).expanduser().resolve(strict=False)
        return os.path.normcase(str(path))

    def normalized_photo_directories(self) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for raw_path in self.photo_directories:
            clean = raw_path.strip()
            if not clean:
                continue
            normalized_path = self._normalize_path_text(clean)
            if normalized_path in seen:
                continue
            seen.add(normalized_path)
            normalized.append(normalized_path)
        return normalized

    def overlapping_photo_directories(self) -> list[tuple[str, str]]:
        normalized = self.normalized_photo_directories()
        overlaps: list[tuple[str, str]] = []
        for left_index, left in enumerate(normalized):
            left_parts = Path(left).parts
            for right in normalized[left_index + 1 :]:
                right_parts = Path(right).parts
                if _is_parent_or_same(left_parts, right_parts):
                    overlaps.append((left, right))
                elif _is_parent_or_same(right_parts, left_parts):
                    overlaps.append((right, left))
        return overlaps

    def indexing_preset_name(self) -> str:
        for preset_name, (model_name, detection_size) in INDEXING_PRESETS.items():
            if self.indexing_model_name == model_name and self.indexing_detection_size == detection_size:
                return preset_name
        return "precis"

    def apply_indexing_preset(self, preset_name: str) -> None:
        model_name, detection_size = INDEXING_PRESETS.get(preset_name, INDEXING_PRESETS["precis"])
        self.indexing_model_name = model_name
        self.indexing_detection_size = detection_size

    @classmethod
    def load(cls, settings_path: Path | None = None) -> "AppSettings":
        path = settings_path or default_settings_path()
        if not path.exists():
            return cls.default_for_workspace()

        data = json.loads(path.read_text(encoding="utf-8"))
        window_data = data.get("window", {})
        return cls(
            photo_directories=list(data.get("photo_directories", [])),
            reference_directory=str(data.get("reference_directory", "")),
            storage_directory=str(data.get("storage_directory", "")),
            similarity_threshold=float(data.get("similarity_threshold", 0.55)),
            indexing_model_name=str(data.get("indexing_model_name", "buffalo_l")),
            indexing_detection_size=max(320, int(data.get("indexing_detection_size", 640))),
            window=WindowState(
                width=int(window_data.get("width", 1280)),
                height=int(window_data.get("height", 860)),
            ),
            last_selected_person=str(data.get("last_selected_person", "")),
        )

    def save(self, settings_path: Path | None = None) -> Path:
        path = settings_path or default_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def default_for_workspace(cls, workspace_root: Path | None = None) -> "AppSettings":
        root = workspace_root or detect_workspace_root()
        if root is None:
            return cls()

        photo_directory = root / "MesPhotos"
        reference_directory = root / "Ref"
        storage_directory = root / ".faceid-data"

        return cls(
            photo_directories=[str(photo_directory)] if photo_directory.exists() else [],
            reference_directory=str(reference_directory) if reference_directory.exists() else "",
            storage_directory=str(storage_directory),
        )


def default_settings_directory() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / APP_FOLDER_NAME
    return Path.home() / ".faceid-local-app"


def default_settings_path() -> Path:
    return default_settings_directory() / SETTINGS_FILE_NAME


def detect_workspace_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]

    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
        if (candidate / "MesPhotos").exists() or (candidate / "Ref").exists():
            return candidate

    return None


def _is_parent_or_same(parent_parts: Iterable[str], child_parts: Iterable[str]) -> bool:
    parent = tuple(parent_parts)
    child = tuple(child_parts)
    if len(parent) > len(child):
        return False
    return child[: len(parent)] == parent
