from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


@dataclass(slots=True)
class EngineVersionInfo:
    engine_name: str
    package_version: str | None
    runtime_version: str | None
    available: bool
    model_name: str = "buffalo_l"
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    details: str | None = None
    latest_package_version: str | None = None
    latest_runtime_version: str | None = None
    latest_compatible_package_version: str | None = None
    latest_compatible_runtime_version: str | None = None
    package_url: str = "https://pypi.org/project/insightface/"
    runtime_url: str = "https://pypi.org/project/onnxruntime/"
    install_command: str = "pip install -U insightface onnxruntime"


@dataclass(slots=True)
class FaceDetection:
    bbox: tuple[float, float, float, float]
    embedding: np.ndarray
    score: float


@dataclass(slots=True)
class IndexedPhoto:
    id: int
    path: Path
    root_path: Path
    fingerprint: str
    status: str


@dataclass(slots=True)
class IndexedReference:
    id: int
    path: Path
    person_name: str
    fingerprint: str
    status: str


@dataclass(slots=True)
class MatchCandidate:
    person_id: int
    person_name: str
    photo_face_id: int
    photo_path: Path
    similarity: float
    bbox: tuple[float, float, float, float]
    source: str = "automatic"


@dataclass(slots=True)
class SearchResult:
    person_name: str
    threshold: float
    matches: list[MatchCandidate] = field(default_factory=list)


@dataclass(slots=True)
class UserValidation:
    person_id: int
    photo_face_id: int
    state: str
    source: str = "user"


class FaceExtractorProtocol(Protocol):
    def extract_faces(self, image_path: Path) -> Sequence[FaceDetection]:
        ...
