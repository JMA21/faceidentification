from __future__ import annotations

import json
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Sequence
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version
from PIL import Image

from faceid_app.models import EngineVersionInfo, FaceDetection


def runtime_available_providers() -> tuple[str, ...]:
    try:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        if isinstance(providers, list):
            return tuple(str(provider) for provider in providers)
    except Exception:
        return ()
    return ()


def select_engine_providers(requested: Sequence[str] | None = None) -> tuple[str, ...]:
    available = runtime_available_providers()
    if not available:
        return ("CPUExecutionProvider",)

    if requested:
        selected = tuple(provider for provider in requested if provider in available)
        if selected:
            return selected

    priority = (
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )
    selected = tuple(provider for provider in priority if provider in available)
    return selected or tuple(available)


def _fetch_pypi_payload(package_name: str) -> dict | None:
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        with urlopen(url, timeout=2.5) as response:  # nosec B310 - trusted PyPI JSON endpoint
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def _fetch_latest_pypi_version(package_name: str) -> str | None:
    return _latest_pypi_version(_fetch_pypi_payload(package_name))


def _latest_pypi_version(payload: dict | None) -> str | None:
    info = payload.get("info") if isinstance(payload, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    return str(version) if version else None


def _is_python_compatible(requires_python: str | None) -> bool:
    if not requires_python:
        return True
    try:
        spec = SpecifierSet(requires_python)
    except Exception:
        return True
    import sys

    py_version = Version(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return py_version in spec


@lru_cache(maxsize=1)
def _current_tags() -> frozenset:
    return frozenset(sys_tags())


def _release_has_compatible_file(files: list[dict]) -> bool:
    current_tags = _current_tags()
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        if bool(file_info.get("yanked")):
            continue
        requires_python = file_info.get("requires_python")
        if not _is_python_compatible(str(requires_python) if requires_python is not None else None):
            continue

        packagetype = str(file_info.get("packagetype") or "")
        filename = str(file_info.get("filename") or "")

        if packagetype == "sdist":
            return True

        if packagetype == "bdist_wheel" and filename:
            try:
                _, _, _, wheel_tags = parse_wheel_filename(filename)
            except (InvalidWheelFilename, InvalidVersion):
                continue
            if wheel_tags & current_tags:
                return True
    return False


def _fetch_latest_compatible_version(package_name: str) -> str | None:
    return _latest_compatible_version(_fetch_pypi_payload(package_name))


def _latest_compatible_version(payload: dict | None) -> str | None:
    if not payload:
        return None
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        return None

    candidates: list[tuple[Version, list[dict]]] = []
    for version_str, files in releases.items():
        if not isinstance(files, list) or not files:
            continue
        try:
            parsed_version = Version(str(version_str))
        except InvalidVersion:
            continue
        candidates.append((parsed_version, files))
    for version, files in sorted(candidates, key=lambda item: item[0], reverse=True):
        if _release_has_compatible_file(files):
            return str(version)
    return None


def detect_engine_version(*, check_updates: bool = True) -> EngineVersionInfo:
    package_payload = _fetch_pypi_payload("insightface") if check_updates else None
    runtime_payload = _fetch_pypi_payload("onnxruntime") if check_updates else None
    latest_package_version = _latest_pypi_version(package_payload)
    latest_runtime_version = _latest_pypi_version(runtime_payload)
    latest_compatible_package_version = _latest_compatible_version(package_payload)
    latest_compatible_runtime_version = _latest_compatible_version(runtime_payload)
    try:
        package_version = metadata.version("insightface")
        runtime_version = metadata.version("onnxruntime")
        selected_providers = select_engine_providers()
        return EngineVersionInfo(
            engine_name="InsightFace",
            package_version=package_version,
            runtime_version=runtime_version,
            available=True,
            providers=selected_providers,
            details="InsightFace package detected.",
            latest_package_version=latest_package_version,
            latest_runtime_version=latest_runtime_version,
            latest_compatible_package_version=latest_compatible_package_version,
            latest_compatible_runtime_version=latest_compatible_runtime_version,
        )
    except metadata.PackageNotFoundError as exc:
        return EngineVersionInfo(
            engine_name="InsightFace",
            package_version=None,
            runtime_version=None,
            available=False,
            details=f"Package not installed: {exc}",
            latest_package_version=latest_package_version,
            latest_runtime_version=latest_runtime_version,
            latest_compatible_package_version=latest_compatible_package_version,
            latest_compatible_runtime_version=latest_compatible_runtime_version,
        )


class InsightFaceEngine:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: Sequence[str] | None = None,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self.model_name = model_name
        self.providers = select_engine_providers(providers)
        self.det_size = det_size
        self._app = None

    def prepare(self) -> None:
        if self._app is not None:
            return

        from insightface.app import FaceAnalysis  # type: ignore

        app = FaceAnalysis(
            name=self.model_name,
            providers=list(self.providers),
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=0, det_size=self.det_size)
        self._app = app

    def extract_faces(self, image_path: Path) -> list[FaceDetection]:
        self.prepare()
        assert self._app is not None

        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            image_array = np.array(rgb)
        bgr = np.ascontiguousarray(image_array[:, :, ::-1])

        faces = self._app.get(bgr)
        detections: list[FaceDetection] = []
        for face in faces:
            bbox = tuple(float(value) for value in face.bbox.tolist())
            embedding = np.asarray(face.embedding, dtype=np.float32)
            score = float(getattr(face, "det_score", 0.0))
            detections.append(FaceDetection(bbox=bbox, embedding=embedding, score=score))
        return detections
