from pathlib import Path
import os

from faceid_app.config.settings import AppSettings, default_settings_path


def test_settings_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings(
        photo_directories=["C:/Photos/A", "C:/Photos/B"],
        reference_directory="C:/Ref",
        storage_directory="C:/Store",
        similarity_threshold=0.72,
        indexing_model_name="buffalo_s",
        indexing_detection_size=416,
    )

    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded.photo_directories == settings.photo_directories
    assert loaded.reference_directory == settings.reference_directory
    assert loaded.storage_directory == settings.storage_directory
    assert loaded.similarity_threshold == settings.similarity_threshold
    assert loaded.indexing_model_name == settings.indexing_model_name
    assert loaded.indexing_detection_size == settings.indexing_detection_size


def test_default_settings_path_is_stable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))
    path = default_settings_path()
    assert path.parent == tmp_path / "FaceIdLocalApp"
    assert path.name == "settings.json"


def test_default_settings_preload_workspace_directories(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "MesPhotos").mkdir()
    (tmp_path / "Ref").mkdir()

    monkeypatch.chdir(tmp_path)
    settings = AppSettings.load(tmp_path / "missing-settings.json")

    assert settings.photo_directories == [str(tmp_path / "MesPhotos")]
    assert settings.reference_directory == str(tmp_path / "Ref")
    assert settings.storage_directory == str(tmp_path / ".faceid-data")


def test_normalized_photo_directories_deduplicates_equivalent_paths(tmp_path: Path) -> None:
    photos_dir = tmp_path / "Photos"
    photos_dir.mkdir()
    equivalent = photos_dir / ".." / "Photos"

    settings = AppSettings(photo_directories=[str(photos_dir), str(equivalent), str(photos_dir)])

    normalized = settings.normalized_photo_directories()
    assert normalized == [os.path.normcase(str(photos_dir.resolve()))]


def test_overlapping_photo_directories_detects_nested_paths(tmp_path: Path) -> None:
    root = tmp_path / "photos"
    child = root / "nested"
    child.mkdir(parents=True)

    settings = AppSettings(photo_directories=[str(root), str(child)])

    overlaps = settings.overlapping_photo_directories()
    assert overlaps == [
        (
            os.path.normcase(str(root.resolve())),
            os.path.normcase(str(child.resolve())),
        )
    ]


def test_indexing_preset_helpers_apply_and_detect() -> None:
    settings = AppSettings()
    settings.apply_indexing_preset("rapide")
    assert settings.indexing_model_name == "buffalo_s"
    assert settings.indexing_detection_size == 416
    assert settings.indexing_preset_name() == "rapide"
