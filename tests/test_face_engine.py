import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from faceid_app.core.face_engine import InsightFaceEngine
from faceid_app.core.photo_indexer import PhotoIndexer
from faceid_app.core.reference_indexer import ReferenceIndexer
from faceid_app.storage.database import connect, initialize_database
from faceid_app.storage.repositories import FaceIndexRepository


@pytest.fixture
def fake_analysis(monkeypatch):
    instances = []

    class FakeAnalysis:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.prepare_calls = []
            instances.append(self)

        def prepare(self, **kwargs):
            self.prepare_calls.append(kwargs)

        def get(self, image):
            assert image.flags.c_contiguous
            assert image[0, 0].tolist() == [30, 20, 10]
            return [SimpleNamespace(bbox=np.array([1, 2, 3, 4]), embedding=np.array([1, 0]), det_score=0.9)]

    package = ModuleType("insightface")
    module = ModuleType("insightface.app")
    module.FaceAnalysis = FakeAnalysis
    package.app = module
    monkeypatch.setitem(sys.modules, "insightface", package)
    monkeypatch.setitem(sys.modules, "insightface.app", module)
    monkeypatch.setattr("faceid_app.core.face_engine.select_engine_providers", lambda requested=None: ("CPUExecutionProvider",))
    return instances, FakeAnalysis


def test_engine_uses_only_required_models_and_prepares_once(fake_analysis):
    instances, _ = fake_analysis
    engine = InsightFaceEngine()
    engine.prepare()
    engine.prepare()
    assert len(instances) == 1
    assert instances[0].options["allowed_modules"] == ["detection", "recognition"]
    assert instances[0].prepare_calls == [{"ctx_id": 0, "det_size": (640, 640)}]


def test_engine_extracts_from_contiguous_bgr_image(fake_analysis, tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (10, 10), (10, 20, 30)).save(path)
    faces = InsightFaceEngine().extract_faces(path)
    assert len(faces) == 1
    assert faces[0].bbox == (1, 2, 3, 4)
    assert faces[0].embedding.dtype == np.float32
    assert faces[0].score == 0.9


def test_failed_engine_preparation_can_be_retried(fake_analysis, monkeypatch):
    instances, analysis_class = fake_analysis
    original_prepare = analysis_class.prepare
    monkeypatch.setattr(analysis_class, "prepare", lambda *args, **kwargs: 1 / 0)
    engine = InsightFaceEngine()
    with pytest.raises(ZeroDivisionError):
        engine.prepare()
    assert engine._app is None
    monkeypatch.setattr(analysis_class, "prepare", original_prepare)
    engine.prepare()
    assert len(instances) == 2
    assert engine._app is instances[1]


@pytest.mark.parametrize("indexer_class", [PhotoIndexer, ReferenceIndexer])
def test_corrupted_image_is_failed_without_stopping_indexing(fake_analysis, tmp_path, indexer_class):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a-broken.png").write_bytes(b"not an image")
    Image.new("RGB", (10, 10), (10, 20, 30)).save(images / "b-valid.png")
    connection = connect(initialize_database(tmp_path / "index.sqlite3"))
    try:
        repository = FaceIndexRepository(connection)
        indexer = indexer_class(repository, InsightFaceEngine())
        indexer.synchronize_inventory([images] if indexer_class is PhotoIndexer else images)
        result = indexer.process_pending()
        assert result.failed == 1
        assert result.processed == 1
        assert result.detected_faces == 1
        assert indexer.process_pending().processed == 0
    finally:
        connection.close()