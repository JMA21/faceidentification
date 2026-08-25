from faceid_app.core.face_engine import detect_engine_version


def test_detect_engine_version_returns_structured_metadata() -> None:
    info = detect_engine_version()

    assert info.engine_name == "InsightFace"
    assert isinstance(info.available, bool)
    assert info.model_name == "buffalo_l"
    assert info.providers
