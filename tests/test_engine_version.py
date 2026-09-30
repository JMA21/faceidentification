from faceid_app.core.face_engine import detect_engine_version


def test_detect_engine_version_returns_structured_metadata() -> None:
    info = detect_engine_version(check_updates=False)

    assert info.engine_name == "InsightFace"
    assert isinstance(info.available, bool)
    assert info.model_name == "buffalo_l"
    assert info.providers


def test_local_version_check_does_not_access_network(monkeypatch) -> None:
    def unexpected_fetch(_name):
        raise AssertionError("Local version detection must not access the network")

    monkeypatch.setattr("faceid_app.core.face_engine._fetch_pypi_payload", unexpected_fetch)
    assert detect_engine_version(check_updates=False).engine_name == "InsightFace"


def test_update_check_fetches_each_package_only_once(monkeypatch) -> None:
    calls = []

    def fetch(name):
        calls.append(name)
        return {
            "info": {"version": "2.0"},
            "releases": {
                "2.0": [{"packagetype": "sdist", "requires_python": ">=99"}],
                "1.0": [{"packagetype": "sdist", "requires_python": ">=3.10"}],
            },
        }

    monkeypatch.setattr("faceid_app.core.face_engine._fetch_pypi_payload", fetch)
    info = detect_engine_version()
    assert calls == ["insightface", "onnxruntime"]
    assert info.latest_package_version == "2.0"
    assert info.latest_compatible_package_version == "1.0"
