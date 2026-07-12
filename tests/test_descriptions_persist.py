"""Tests for live description cache write/merge helpers."""

import json

from src.pipeline import (
    merge_descriptions_into_file,
    persist_live_descriptions,
    write_descriptions_cache,
)


def test_write_descriptions_cache_shape(tmp_path):
    path = tmp_path / "descriptions.json"
    write_descriptions_cache(path, {"e01": "a", "e02": "b"}, meta={"source": "test"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["descriptions"] == {"e01": "a", "e02": "b"}
    assert data["meta"]["source"] == "test"


def test_merge_preserves_other_keys(tmp_path):
    path = tmp_path / "fixture.json"
    write_descriptions_cache(path, {"e01": "old", "e99": "keep"})
    merge_descriptions_into_file(path, {"e01": "new", "e02": "added"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["descriptions"]["e01"] == "new"
    assert data["descriptions"]["e02"] == "added"
    assert data["descriptions"]["e99"] == "keep"
    assert data["meta"]["updated_from_live"] is True


def test_persist_live_writes_sibling_and_optional_fixture(tmp_path, monkeypatch):
    results = tmp_path / "out" / "results.json"
    results.parent.mkdir(parents=True)
    results.write_text("[]", encoding="utf-8")
    fixture = tmp_path / "fixture.json"
    write_descriptions_cache(fixture, {"e99": "keep"})
    monkeypatch.setenv("DESCRIPTIONS_UPDATE_PATH", str(fixture))

    persist_live_descriptions(
        results,
        {"e01": "live text"},
        update_fixture=True,
    )
    live = results.parent / "descriptions_live.json"
    assert live.is_file()
    live_data = json.loads(live.read_text(encoding="utf-8"))
    assert live_data["descriptions"]["e01"] == "live text"

    fixture_data = json.loads(fixture.read_text(encoding="utf-8"))
    assert fixture_data["descriptions"]["e01"] == "live text"
    assert fixture_data["descriptions"]["e99"] == "keep"


def test_persist_skips_fixture_when_disabled(tmp_path, monkeypatch):
    results = tmp_path / "results.json"
    results.write_text("[]", encoding="utf-8")
    fixture = tmp_path / "fixture.json"
    write_descriptions_cache(fixture, {"e99": "keep"})
    monkeypatch.setenv("DESCRIPTIONS_UPDATE_PATH", str(fixture))

    persist_live_descriptions(
        results,
        {"e01": "live text"},
        update_fixture=False,
    )
    fixture_data = json.loads(fixture.read_text(encoding="utf-8"))
    assert "e01" not in fixture_data["descriptions"]
    assert (results.parent / "descriptions_live.json").is_file()
