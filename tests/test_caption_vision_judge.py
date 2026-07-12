"""Unit tests for caption-vs-video accuracy parse."""

from src.caption_vision_judge import (
    parse_caption_vision_accuracy_response,
    caption_vision_accuracy_enabled,
)


def test_parse_caption_vision_json():
    raw = '{"accuracy":0.85,"issue":"invented UI"}'
    score = parse_caption_vision_accuracy_response(raw, judge_model="m3")
    assert score.ok
    assert score.accuracy == 0.85
    assert score.issue == "invented UI"


def test_parse_rejects_out_of_range():
    raw = '{"accuracy":1.5,"issue":""}'
    score = parse_caption_vision_accuracy_response(raw)
    assert not score.ok


def test_parse_regex_fallback():
    raw = 'noise "accuracy": 0.7, "issue": "thin" more'
    score = parse_caption_vision_accuracy_response(raw)
    assert score.ok
    assert score.accuracy == 0.7


def test_vision_accuracy_off_by_default(monkeypatch):
    monkeypatch.delenv("CAPTION_VISION_ACCURACY", raising=False)
    assert caption_vision_accuracy_enabled() is False
    monkeypatch.setenv("CAPTION_VISION_ACCURACY", "1")
    assert caption_vision_accuracy_enabled() is True
