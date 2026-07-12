"""Unit tests for caption-vs-video accuracy parse and panel aggregate."""

from src.caption_vision_judge import (
    CaptionVisionAccuracy,
    aggregate_vision_panel,
    parse_caption_vision_accuracy_response,
    caption_vision_accuracy_enabled,
    resolve_caption_vision_judge_panel,
)


def test_parse_caption_vision_json():
    raw = '{"accuracy":0.85,"confidence":0.9,"issue":"invented UI"}'
    score = parse_caption_vision_accuracy_response(raw, judge_model="m3")
    assert score.ok
    assert score.accuracy == 0.85
    assert score.confidence == 0.9
    assert score.issue == "invented UI"


def test_parse_legacy_without_confidence_defaults_high():
    raw = '{"accuracy":0.85,"issue":"invented UI"}'
    score = parse_caption_vision_accuracy_response(raw, judge_model="m3")
    assert score.ok
    assert score.confidence == 1.0


def test_parse_rejects_out_of_range():
    raw = '{"accuracy":1.5,"confidence":0.9,"issue":""}'
    score = parse_caption_vision_accuracy_response(raw)
    assert not score.ok


def test_parse_regex_fallback():
    raw = 'noise "accuracy": 0.7, "confidence": 0.8, "issue": "thin" more'
    score = parse_caption_vision_accuracy_response(raw)
    assert score.ok
    assert score.accuracy == 0.7
    assert score.confidence == 0.8


def test_vision_accuracy_off_by_default(monkeypatch):
    monkeypatch.delenv("CAPTION_VISION_ACCURACY", raising=False)
    assert caption_vision_accuracy_enabled() is False
    monkeypatch.setenv("CAPTION_VISION_ACCURACY", "1")
    assert caption_vision_accuracy_enabled() is True


def test_aggregate_panel_mean_and_disagreement(monkeypatch):
    monkeypatch.setenv("CAPTION_VISION_JUDGE_MIN_CONFIDENCE", "0.7")
    monkeypatch.setenv("CAPTION_VISION_JUDGE_MAX_DISAGREE", "0.25")
    members = [
        CaptionVisionAccuracy(accuracy=0.9, confidence=0.85, judge_model="m3"),
        CaptionVisionAccuracy(accuracy=0.8, confidence=0.9, judge_model="kimi"),
    ]
    panel = aggregate_vision_panel(members)
    assert panel.ok
    assert panel.usable
    assert panel.accuracy == 0.85
    assert panel.disagreement == 0.1


def test_aggregate_panel_excludes_high_disagreement(monkeypatch):
    monkeypatch.setenv("CAPTION_VISION_JUDGE_MIN_CONFIDENCE", "0.7")
    monkeypatch.setenv("CAPTION_VISION_JUDGE_MAX_DISAGREE", "0.25")
    members = [
        CaptionVisionAccuracy(accuracy=1.0, confidence=0.9, judge_model="m3"),
        CaptionVisionAccuracy(accuracy=0.0, confidence=0.9, judge_model="kimi"),
    ]
    panel = aggregate_vision_panel(members)
    assert panel.ok
    assert not panel.usable
    assert panel.disagreement == 1.0


def test_aggregate_panel_excludes_low_confidence(monkeypatch):
    monkeypatch.setenv("CAPTION_VISION_JUDGE_MIN_CONFIDENCE", "0.7")
    members = [
        CaptionVisionAccuracy(accuracy=0.9, confidence=0.4, judge_model="m3"),
        CaptionVisionAccuracy(accuracy=0.85, confidence=0.5, judge_model="kimi"),
    ]
    panel = aggregate_vision_panel(members)
    assert panel.ok
    assert not panel.usable


def test_default_panel_is_m3_and_kimi(monkeypatch):
    monkeypatch.delenv("CAPTION_VISION_JUDGE_PANEL", raising=False)
    monkeypatch.delenv("CAPTION_VISION_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("CAPTION_VISION_JUDGE_ALT", raising=False)
    panel = resolve_caption_vision_judge_panel()
    assert len(panel) == 2
    assert "minimax-m3" in panel[0]
    assert "kimi-k2p6" in panel[1]
