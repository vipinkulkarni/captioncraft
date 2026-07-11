"""Tests for describe quality gate and long-clip frame tuning."""

from src.describe_quality import describe_quality_issue, validate_describe_quality
from src.describe_schema import VideoDescription
from src.env import resolve_frame_count


GOOD_DESCRIPTION = VideoDescription(
    subjects=[{"name": "coastal cliffs", "colors": ["brown"], "distinguishing": []}],
    setting="rocky coastline with teal ocean waves",
    actions_early="waves crash repeatedly against dark rocks",
    actions_late="foam spreads across the shore as swells continue",
    background="open sea and cloudy sky",
    notable_moments=["white foam spreads across rocks"],
)


class TestDescribeQuality:
    def test_accepts_rich_long_clip_description(self):
        ok, reason = validate_describe_quality(
            GOOD_DESCRIPTION,
            duration_s=86.0,
        )
        assert ok, reason

    def test_rejects_static_actions(self):
        description = VideoDescription(
            subjects=[{"name": "kitten", "colors": [], "distinguishing": []}],
            setting="garden path in daylight",
            actions_early="the kitten sits still among bushes",
            actions_late="the kitten sits still among bushes",
            background="green foliage",
        )
        ok, reason = validate_describe_quality(description, duration_s=30.0)
        assert not ok
        assert reason == "StaticActions"

    def test_rejects_weak_long_clip_without_background(self):
        description = VideoDescription(
            subjects=[{"name": "coast", "colors": [], "distinguishing": []}],
            setting="rocky coastline with ocean waves",
            actions_early="waves crash against rocks repeatedly",
            actions_late="foam spreads across the rocky shore",
            background="",
        )
        ok, reason = validate_describe_quality(description, duration_s=86.0)
        assert not ok
        assert reason == "MissingBackground"

    def test_describe_quality_issue_flags_weak_json(self, monkeypatch):
        monkeypatch.setenv("DESCRIBE_QUALITY_GATE", "1")
        raw = """{
          "subjects": [{"name": "park", "colors": [], "distinguishing": []}],
          "setting": "outdoor park with paved paths",
          "actions_early": "camera stays still overlooking trees",
          "actions_late": "camera stays still overlooking trees",
          "background": "trees"
        }"""
        assert describe_quality_issue(raw, duration_s=30.0) == "StaticActions"


class TestLongClipFrames:
    def test_long_clip_uses_higher_frame_cap(self, monkeypatch):
        monkeypatch.delenv("FRAME_COUNT", raising=False)
        monkeypatch.setenv("FRAME_LONG_DURATION_S", "60")
        monkeypatch.setenv("FRAME_LONG_INTERVAL_S", "3")
        monkeypatch.setenv("FRAME_LONG_COUNT_MIN", "12")
        monkeypatch.setenv("FRAME_LONG_COUNT_MAX", "28")
        assert resolve_frame_count(86.0) == 28

    def test_short_clip_uses_default_cap(self, monkeypatch):
        monkeypatch.delenv("FRAME_COUNT", raising=False)
        monkeypatch.setenv("FRAME_INTERVAL_S", "4")
        monkeypatch.setenv("FRAME_COUNT_MIN", "8")
        monkeypatch.setenv("FRAME_COUNT_MAX", "24")
        assert resolve_frame_count(30.0) == 8

    def test_duration_hint_path_uses_long_clip_count(self, monkeypatch):
        monkeypatch.delenv("FRAME_COUNT", raising=False)
        monkeypatch.setenv("FRAME_LONG_DURATION_S", "60")
        monkeypatch.setenv("FRAME_LONG_INTERVAL_S", "3")
        monkeypatch.setenv("FRAME_LONG_COUNT_MAX", "28")
        assert resolve_frame_count(86.0) == 28
