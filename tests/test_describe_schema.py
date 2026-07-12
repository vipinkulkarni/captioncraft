"""Tests for structured describe JSON parsing."""

from src.describe_schema import VideoDescription, parse_describe_json


SAMPLE_JSON = """
{
  "subjects": [
    {
      "name": "orange-and-white kitten",
      "colors": ["orange", "white"],
      "distinguishing": ["pink nose"]
    }
  ],
  "setting": "outdoor garden, daylight",
  "actions_early": "sits still among foliage",
  "actions_late": "walks toward the camera with tail raised",
  "background": "green leaves and branches",
  "notable_moments": ["tail raised"]
}
"""


class TestDescribeSchema:
    def test_parse_valid_json(self):
        ok, reason, formatted = parse_describe_json(SAMPLE_JSON)
        assert ok, reason
        assert "orange-and-white kitten" in formatted
        assert "Actions (early):" in formatted
        assert "Actions (late):" in formatted

    def test_parse_valid_json_with_camera(self):
        raw = SAMPLE_JSON.replace(
            '"background": "green leaves and branches"',
            '"camera": "static close view", "background": "green leaves and branches"',
        )
        ok, reason, formatted = parse_describe_json(raw)
        assert ok, reason
        assert "Camera: static close view" in formatted

    def test_salvages_empty_subjects_from_setting(self):
        raw = """{
          "subjects": [],
          "setting": "outdoor park with a paved path and many green trees during daytime",
          "actions_early": "the camera remains stationary overlooking a park path",
          "actions_late": "the camera remains stationary overlooking a park path",
          "background": "lush green grass and large trees"
        }"""
        ok, reason, formatted = parse_describe_json(raw)
        assert ok, reason
        assert "park scene" in formatted

    def test_rejects_missing_required_fields(self):
        ok, reason, _formatted = parse_describe_json(
            '{"setting":"","actions_early":"a","actions_late":"b","subjects":[]}'
        )
        assert not ok
        assert reason == "InvalidJSON"

    def test_rejects_invalid_json(self):
        ok, reason, _formatted = parse_describe_json("{not json")
        assert not ok
        assert reason == "InvalidJSON"

    def test_to_style_context_includes_colors(self):
        description = VideoDescription(
            subjects=[{"name": "blue bus", "colors": ["blue"], "distinguishing": []}],
            setting="urban street",
            actions_early="drives left to right",
            actions_late="continues across frame",
        )
        text = description.to_style_context()
        assert "blue bus" in text
        assert "colors: blue" in text

    def test_on_screen_text_optional(self):
        raw = SAMPLE_JSON.replace(
            '"notable_moments": ["tail raised"]',
            '"on_screen_text": ["if (initial == 0)"], "notable_moments": ["tail raised"]',
        )
        ok, reason, formatted = parse_describe_json(raw)
        assert ok, reason
        assert "On-screen text: if (initial == 0)" in formatted
