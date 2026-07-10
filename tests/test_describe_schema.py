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

    def test_rejects_missing_subjects(self):
        ok, reason, _formatted = parse_describe_json(
            '{"setting":"x","actions_early":"a","actions_late":"b","subjects":[]}'
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
