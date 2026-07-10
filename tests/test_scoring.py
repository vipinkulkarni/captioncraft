"""Tests for regex caption scoring."""

from src.scoring import is_structural_failure, score_caption


class TestTechJargonFalsePositives:
    def test_calm_patch_not_tech_jargon(self):
        text = (
            "The pair in dark swimwear found the one calm patch while the reef "
            "takes all the wave action. They've mastered doing absolutely nothing."
        )
        ok, reason = score_caption(text, "humorous_non_tech")
        assert ok, reason

    def test_neck_patch_not_tech_jargon(self):
        text = (
            "That wood pigeon with the white neck patch keeps pecking at twigs "
            "like it is searching for buried treasure. It lifts its head often."
        )
        ok, reason = score_caption(text, "humorous_non_tech")
        assert ok, reason


class TestStructuralFailure:
    def test_failed_caption_is_structural(self):
        assert is_structural_failure("Failed to caption: MetaLeak") == (True, "error")

    def test_calm_patch_is_not_structural(self):
        text = "They found a calm patch of water and stayed there all afternoon."
        assert is_structural_failure(text) == (False, "")
