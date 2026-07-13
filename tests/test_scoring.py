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
        text = (
            "They found a calm patch of water and stayed there all afternoon. "
            "The reef nearby took the heavier waves."
        )
        assert is_structural_failure(text) == (False, "")

    def test_truncated_one_liner_is_incomplete(self):
        text = "The black editor screen watches its mult."
        assert is_structural_failure(text) == (True, "incomplete")

    def test_finished_punchy_one_liner_ok(self):
        text = (
            "A kitten outdoors, clearly plotting something elaborate and fully "
            "confident it will succeed."
        )
        assert is_structural_failure(text) == (False, "")
        ok, reason = score_caption(text, "sarcastic")
        assert ok, reason

    def test_describe_dump_is_structural(self):
        text = (
            "Background: coral reef with turquoise water. "
            "Notable moments: fish swimming past."
        )
        assert is_structural_failure(text) == (True, "describe-dump")

    def test_drafting_phrase_fails_score(self):
        text = (
            "Also need to reference the pink syntax. "
            "The dark editor shows an if statement on screen."
        )
        ok, reason = score_caption(text, "formal")
        assert not ok
        assert reason == "meta-leak"