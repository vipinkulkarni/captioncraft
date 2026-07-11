"""Tests for description-aware caption grounding score."""

from src.caption_grounding import (
    extract_action_verbs,
    extract_description_anchors,
    grounding_bonus,
)
from src.caption_selector import rank_caption, select_best_candidate, CaptionCandidate
from src.results import CaptionResult


SAMPLE_CONTEXT = """Setting: outdoor garden with dappled sunlight
Actions (early): ginger kitten sits alert among bushes
Actions (late): kitten walks forward toward camera
Background: dirt ground and green foliage
Subject 1: ginger kitten (colors: orange, white)
Notable moments: kitten sits still; kitten walks forward"""


class TestCaptionGrounding:
    def test_extract_anchors_from_formatted_context(self):
        anchors = extract_description_anchors(SAMPLE_CONTEXT)
        assert "kitten" in anchors
        assert "ginger" in anchors or "orange" in anchors

    def test_grounded_caption_scores_higher(self):
        grounded = (
            "The ginger kitten sits alert among green bushes in dappled sunlight. "
            "It then walks forward across the dirt ground toward the camera."
        )
        vague = (
            "A small animal pauses in the shade before moving closer. "
            "It continues along without much fanfare."
        )
        base_g, _ = rank_caption(grounded, "formal")
        base_v, _ = rank_caption(vague, "formal")
        g_g, _ = rank_caption(grounded, "formal", description=SAMPLE_CONTEXT)
        g_v, _ = rank_caption(vague, "formal", description=SAMPLE_CONTEXT)
        assert g_g > base_g
        assert g_g > g_v

    def test_select_best_prefers_grounded_candidate(self):
        description = SAMPLE_CONTEXT
        grounded = CaptionCandidate(
            text=(
                "The ginger kitten sits among bushes in sunlight. "
                "It walks forward on dirt toward the camera."
            ),
            model="a",
            label="a",
            result=CaptionResult(text="ok", error=None),
            score_rank=rank_caption(
                "The ginger kitten sits among bushes in sunlight. "
                "It walks forward on dirt toward the camera.",
                "formal",
                description=description,
            )[0],
            score_reason="ok",
        )
        vague = CaptionCandidate(
            text="An animal rests quietly then moves along. Nothing else happens.",
            model="b",
            label="b",
            result=CaptionResult(text="ok", error=None),
            score_rank=rank_caption(
                "An animal rests quietly then moves along. Nothing else happens.",
                "formal",
                description=description,
            )[0],
            score_reason="ok",
        )
        best = select_best_candidate([vague, grounded])
        assert best is not None
        assert best.label == "a"

    def test_grounding_bonus_returns_reason(self):
        bonus, reason = grounding_bonus(
            SAMPLE_CONTEXT,
            "The ginger kitten walks on dirt among foliage.",
        )
        assert bonus > 0
        assert "grounding=" in reason

    def test_action_verbs_boost_grounded_caption(self):
        verbs = extract_action_verbs(SAMPLE_CONTEXT)
        assert "sits" in verbs or "walks" in verbs
        grounded = (
            "The ginger kitten sits alert among bushes in sunlight. "
            "It walks forward across dirt toward the camera."
        )
        invented = (
            "The ginger kitten flies over bushes in sunlight. "
            "It swims forward across dirt toward the camera."
        )
        g_bonus, _ = grounding_bonus(SAMPLE_CONTEXT, grounded)
        i_bonus, i_reason = grounding_bonus(SAMPLE_CONTEXT, invented)
        assert g_bonus > i_bonus
        assert "mismatch" in i_reason
