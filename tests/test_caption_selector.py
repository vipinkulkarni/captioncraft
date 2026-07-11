"""Tests for best-of-N caption selection."""

from src.caption_selector import CaptionCandidate, is_friendly_placeholder, rank_caption, select_best_candidate
from src.results import CaptionResult


class TestCaptionSelector:
    def test_friendly_placeholder_ranks_lowest(self):
        good = "Waves roll toward the rocky shore like a slow-motion stampede. The foam goes nowhere."
        friendly = "Something's clearly happening here, but the caption never quite came together."
        assert rank_caption(good, "sarcastic")[0] > rank_caption(friendly, "sarcastic")[0]
        assert is_friendly_placeholder(friendly)

    def test_select_best_prefers_structural_pass(self):
        candidates = [
            CaptionCandidate(
                text="Something's clearly happening here, but the caption never quite came together.",
                model="accounts/fireworks/models/glm-5p1",
                label="glm-5p1",
                result=CaptionResult(text=None, error=None),
                score_rank=0,
                score_reason="friendly-placeholder",
            ),
            CaptionCandidate(
                text=(
                    "Teal waves crash against dark rocks like an overworked dishwasher. "
                    "The foam still pretends it is going somewhere."
                ),
                model="accounts/fireworks/models/deepseek-v4-flash",
                label="deepseek-v4-flash",
                result=CaptionResult(text="ok", error=None),
                score_rank=105,
                score_reason="ok",
            ),
        ]
        best = select_best_candidate(candidates)
        assert best is not None
        assert best.label == "deepseek-v4-flash"
