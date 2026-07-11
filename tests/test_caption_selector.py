"""Tests for best-of-N caption selection."""

from unittest.mock import MagicMock, patch

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

    def test_tiebreak_uses_judge_when_scores_are_close(self, monkeypatch):
        monkeypatch.setenv("CAPTION_JUDGE_TIEBREAK", "1")
        monkeypatch.setenv("CAPTION_TIEBREAK_MARGIN", "8")
        close_a = CaptionCandidate(
            text=(
                "Teal waves crash against dark rocks like an overworked dishwasher. "
                "The foam still pretends it is going somewhere."
            ),
            model="a",
            label="a",
            result=CaptionResult(text="ok", error=None),
            score_rank=108,
            score_reason="ok",
        )
        close_b = CaptionCandidate(
            text=(
                "Dark rocks meet teal waves in a slow-motion argument. "
                "The foam keeps auditioning for a travel brochure."
            ),
            model="b",
            label="b",
            result=CaptionResult(text="ok", error=None),
            score_rank=105,
            score_reason="ok",
        )
        with patch(
            "src.llm_judge.judge_tiebreak_pick",
            return_value=1,
        ) as tiebreak:
            best = select_best_candidate(
                [close_a, close_b],
                style="sarcastic",
                description="Setting: rocky coast",
                client=MagicMock(),
                task_id="e01",
            )
        tiebreak.assert_called_once()
        assert best is not None
        assert best.label == "b"
