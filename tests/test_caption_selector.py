"""Tests for best-of-N caption selection."""

from unittest.mock import MagicMock, patch

from src.caption_selector import (
    CaptionCandidate,
    generate_best_of_n_caption,
    is_friendly_placeholder,
    pool_candidate_temperature,
    rank_caption,
    select_best_candidate,
)
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

    def test_pool_candidate_temperature_delta(self, monkeypatch):
        monkeypatch.setenv("CAPTION_POOL_TEMP_DELTA", "0.15")
        assert pool_candidate_temperature("formal", 0) is None
        assert pool_candidate_temperature("formal", 1) == 0.65  # 0.5 + 0.15

    def test_best_of_n_uses_higher_temp_for_second_candidate(self, monkeypatch):
        monkeypatch.setenv("CAPTION_POOL_TEMP_DELTA", "0.15")
        monkeypatch.setenv("CAPTION_JUDGE_TIEBREAK", "0")
        models = [
            ("a", "accounts/fireworks/models/deepseek-v4-flash"),
            ("b", "accounts/fireworks/models/deepseek-v4-flash"),
        ]
        ok = CaptionResult(
            text=(
                "Waves roll toward the rocky shore like a slow-motion stampede. "
                "The foam goes nowhere useful."
            ),
            error=None,
        )

        with patch(
            "src.caption_selector.generate_styled_caption_from_text",
            return_value=ok,
        ) as gen:
            generate_best_of_n_caption(
                client=MagicMock(),
                models=models,
                style="formal",
                description="Setting: rocky coast",
                parallel=False,
            )
        assert gen.call_count == 2
        temps = [c.kwargs.get("temperature_override") for c in gen.call_args_list]
        assert temps[0] is None
        assert temps[1] == 0.65
