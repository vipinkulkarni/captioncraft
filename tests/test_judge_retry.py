"""Tests for post-pass judge retry helpers."""

import time
from unittest.mock import MagicMock, patch

from src.judge_retry import PipelinedJudgeRetry, list_judge_failures
from src.llm_judge import CaptionJudgeScore, ClipJudgeResult


def test_list_judge_failures():
    clip = ClipJudgeResult(
        task_id="e01",
        captions={
            "formal": CaptionJudgeScore(
                style="formal", style_fit=5, accuracy=5, specificity=5
            ),
            "sarcastic": CaptionJudgeScore(
                style="sarcastic", style_fit=2, accuracy=5, specificity=5
            ),
        },
    )
    assert list_judge_failures(clip, min_score=3) == [("e01", "sarcastic")]


def test_pipelined_judge_retries_failed_style(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_RETRY", "1")
    monkeypatch.setenv("JUDGE_MIN_REMAINING_S", "0")
    monkeypatch.setenv("JUDGE_SKIP_DISTINCTNESS", "1")
    monkeypatch.setenv("JUDGE_PARALLEL_STYLES", "0")

    results = [
        {
            "task_id": "e01",
            "captions": {
                "formal": "A formal caption.",
                "sarcastic": "A sarcastic caption.",
                "humorous_tech": "Tech joke caption.",
                "humorous_non_tech": "Plain humor caption.",
            },
        }
    ]
    results_path = tmp_path / "results.json"
    descriptions = {"e01": "Scene facts about a bird."}

    clip = ClipJudgeResult(
        task_id="e01",
        captions={
            "formal": CaptionJudgeScore(
                style="formal", style_fit=5, accuracy=5, specificity=5
            ),
            "sarcastic": CaptionJudgeScore(
                style="sarcastic", style_fit=2, accuracy=5, specificity=5
            ),
            "humorous_tech": CaptionJudgeScore(
                style="humorous_tech", style_fit=5, accuracy=5, specificity=5
            ),
            "humorous_non_tech": CaptionJudgeScore(
                style="humorous_non_tech", style_fit=5, accuracy=5, specificity=5
            ),
        },
    )

    coordinator = PipelinedJudgeRetry(
        results=results,
        results_path=results_path,
        descriptions=descriptions,
        caption_client=MagicMock(),
        caption_model="accounts/fireworks/models/deepseek-v4-flash",
        run_start=time.monotonic(),
        time_budget_s=540.0,
        total_clips=1,
    )

    with patch("src.judge_retry.judge_clip_call", return_value=clip):
        with patch(
            "src.judge_retry._regenerate_style_caption",
            return_value="Retried sarcastic caption.",
        ) as regen:
            coordinator.submit(results[0], clip_index=0)
            coordinator.finish()
            regen.assert_called_once()

    assert results[0]["captions"]["sarcastic"] == "Retried sarcastic caption."
