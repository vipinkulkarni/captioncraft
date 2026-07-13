"""Tests for post-pass judge retry helpers."""

import time
from unittest.mock import MagicMock, patch

from src.judge_retry import (
    PipelinedJudgeRetry,
    judge_feedback_nudge,
    list_judge_failures,
)
from src.llm_judge import CaptionJudgeScore, ClipJudgeResult


def test_list_judge_failures():
    clip = ClipJudgeResult(
        task_id="e01",
        captions={
            "formal": CaptionJudgeScore(
                style="formal", style_match=1.0, accuracy=1.0
            ),
            "sarcastic": CaptionJudgeScore(
                style="sarcastic", style_match=0.4, accuracy=1.0
            ),
        },
    )
    assert list_judge_failures(clip, min_score=0.8) == [("e01", "sarcastic")]


def test_list_judge_failures_meta_leak():
    clip = ClipJudgeResult(
        task_id="v8",
        captions={
            "humorous_non_tech": CaptionJudgeScore(
                style="humorous_non_tech",
                style_match=0.95,
                accuracy=0.95,
                meta_leak=True,
                issue="but careful drafting",
            ),
        },
    )
    assert list_judge_failures(clip, min_score=0.8) == [("v8", "humorous_non_tech")]


def test_list_judge_failures_regex(monkeypatch):
    monkeypatch.setenv("JUDGE_RETRY_REGEX", "1")
    clip = ClipJudgeResult(
        task_id="e06",
        captions={
            "humorous_non_tech": CaptionJudgeScore(
                style="humorous_non_tech", style_match=1.0, accuracy=1.0
            ),
        },
    )
    captions = {
        "humorous_non_tech": (
            "The waterfall deploys like an API stream into the green pool below."
        ),
    }
    assert list_judge_failures(clip, min_score=0.8, captions=captions) == [
        ("e06", "humorous_non_tech")
    ]


def test_list_judge_failures_quality_floor(monkeypatch):
    monkeypatch.setenv("JUDGE_RETRY_QUALITY_MIN", "0.8")
    monkeypatch.setenv("JUDGE_ACC_RETRY_FLOOR", "0.75")
    clip = ClipJudgeResult(
        task_id="e01",
        captions={
            "formal": CaptionJudgeScore(
                style="formal", style_match=1.0, accuracy=0.7
            ),
        },
    )
    assert list_judge_failures(clip, min_score=0.8) == [("e01", "formal")]


def test_list_judge_failures_keeps_punchy_pass(monkeypatch):
    """Uneven but strong mean should not be rewritten by retry."""
    monkeypatch.setenv("JUDGE_RETRY_QUALITY_MIN", "0.9")
    monkeypatch.setenv("JUDGE_ACC_RETRY_FLOOR", "0.75")
    monkeypatch.setenv("JUDGE_STYLE_RETRY_FLOOR", "0.70")
    monkeypatch.setenv("JUDGE_RETRY_REGEX", "0")
    clip = ClipJudgeResult(
        task_id="v1",
        captions={
            "sarcastic": CaptionJudgeScore(
                style="sarcastic", accuracy=0.86, style_match=0.96
            ),
        },
    )
    assert list_judge_failures(clip, min_score=0.9) == []


def test_judge_feedback_nudge_includes_issue():
    score = CaptionJudgeScore(
        style="formal",
        accuracy=0.5,
        style_match=0.6,
        issue="invented birds",
    )
    nudge = judge_feedback_nudge(score)
    assert "accuracy=0.50" in nudge
    assert "style_match=0.60" in nudge
    assert "invented birds" in nudge
    assert "not invent" in nudge.lower()


def test_judge_feedback_nudge_meta_leak():
    score = CaptionJudgeScore(
        style="humorous_non_tech",
        accuracy=0.2,
        style_match=0.2,
        meta_leak=True,
        issue="but careful drafting",
    )
    nudge = judge_feedback_nudge(score)
    assert "meta-leak" in nudge.lower() or "drafting" in nudge.lower()
    assert "finished" in nudge.lower() and "caption" in nudge.lower()


def test_apply_vision_accuracy_lowers_text_score(monkeypatch):
    from src.judge_retry import _apply_vision_accuracy
    from src.caption_vision_judge import CaptionVisionAccuracy

    clip = ClipJudgeResult(
        task_id="e01",
        captions={
            "formal": CaptionJudgeScore(
                style="formal", accuracy=1.0, style_match=0.95
            ),
        },
    )

    def _fake_vis(**_kwargs):
        return CaptionVisionAccuracy(accuracy=0.4, issue="invented UI", judge_model="m3")

    monkeypatch.setattr(
        "src.judge_retry.judge_caption_vision_accuracy",
        _fake_vis,
    )
    _apply_vision_accuracy(
        clip=clip,
        captions={"formal": "Someone types React code on a laptop."},
        frames=[b"fake"],
        client=MagicMock(),
        model="m3",
    )
    assert clip.captions["formal"].accuracy == 0.4
    assert "vision:" in clip.captions["formal"].issue


def test_pipelined_judge_retries_failed_style(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_RETRY", "1")
    monkeypatch.setenv("JUDGE_RETRY_REGEX", "0")
    monkeypatch.setenv("JUDGE_MIN_SCORE", "0.8")
    monkeypatch.setenv("JUDGE_RETRY_QUALITY_MIN", "0.8")
    monkeypatch.setenv("JUDGE_MIN_REMAINING_S", "0")
    monkeypatch.setenv("JUDGE_SKIP_DISTINCTNESS", "1")
    monkeypatch.setenv("JUDGE_PARALLEL_STYLES", "0")
    monkeypatch.setenv("JUDGE_RETRY_MAX_PER_STYLE", "2")

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
                style="formal", style_match=1.0, accuracy=1.0
            ),
            "sarcastic": CaptionJudgeScore(
                style="sarcastic",
                style_match=0.4,
                accuracy=1.0,
                issue="weak sarcasm",
            ),
            "humorous_tech": CaptionJudgeScore(
                style="humorous_tech", style_match=1.0, accuracy=1.0
            ),
            "humorous_non_tech": CaptionJudgeScore(
                style="humorous_non_tech", style_match=1.0, accuracy=1.0
            ),
        },
    )
    improved = CaptionJudgeScore(
        style="sarcastic", style_match=0.9, accuracy=0.9, issue=""
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
        judge_client=MagicMock(),
    )

    with patch("src.judge_retry.judge_clip_call", return_value=clip):
        with patch(
            "src.judge_retry._regenerate_style_caption",
            return_value="Retried sarcastic caption.",
        ) as regen:
            with patch(
                "src.judge_retry._judge_single_style",
                return_value=(improved, ""),
            ) as rejudge:
                coordinator.submit(results[0], clip_index=0)
                coordinator.finish()
                regen.assert_called_once()
                kwargs = regen.call_args.kwargs
                assert "weak sarcasm" in (kwargs.get("judge_feedback") or "")
                rejudge.assert_called_once()

    assert results[0]["captions"]["sarcastic"] == "Retried sarcastic caption."


def test_pipelined_judge_caps_attempts_and_keeps_best(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_RETRY", "1")
    monkeypatch.setenv("JUDGE_RETRY_REGEX", "0")
    monkeypatch.setenv("JUDGE_MIN_SCORE", "0.8")
    monkeypatch.setenv("JUDGE_RETRY_QUALITY_MIN", "0.8")
    monkeypatch.setenv("JUDGE_MIN_REMAINING_S", "0")
    monkeypatch.setenv("JUDGE_SKIP_DISTINCTNESS", "1")
    monkeypatch.setenv("JUDGE_PARALLEL_STYLES", "0")
    monkeypatch.setenv("JUDGE_RETRY_MAX_PER_STYLE", "2")

    results = [
        {
            "task_id": "e01",
            "captions": {"formal": "Original formal caption."},
        }
    ]
    results_path = tmp_path / "results.json"
    descriptions = {"e01": "Scene facts."}

    initial = CaptionJudgeScore(
        style="formal", style_match=0.4, accuracy=0.5, issue="invented UI"
    )
    mid = CaptionJudgeScore(
        style="formal", style_match=0.5, accuracy=0.6, issue="still vague"
    )
    worse = CaptionJudgeScore(
        style="formal", style_match=0.3, accuracy=0.4, issue="worse"
    )
    clip = ClipJudgeResult(task_id="e01", captions={"formal": initial})

    coordinator = PipelinedJudgeRetry(
        results=results,
        results_path=results_path,
        descriptions=descriptions,
        caption_client=MagicMock(),
        caption_model="accounts/fireworks/models/deepseek-v4-flash",
        run_start=time.monotonic(),
        time_budget_s=540.0,
        total_clips=1,
        judge_client=MagicMock(),
    )

    texts = ["Better formal caption.", "Worse formal caption."]
    scores = [(mid, ""), (worse, "")]

    with patch("src.judge_retry.judge_clip_call", return_value=clip):
        with patch(
            "src.judge_retry._regenerate_style_caption",
            side_effect=texts,
        ) as regen:
            with patch(
                "src.judge_retry._judge_single_style",
                side_effect=scores,
            ):
                coordinator.submit(results[0], clip_index=0)
                coordinator.finish()
                assert regen.call_count == 2
                assert "invented UI" in (regen.call_args_list[0].kwargs.get("judge_feedback") or "")

    assert results[0]["captions"]["formal"] == "Better formal caption."
