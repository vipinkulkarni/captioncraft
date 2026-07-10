"""Tests for judge-facing friendly failure captions."""

import sys
from pathlib import Path

import pytest

from src.caption import STYLES, public_caption

_EVAL_DIR = Path(__file__).resolve().parent.parent / "misc" / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from score_results import score_caption  # noqa: E402


@pytest.fixture
def friendly_failures_on(monkeypatch):
    monkeypatch.setenv("FRIENDLY_FAILURES", "1")


class TestPublicCaption:
    def test_passthrough_when_disabled(self, monkeypatch):
        monkeypatch.setenv("FRIENDLY_FAILURES", "0")
        raw = "Failed to caption: Truncated"
        assert public_caption(raw, style="formal") == raw

    @pytest.mark.parametrize("style", STYLES)
    def test_describe_failure_passes_scorer(self, friendly_failures_on, style):
        text = public_caption("Failed to describe video: EmptyResponse", style=style)
        ok, reason = score_caption(text, style)
        assert ok, reason

    @pytest.mark.parametrize("style", STYLES)
    def test_caption_failure_passes_scorer(self, friendly_failures_on, style):
        text = public_caption("Failed to caption: MetaLeak", style=style)
        ok, reason = score_caption(text, style)
        assert ok, reason

    @pytest.mark.parametrize("style", STYLES)
    def test_process_failure_passes_scorer(self, friendly_failures_on, style):
        text = public_caption("Failed to process video: RuntimeError", style=style)
        ok, reason = score_caption(text, style)
        assert ok, reason

    @pytest.mark.parametrize("style", STYLES)
    def test_invalid_task_passes_scorer(self, friendly_failures_on, style):
        text = public_caption("Invalid task input.", style=style)
        ok, reason = score_caption(text, style)
        assert ok, reason
