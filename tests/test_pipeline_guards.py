"""Tests for the deadline guard and prefetch scheduling."""

import time

from src.pipeline import _next_prefetch_urls, _RunContext


def _make_ctx(**overrides) -> _RunContext:
    defaults = dict(
        dry_run=False,
        frame_width=384,
        parallel_styles=True,
        descriptions_cache={},
        client=None,
        caption_client=None,
        vision_model="google-ai/gemma-4-26b-a4b-it",
        caption_model="accounts/fireworks/models/deepseek-v4-flash",
        run_start=time.monotonic(),
        time_budget_s=0.0,
        deadline_reserve_s=30.0,
    )
    defaults.update(overrides)
    return _RunContext(**defaults)


class TestDeadlineGuard:
    def test_disabled_when_no_budget(self, monkeypatch):
        monkeypatch.setenv("VISION_FALLBACK_MODEL", "accounts/fireworks/models/minimax-m3")
        ctx = _make_ctx(time_budget_s=0.0)
        assert not ctx.should_skip_primary_describe(remaining_clips=12)

    def test_disabled_without_fallback_or_dual(self, monkeypatch):
        monkeypatch.delenv("VISION_FALLBACK_MODEL", raising=False)
        monkeypatch.setenv("DESCRIBE_DUAL", "0")
        monkeypatch.delenv("VISION_ALT_MODEL", raising=False)
        ctx = _make_ctx(time_budget_s=540.0, run_start=time.monotonic() - 530.0)
        assert not ctx.should_skip_primary_describe(remaining_clips=5)

    def test_skips_when_dual_enabled_and_budget_tight(self, monkeypatch):
        monkeypatch.delenv("VISION_FALLBACK_MODEL", raising=False)
        monkeypatch.setenv("DESCRIBE_DUAL", "1")
        monkeypatch.setenv("VISION_ALT_MODEL", "accounts/fireworks/models/qwen3p7-plus")
        ctx = _make_ctx(time_budget_s=540.0, run_start=time.monotonic() - 500.0)
        assert ctx.should_skip_primary_describe(remaining_clips=5)

    def test_skips_primary_when_budget_tight(self, monkeypatch):
        monkeypatch.setenv("VISION_FALLBACK_MODEL", "accounts/fireworks/models/minimax-m3")
        # 540s budget, 500s elapsed, 5 clips x 30s reserve = 150s needed > 40s left.
        ctx = _make_ctx(time_budget_s=540.0, run_start=time.monotonic() - 500.0)
        assert ctx.should_skip_primary_describe(remaining_clips=5)

    def test_allows_primary_with_headroom(self, monkeypatch):
        monkeypatch.setenv("VISION_FALLBACK_MODEL", "accounts/fireworks/models/minimax-m3")
        ctx = _make_ctx(time_budget_s=540.0, run_start=time.monotonic() - 60.0)
        assert not ctx.should_skip_primary_describe(remaining_clips=5)


class TestPrefetchUrls:
    TASKS = [
        {"task_id": "a", "video_url": "http://x/a.mp4"},
        {"task_id": "b", "video_url": "http://x/b.mp4"},
        {"task_id": "c", "video_url": "http://x/c.mp4"},
        {"task_id": "d", "video_url": "http://x/d.mp4"},
    ]

    def test_returns_up_to_depth(self):
        urls = _next_prefetch_urls(
            self.TASKS, 0, descriptions_cache={}, dry_run=False, depth=2
        )
        assert urls == ["http://x/b.mp4", "http://x/c.mp4"]

    def test_skips_cached_tasks(self):
        urls = _next_prefetch_urls(
            self.TASKS,
            0,
            descriptions_cache={"b": "cached description"},
            dry_run=False,
            depth=2,
        )
        assert urls == ["http://x/c.mp4", "http://x/d.mp4"]

    def test_empty_at_end(self):
        urls = _next_prefetch_urls(
            self.TASKS, 3, descriptions_cache={}, dry_run=False, depth=2
        )
        assert urls == []
