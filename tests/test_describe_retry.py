"""Tests for describe retry policy."""

from src.pipeline import _describe_max_attempts, _describe_no_retry_reasons
from src.retry import RetryPolicy, call_with_retry


def test_describe_max_attempts_with_fallback(monkeypatch):
    monkeypatch.setenv("DESCRIBE_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("DESCRIBE_MAX_ATTEMPTS_WITH_FALLBACK", "1")
    monkeypatch.setenv("VISION_FALLBACK_MODEL", "accounts/fireworks/models/minimax-m3")
    assert _describe_max_attempts() == 1

    monkeypatch.delenv("VISION_FALLBACK_MODEL", raising=False)
    assert _describe_max_attempts() == 2


def test_call_with_retry_stops_on_timeout():
    policy = RetryPolicy(max_attempts=3, base_sleep_s=0.0, jitter_s=0.0)
    attempts = {"count": 0}

    def attempt_fn(_attempt: int) -> str:
        attempts["count"] += 1
        return "fail"

    def classify(_attempt: int, _result: str) -> str | None:
        return "ReadTimeout"

    def should_retry(_attempt: int, reason: str) -> bool:
        return reason not in _describe_no_retry_reasons()

    _, reasons = call_with_retry(
        policy=policy,
        attempt=attempt_fn,
        classify=classify,
        should_retry=should_retry,
        should_sleep=should_retry,
    )
    assert attempts["count"] == 1
    assert reasons == ["ReadTimeout"]
