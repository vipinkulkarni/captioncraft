"""Unit tests for shared retry helper."""

from unittest.mock import patch

import pytest

from src.retry import RetryPolicy, call_with_retry


class TestCallWithRetry:
    def test_returns_on_first_success(self):
        calls: list[int] = []

        def attempt(n: int) -> int:
            calls.append(n)
            return 42

        result, reasons = call_with_retry(
            policy=RetryPolicy(max_attempts=3, base_sleep_s=0, jitter_s=0),
            attempt=attempt,
            classify=lambda _n, value: None if value == 42 else "bad",
        )
        assert result == 42
        assert reasons == []
        assert calls == [1]

    def test_retries_then_succeeds(self):
        state = {"n": 0}

        def attempt(_n: int) -> int:
            state["n"] += 1
            return state["n"]

        with patch("src.retry.time.sleep") as sleep_mock:
            result, reasons = call_with_retry(
                policy=RetryPolicy(max_attempts=3, base_sleep_s=1.0, jitter_s=0),
                attempt=attempt,
                classify=lambda _n, value: None if value >= 2 else "retry",
            )

        assert result == 2
        assert reasons == ["retry"]
        sleep_mock.assert_called_once_with(1.0)

    def test_jitter_added_to_sleep(self):
        with patch("src.retry.random.uniform", return_value=0.25):
            with patch("src.retry.time.sleep") as sleep_mock:
                call_with_retry(
                    policy=RetryPolicy(max_attempts=2, base_sleep_s=1.0, jitter_s=0.5),
                    attempt=lambda _n: 0,
                    classify=lambda _n, _v: "fail",
                )
        sleep_mock.assert_called_once_with(1.25)

    def test_should_sleep_gate(self):
        with patch("src.retry.time.sleep") as sleep_mock:
            call_with_retry(
                policy=RetryPolicy(max_attempts=3, base_sleep_s=1.0, jitter_s=0),
                attempt=lambda n: n,
                classify=lambda n, _v: None if n >= 3 else "Truncated",
                should_sleep=lambda _attempt, reason: reason == "EmptyResponse",
            )
        assert sleep_mock.call_count == 0

    def test_exhausts_attempts(self):
        result, reasons = call_with_retry(
            policy=RetryPolicy(max_attempts=2, base_sleep_s=0, jitter_s=0),
            attempt=lambda _n: "",
            classify=lambda _n, value: "EmptyResponse" if not value else None,
        )
        assert result == ""
        assert reasons == ["EmptyResponse", "EmptyResponse"]

    def test_invalid_max_attempts(self):
        with pytest.raises(ValueError):
            call_with_retry(
                policy=RetryPolicy(max_attempts=0, base_sleep_s=0, jitter_s=0),
                attempt=lambda _n: 0,
                classify=lambda _n, _v: None,
            )
