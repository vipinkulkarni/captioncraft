"""Shared retry loop with jitter for API calls."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    base_sleep_s: float
    jitter_s: float = 0.5

    def sleep_before_retry(self, *, enabled: bool = True) -> None:
        if not enabled:
            return
        delay = self.base_sleep_s + random.uniform(0, max(self.jitter_s, 0.0))
        if delay > 0:
            time.sleep(delay)


def call_with_retry(
    *,
    policy: RetryPolicy,
    attempt: Callable[[int], T],
    classify: Callable[[int, T], str | None],
    should_sleep: Callable[[int, str], bool] | None = None,
    should_retry: Callable[[int, str], bool] | None = None,
    on_failure: Callable[[int, str, T], None] | None = None,
) -> tuple[T, list[str]]:
    """Run up to max_attempts. classify returns None on success, else a failure reason."""
    if policy.max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    sleep_gate = should_sleep or (lambda _attempt, _reason: True)
    reasons: list[str] = []
    last: T | None = None

    for attempt_idx in range(1, policy.max_attempts + 1):
        last = attempt(attempt_idx)
        reason = classify(attempt_idx, last)
        if reason is None:
            return last, reasons

        reasons.append(reason)
        if on_failure is not None:
            on_failure(attempt_idx, reason, last)
        if should_retry is not None and not should_retry(attempt_idx, reason):
            break
        if attempt_idx < policy.max_attempts:
            policy.sleep_before_retry(enabled=sleep_gate(attempt_idx, reason))

    assert last is not None
    return last, reasons
