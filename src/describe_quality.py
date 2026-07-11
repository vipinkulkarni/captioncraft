"""Heuristic quality gate for structured describe output."""

from __future__ import annotations

from src.describe_schema import VideoDescription, parse_video_description
from src.env import get_float_env, get_int_env


def _word_count(text: str) -> int:
    return len(text.split())


def validate_describe_quality(
    description: VideoDescription,
    *,
    duration_s: float = 0.0,
) -> tuple[bool, str]:
    min_action_words = max(get_int_env("DESCRIBE_MIN_ACTION_WORDS", 4), 1)
    min_setting_words = max(get_int_env("DESCRIBE_MIN_SETTING_WORDS", 3), 1)

    if _word_count(description.setting) < min_setting_words:
        return False, "WeakSetting"

    if _word_count(description.actions_early) < min_action_words:
        return False, "WeakActionsEarly"
    if _word_count(description.actions_late) < min_action_words:
        return False, "WeakActionsLate"

    early = description.actions_early.strip().lower()
    late = description.actions_late.strip().lower()
    if early and early == late:
        return False, "StaticActions"

    long_threshold = get_float_env("DESCRIBE_LONG_DURATION_S", 60.0)
    if duration_s >= long_threshold:
        if not description.notable_moments:
            early_words = set(early.split())
            late_words = set(late.split())
            if len(early_words.symmetric_difference(late_words)) < 2:
                return False, "WeakLongClipActions"
        if not description.background.strip():
            return False, "MissingBackground"

    return True, ""


def describe_quality_issue(raw_text: str, *, duration_s: float = 0.0) -> str | None:
    if get_int_env("DESCRIBE_QUALITY_GATE", 1) != 1:
        return None
    ok, reason, description = parse_video_description(raw_text)
    if not ok or description is None:
        return reason or "InvalidJSON"
    ok_quality, quality_reason = validate_describe_quality(
        description,
        duration_s=duration_s,
    )
    if ok_quality:
        return None
    return quality_reason or "WeakDescribe"
