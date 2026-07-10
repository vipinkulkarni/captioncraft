"""Typed pipeline results and error enums."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DescribeError(str, Enum):
    EMPTY = "empty"
    TRUNCATED = "truncated"
    API = "api"
    TIMEOUT = "timeout"


class CaptionError(str, Enum):
    EMPTY = "empty"
    TRUNCATED = "truncated"
    META_LEAK = "meta_leak"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    API = "api"


class ProcessError(str, Enum):
    INVALID_TASK = "invalid_task"
    UNSUPPORTED_STYLE = "unsupported_style"
    PROCESSING = "processing"


_DESCRIBE_ERROR_LABELS: dict[DescribeError, str] = {
    DescribeError.EMPTY: "EmptyResponse",
    DescribeError.TRUNCATED: "Truncated",
    DescribeError.API: "APIError",
    DescribeError.TIMEOUT: "Timeout",
}

_CAPTION_ERROR_LABELS: dict[CaptionError, str] = {
    CaptionError.EMPTY: "EmptyResponse",
    CaptionError.TRUNCATED: "Truncated",
    CaptionError.META_LEAK: "MetaLeak",
    CaptionError.TOO_SHORT: "TooShort",
    CaptionError.TOO_LONG: "TooLong",
    CaptionError.API: "APIError",
}

DESCRIBE_FAILURE_PREFIX = "Failed to describe video:"
CAPTION_FAILURE_PREFIX = "Failed to caption:"
PROCESS_FAILURE_PREFIX = "Failed to process video:"


def describe_error_from_reason(reason: str) -> DescribeError:
    if reason in ("EmptyResponse", "empty"):
        return DescribeError.EMPTY
    if reason in ("Truncated", "truncated", "InvalidJSON"):
        return DescribeError.TRUNCATED
    if "timeout" in reason.lower():
        return DescribeError.TIMEOUT
    return DescribeError.API


def caption_error_from_reason(reason: str) -> CaptionError:
    mapping: dict[str, CaptionError] = {
        "EmptyResponse": CaptionError.EMPTY,
        "Truncated": CaptionError.TRUNCATED,
        "MetaLeak": CaptionError.META_LEAK,
        "TooShort": CaptionError.TOO_SHORT,
        "TooLong": CaptionError.TOO_LONG,
    }
    return mapping.get(reason, CaptionError.API)


@dataclass
class DescribeResult:
    text: str | None
    error: DescribeError | None
    error_detail: str = ""
    attempts: int = 0
    total_ms: float = 0.0
    raw_json: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)

    def failure_label(self) -> str:
        if self.error_detail:
            return self.error_detail
        if self.error is not None:
            return _DESCRIBE_ERROR_LABELS.get(self.error, self.error.value)
        return "Unknown"

    def to_failure_string(self) -> str:
        return f"{DESCRIBE_FAILURE_PREFIX} {self.failure_label()}"


@dataclass
class CaptionResult:
    text: str | None
    error: CaptionError | None
    error_detail: str = ""
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)

    def failure_label(self) -> str:
        if self.error_detail:
            return self.error_detail
        if self.error is not None:
            return _CAPTION_ERROR_LABELS.get(self.error, self.error.value)
        return "Unknown"

    def to_failure_string(self) -> str:
        return f"{CAPTION_FAILURE_PREFIX} {self.failure_label()}"


def process_failure_string(error: ProcessError, *, detail: str = "") -> str:
    if error == ProcessError.INVALID_TASK:
        return "Invalid task input."
    if error == ProcessError.UNSUPPORTED_STYLE:
        return "Unsupported style requested."
    label = detail or error.value
    return f"{PROCESS_FAILURE_PREFIX} {label}"
