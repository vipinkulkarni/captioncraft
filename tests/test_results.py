"""Tests for typed pipeline results and public rendering."""

import pytest

from src.caption import (
    STYLES,
    public_caption_result,
    public_describe_result,
    public_process_failure,
)
from src.results import (
    CaptionError,
    CaptionResult,
    DescribeError,
    DescribeResult,
    ProcessError,
    caption_error_from_reason,
    describe_error_from_reason,
)


class TestErrorMapping:
    def test_describe_error_from_reason(self):
        assert describe_error_from_reason("EmptyResponse") == DescribeError.EMPTY
        assert describe_error_from_reason("Truncated") == DescribeError.TRUNCATED
        assert describe_error_from_reason("APITimeoutError") == DescribeError.TIMEOUT
        assert describe_error_from_reason("RateLimitError") == DescribeError.API

    def test_caption_error_from_reason(self):
        assert caption_error_from_reason("MetaLeak") == CaptionError.META_LEAK
        assert caption_error_from_reason("TooShort") == CaptionError.TOO_SHORT
        assert caption_error_from_reason("UnknownThing") == CaptionError.API


class TestTypedPublicCaptions:
    @pytest.mark.parametrize("style", STYLES)
    def test_describe_result_failure_string(self, monkeypatch, style):
        monkeypatch.setenv("FRIENDLY_FAILURES", "0")
        result = DescribeResult(
            text=None,
            error=DescribeError.EMPTY,
            error_detail="EmptyResponse",
            attempts=2,
            total_ms=1200.0,
        )
        assert public_describe_result(result, style=style) == result.to_failure_string()

    @pytest.mark.parametrize("style", STYLES)
    def test_caption_result_failure_string(self, monkeypatch, style):
        monkeypatch.setenv("FRIENDLY_FAILURES", "0")
        result = CaptionResult(
            text=None,
            error=CaptionError.META_LEAK,
            error_detail="MetaLeak",
            attempts=2,
        )
        assert public_caption_result(result, style=style) == result.to_failure_string()

    @pytest.mark.parametrize("style", STYLES)
    def test_process_failure_string(self, monkeypatch, style):
        monkeypatch.setenv("FRIENDLY_FAILURES", "0")
        text = public_process_failure(
            ProcessError.PROCESSING,
            style=style,
            detail="RuntimeError",
        )
        assert text == "Failed to process video: RuntimeError"

    @pytest.mark.parametrize("style", STYLES)
    def test_typed_friendly_describe(self, monkeypatch, style):
        monkeypatch.setenv("FRIENDLY_FAILURES", "1")
        result = DescribeResult(
            text=None,
            error=DescribeError.EMPTY,
            error_detail="EmptyResponse",
        )
        text = public_describe_result(result, style=style)
        assert not text.startswith("Failed to describe video:")
