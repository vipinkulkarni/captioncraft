"""Baseline unit tests for pure pipeline helpers."""

from src.pipeline import _frame_indices, _frame_times_ms, _is_bad_download_content_type


class TestFrameIndices:
    def test_returns_requested_count_for_long_video(self):
        indices = _frame_indices(100, 8)
        assert len(indices) == 8
        assert indices == sorted(indices)
        assert all(0 <= i <= 99 for i in indices)

    def test_dedupes_when_fewer_frames_than_slots(self):
        indices = _frame_indices(3, 8)
        assert indices == sorted(set(indices))
        assert all(0 <= i <= 2 for i in indices)

    def test_single_slot(self):
        assert _frame_indices(100, 1) == [0]

    def test_zero_frames(self):
        assert _frame_indices(0, 8) == []


class TestFrameTimesMs:
    def test_evenly_spaced_non_negative(self):
        times = _frame_times_ms(30.0, 8)
        assert len(times) == 8
        assert times == sorted(times)
        assert all(t >= 0 for t in times)
        assert times[0] == 0.0
        assert times[-1] == 30000.0

    def test_dedupes_on_very_short_clip(self):
        times = _frame_times_ms(0.05, 8)
        assert times == sorted(set(times))
        assert all(t >= 0 for t in times)

    def test_single_slot(self):
        assert _frame_times_ms(12.5, 1) == [0.0]

    def test_zero_duration(self):
        assert _frame_times_ms(0.0, 8) == []


class TestContentTypeDenylist:
    @staticmethod
    def _bad_types():
        return [
            "text/html",
            "text/html; charset=utf-8",
            "application/json",
            "text/plain",
            "application/xml",
            "text/xml",
        ]

    @staticmethod
    def _ok_types():
        return [
            "",
            "video/mp4",
            "application/octet-stream",
            "binary/octet-stream",
        ]

    def test_rejects_html_and_json(self):
        for ct in self._bad_types():
            assert _is_bad_download_content_type(ct) is True

    def test_allows_video_and_missing(self):
        for ct in self._ok_types():
            assert _is_bad_download_content_type(ct) is False
