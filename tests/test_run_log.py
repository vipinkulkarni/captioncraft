"""Tests for structured run logging."""

import json

from src import run_log


class TestRunLog:
    def test_emit_event_writes_json(self, monkeypatch, capsys):
        monkeypatch.setenv("STRUCTURED_LOGS", "1")
        monkeypatch.setenv("VERBOSE_LOGS", "0")
        run_log.emit_event({"stage": "complete", "task_id": "e01", "total_s": 1.2})
        captured = capsys.readouterr()
        line = captured.err.strip().splitlines()[-1]
        data = json.loads(line)
        assert data["stage"] == "complete"
        assert data["task_id"] == "e01"
        assert data["total_s"] == 1.2
        assert "ts" in data

    def test_human_logs_disabled(self, monkeypatch, capsys):
        monkeypatch.setenv("VERBOSE_LOGS", "0")
        run_log.log_human("hello human")
        assert capsys.readouterr().err == ""

    def test_structured_logs_disabled(self, monkeypatch, capsys):
        monkeypatch.setenv("STRUCTURED_LOGS", "0")
        run_log.emit_event({"stage": "complete", "task_id": "e01"})
        assert capsys.readouterr().err == ""
