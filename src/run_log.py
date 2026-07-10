"""Structured run logging (JSON lines) and optional human-readable stderr logs."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip() == "1"


def human_logs_enabled() -> bool:
    return _env_flag("VERBOSE_LOGS", "1")


def structured_logs_enabled() -> bool:
    return _env_flag("STRUCTURED_LOGS", "1")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_human(message: str) -> None:
    if human_logs_enabled():
        print(message, file=sys.stderr)


def emit_event(event: dict) -> None:
    if not structured_logs_enabled():
        return
    payload = {"ts": utc_now_iso(), **event}
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), file=sys.stderr)


def emit_config_event(**fields: object) -> None:
    emit_event({"stage": "config", **fields})
