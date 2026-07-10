"""Parse and format structured describe JSON for style captioning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoDescription:
    subjects: list[dict[str, Any]]
    setting: str
    actions_early: str
    actions_late: str
    background: str = ""
    notable_moments: list[str] = field(default_factory=list)

    def to_style_context(self) -> str:
        lines = [
            f"Setting: {self.setting}",
            f"Actions (early): {self.actions_early}",
            f"Actions (late): {self.actions_late}",
        ]
        if self.background:
            lines.append(f"Background: {self.background}")
        for index, subject in enumerate(self.subjects, start=1):
            name = str(subject.get("name", "")).strip() or f"subject {index}"
            colors = [str(c).strip() for c in (subject.get("colors") or []) if str(c).strip()]
            distinguishing = [
                str(d).strip() for d in (subject.get("distinguishing") or []) if str(d).strip()
            ]
            part = f"Subject {index}: {name}"
            if colors:
                part += f" (colors: {', '.join(colors)})"
            if distinguishing:
                part += f" [{', '.join(distinguishing)}]"
            lines.append(part)
        if self.notable_moments:
            lines.append("Notable moments: " + "; ".join(self.notable_moments))
        return "\n".join(lines)


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _validate_payload(data: Any) -> tuple[bool, str, VideoDescription | None]:
    if not isinstance(data, dict):
        return False, "InvalidJSON", None

    subjects_raw = data.get("subjects")
    if not isinstance(subjects_raw, list) or not subjects_raw:
        return False, "InvalidJSON", None

    subjects: list[dict[str, Any]] = []
    for item in subjects_raw:
        if not isinstance(item, dict):
            return False, "InvalidJSON", None
        name = str(item.get("name", "")).strip()
        if not name:
            return False, "InvalidJSON", None
        subjects.append(
            {
                "name": name,
                "colors": _as_string_list(item.get("colors")),
                "distinguishing": _as_string_list(item.get("distinguishing")),
            }
        )

    setting = str(data.get("setting", "")).strip()
    actions_early = str(data.get("actions_early", "")).strip()
    actions_late = str(data.get("actions_late", "")).strip()
    if not setting or not actions_early or not actions_late:
        return False, "InvalidJSON", None

    description = VideoDescription(
        subjects=subjects,
        setting=setting,
        actions_early=actions_early,
        actions_late=actions_late,
        background=str(data.get("background", "")).strip(),
        notable_moments=_as_string_list(data.get("notable_moments")),
    )
    return True, "", description


def parse_describe_json(raw: str) -> tuple[bool, str, str]:
    """Return (ok, failure_reason, formatted_style_context)."""
    text = (raw or "").strip()
    if not text:
        return False, "EmptyResponse", ""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False, "InvalidJSON", ""

    ok, reason, description = _validate_payload(data)
    if not ok or description is None:
        return False, reason or "InvalidJSON", ""
    return True, "", description.to_style_context()
