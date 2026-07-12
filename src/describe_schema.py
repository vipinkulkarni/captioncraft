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
    camera: str = ""
    notable_moments: list[str] = field(default_factory=list)
    on_screen_text: list[str] = field(default_factory=list)

    def to_style_context(self) -> str:
        # Lead with subjects so captions lock onto visual focus, not background traffic.
        lines: list[str] = []
        for index, subject in enumerate(self.subjects, start=1):
            name = str(subject.get("name", "")).strip() or f"subject {index}"
            colors = [str(c).strip() for c in (subject.get("colors") or []) if str(c).strip()]
            distinguishing = [
                str(d).strip() for d in (subject.get("distinguishing") or []) if str(d).strip()
            ]
            label = "Primary subject" if index == 1 else f"Subject {index}"
            part = f"{label}: {name}"
            if colors:
                part += f" (colors: {', '.join(colors)})"
            if distinguishing:
                part += f" [{', '.join(distinguishing)}]"
            lines.append(part)
        lines.extend(
            [
                f"Setting: {self.setting}",
                f"Actions (early): {self.actions_early}",
                f"Actions (late): {self.actions_late}",
            ]
        )
        if self.camera:
            lines.append(f"Camera: {self.camera}")
        if self.background:
            lines.append(f"Background: {self.background}")
        if self.on_screen_text:
            lines.append("On-screen text: " + "; ".join(self.on_screen_text))
        if self.notable_moments:
            lines.append("Notable moments: " + "; ".join(self.notable_moments))
        lines.append(
            "Caption focus: lead with the Primary subject; treat other subjects "
            "and background as secondary context."
        )
        return "\n".join(lines)


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _fallback_subject_from_setting(setting: str) -> dict[str, Any]:
    """Synthesize a scene subject when the model omits subjects for scenery-only clips."""
    lower = setting.strip().lower()
    if "park" in lower:
        name = "park scene"
    elif any(word in lower for word in ("beach", "coast", "shore", "coastline")):
        name = "coastal scene"
    elif any(word in lower for word in ("ocean", "sea", "waves")):
        name = "ocean scene"
    elif "office" in lower:
        name = "office scene"
    elif any(word in lower for word in ("meadow", "field")):
        name = "meadow scene"
    elif "garden" in lower:
        name = "garden scene"
    else:
        clause = setting.split(",")[0].strip()
        for prefix in ("outdoor ", "indoor "):
            if clause.lower().startswith(prefix):
                clause = clause[len(prefix) :].strip()
        words = clause.split()
        name = " ".join(words[:4]).strip() or "scene"
        if "scene" not in name.lower():
            name = f"{name} scene"
    return {"name": name, "colors": [], "distinguishing": []}


def _validate_payload(data: Any) -> tuple[bool, str, VideoDescription | None]:
    if not isinstance(data, dict):
        return False, "InvalidJSON", None

    setting = str(data.get("setting", "")).strip()
    actions_early = str(data.get("actions_early", "")).strip()
    actions_late = str(data.get("actions_late", "")).strip()
    if not setting or not actions_early or not actions_late:
        return False, "InvalidJSON", None

    subjects_raw = data.get("subjects")
    if not isinstance(subjects_raw, list):
        return False, "InvalidJSON", None

    subjects: list[dict[str, Any]] = []
    for item in subjects_raw:
        if not isinstance(item, dict):
            return False, "InvalidJSON", None
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        subjects.append(
            {
                "name": name,
                "colors": _as_string_list(item.get("colors")),
                "distinguishing": _as_string_list(item.get("distinguishing")),
            }
        )

    if not subjects:
        subjects.append(_fallback_subject_from_setting(setting))

    description = VideoDescription(
        subjects=subjects,
        setting=setting,
        actions_early=actions_early,
        actions_late=actions_late,
        background=str(data.get("background", "")).strip(),
        camera=str(data.get("camera", "")).strip(),
        notable_moments=_as_string_list(data.get("notable_moments")),
        on_screen_text=_as_string_list(
            data.get("on_screen_text") or data.get("ocr") or data.get("readable_text")
        ),
    )
    return True, "", description


def _extract_json_object(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def parse_describe_json(raw: str) -> tuple[bool, str, str]:
    """Return (ok, failure_reason, formatted_style_context)."""
    text = _extract_json_object(raw)
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


def parse_video_description(raw: str) -> tuple[bool, str, VideoDescription | None]:
    """Return (ok, failure_reason, parsed description object)."""
    text = _extract_json_object(raw)
    if not text:
        return False, "EmptyResponse", None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False, "InvalidJSON", None

    return _validate_payload(data)
