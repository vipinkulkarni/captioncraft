"""Heuristic grounding score: caption coverage of describe facts."""

from __future__ import annotations

import json
import re

from src.describe_schema import _extract_json_object, _validate_payload

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "under",
        "along",
        "their",
        "there",
        "while",
        "then",
        "early",
        "later",
        "scene",
        "video",
        "clip",
        "visible",
        "clearly",
        "frames",
        "subject",
        "setting",
        "actions",
        "background",
        "camera",
        "notable",
        "moments",
        "colors",
    }
)


def _tokenize(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def _anchors_from_formatted(description: str) -> set[str]:
    anchors: set[str] = set()
    for line in description.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("subject "):
            body = stripped.split(":", 1)[-1].strip()
            name = body.split("(", 1)[0].strip()
            anchors |= _tokenize(name)
            color_match = re.search(r"\(colors:\s*([^)]+)\)", body, re.I)
            if color_match:
                for part in color_match.group(1).split(","):
                    anchors |= _tokenize(part)
            dist_match = re.search(r"\[([^\]]+)\]", body)
            if dist_match:
                anchors |= _tokenize(dist_match.group(1))
        elif lower.startswith("background:"):
            anchors |= _tokenize(stripped.split(":", 1)[-1])
        elif lower.startswith("notable moments:"):
            anchors |= _tokenize(stripped.split(":", 1)[-1])
        elif lower.startswith("setting:"):
            anchors |= _tokenize(stripped.split(":", 1)[-1])
    return {a for a in anchors if len(a) > 2}


def _anchors_from_raw_json(raw: str) -> set[str]:
    text = _extract_json_object(raw)
    if not text:
        return set()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return set()
    ok, _, description = _validate_payload(data)
    if not ok or description is None:
        return set()
    return _anchors_from_formatted(description.to_style_context())


def extract_description_anchors(description: str) -> set[str]:
    """Pull matchable tokens from structured describe JSON or formatted context."""
    text = (description or "").strip()
    if not text:
        return set()
    if text.startswith("{"):
        anchors = _anchors_from_raw_json(text)
        if anchors:
            return anchors
    return _anchors_from_formatted(text)


def grounding_bonus(description: str, caption: str) -> tuple[int, str]:
    """Return 0–35 bonus points for covering describe anchors in the caption."""
    anchors = extract_description_anchors(description)
    if not anchors:
        return 0, "no-anchors"

    caption_tokens = _tokenize(caption)
    if not caption_tokens:
        return 0, "empty-caption"

    matched = {a for a in anchors if a in caption_tokens}
    coverage = len(matched) / len(anchors)
    bonus = int(round(coverage * 25))
    if len(matched) >= 3:
        bonus += 5
    if len(matched) >= 5:
        bonus += 5
    bonus = min(bonus, 35)
    return bonus, f"grounding={len(matched)}/{len(anchors)}"
