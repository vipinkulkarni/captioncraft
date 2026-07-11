"""Heuristic grounding score: caption coverage of describe facts."""

from __future__ import annotations

import json
import re

from src.describe_schema import _extract_json_object, _validate_payload

_ACTION_LINE_PREFIXES = (
    "actions (early):",
    "actions (late):",
    "notable moments:",
)

_ACTION_VERB_HINTS = frozenset(
    {
        "sit",
        "sits",
        "sitting",
        "stand",
        "stands",
        "standing",
        "walk",
        "walks",
        "walking",
        "run",
        "runs",
        "running",
        "fly",
        "flies",
        "flying",
        "swim",
        "swims",
        "swimming",
        "preen",
        "preens",
        "preening",
        "turn",
        "turns",
        "turning",
        "look",
        "looks",
        "looking",
        "gaze",
        "gazes",
        "gazing",
        "type",
        "types",
        "typing",
        "pan",
        "pans",
        "panning",
        "move",
        "moves",
        "moving",
        "hop",
        "hops",
        "hopping",
        "jump",
        "jumps",
        "jumping",
        "eat",
        "eats",
        "eating",
        "drink",
        "drinks",
        "drinking",
        "splash",
        "splashes",
        "splashing",
        "crash",
        "crashes",
        "crashing",
        "slide",
        "slides",
        "sliding",
        "crouch",
        "crouches",
        "crouching",
        "stretch",
        "stretches",
        "stretching",
        "blink",
        "blinks",
        "blinking",
        "wag",
        "wags",
        "wagging",
        "perch",
        "perches",
        "perching",
        "land",
        "lands",
        "landing",
        "takeoff",
        "depart",
        "departs",
        "departing",
    }
)


def _action_text_chunks(description: str) -> list[str]:
    chunks: list[str] = []
    for line in description.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        for prefix in _ACTION_LINE_PREFIXES:
            if lower.startswith(prefix):
                chunks.append(stripped.split(":", 1)[-1])
                break
    return chunks


def extract_action_verbs(description: str) -> set[str]:
    """Pull likely action verbs from describe action lines."""
    verbs: set[str] = set()
    for chunk in _action_text_chunks(description):
        for token in _tokenize(chunk):
            if token in _ACTION_VERB_HINTS or token.endswith("ing"):
                verbs.add(token)
    return verbs


def _action_coverage_bonus(desc_verbs: set[str], caption_tokens: set[str]) -> int:
    if not desc_verbs:
        return 0
    matched = {v for v in desc_verbs if v in caption_tokens}
    return int(round(len(matched) / len(desc_verbs) * 10))


def _verb_mismatch_penalty(desc_verbs: set[str], caption_tokens: set[str]) -> int:
    if not desc_verbs:
        return 0
    invented = {v for v in caption_tokens if v in _ACTION_VERB_HINTS and v not in desc_verbs}
    return min(len(invented) * 3, 8)


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
    """Return 0–35 bonus points for covering describe anchors and actions."""
    anchors = extract_description_anchors(description)
    if not anchors:
        return 0, "no-anchors"

    caption_tokens = _tokenize(caption)
    if not caption_tokens:
        return 0, "empty-caption"

    matched = {a for a in anchors if a in caption_tokens}
    coverage = len(matched) / len(anchors)
    bonus = int(round(coverage * 20))
    if len(matched) >= 3:
        bonus += 5
    if len(matched) >= 5:
        bonus += 5

    desc_verbs = extract_action_verbs(description)
    action_bonus = _action_coverage_bonus(desc_verbs, caption_tokens)
    mismatch_penalty = _verb_mismatch_penalty(desc_verbs, caption_tokens)
    bonus += action_bonus - mismatch_penalty
    bonus = max(0, min(bonus, 35))

    parts = [f"grounding={len(matched)}/{len(anchors)}"]
    if desc_verbs:
        matched_verbs = len(desc_verbs & caption_tokens)
        parts.append(f"actions={matched_verbs}/{len(desc_verbs)}")
    if mismatch_penalty:
        parts.append(f"mismatch=-{mismatch_penalty}")
    return bonus, "+".join(parts)
