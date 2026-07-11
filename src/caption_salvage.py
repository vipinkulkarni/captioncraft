"""Extract and compress captions from leaky or over-long model output."""

from __future__ import annotations

import json
import re
from typing import Callable

from openai import OpenAI

from src.env import get_int_env

_META_LEAK_PREFIXES = (
    "we need to",
    "the user wants",
    "the user asks",
    "i need to",
    "i will",
    "here is",
    "here's",
    "caption:",
    "final caption:",
)

_LABEL_PREFIXES = (
    "here's the caption:",
    "here is the caption:",
    "final caption:",
    "caption:",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_STYLE_TARGET_WORDS = 50

_DRAFTING_PREFIXES = (
    "better:",
    "count words:",
    "check word count:",
    "check:",
    "possible caption:",
    "possible metaphor:",
    "let's craft:",
    "let's go with",
    "let's write:",
    "but need to",
    "but we need to",
    "punchline:",
    "actually:",
    "that uses ",
    "drafting",
    "we can adapt",
    "we can use",
    "first option",
    "that's a ",
    "could say ",
    "i'll try:",
    "alternative:",
    "alternatively:",
    "alternatively,",
    "use \"",
    "use '",
    "actions:",
    "action:",
    "final:",
    "then it adds",
    "mention ",
)

_DRAFTING_MARKERS = (
    "count words:",
    "check word count:",
    "includes colors:",
    "under 50 words",
    "match setting",
    "no emojis",
    "drafting the caption",
    "[later action",
    "let me count:",
    "is dev reference",
    "merge conflict",
    "might work",
    "keep it natural",
    "first sentence",
    "second sentence",
    "punchline exagger",
    "total 27",
    "also mention",
    "types might not",
    "might not be accurate",
)

_FRAGMENT_TAIL_RE = re.compile(
    r"\b(?:the|and|in|to|a|an|or|but|like)\.?$",
    re.IGNORECASE,
)
_IN_THE_TAIL_RE = re.compile(r"\bin the\s*\.?$", re.IGNORECASE)

_NUMBERED_PARAM_RE = re.compile(r"\w+\(\d+\)")
_BULLET_LINE_RE = re.compile(r"^\s*[-*•]\s", re.MULTILINE)
_NUMBERED_LIST_RE = re.compile(r"^\s*\d+\.\s", re.MULTILINE)
_INCOMPLETE_END_RE = re.compile(
    r"(?:"
    r",\s*\.|"
    r",\s*but\s*\.|"
    r"\(\s*\.|"
    r"\(\d+\s*\.|"
    r"\b(?:like a|lands on|the incoming|awaiting merge|are the|and late)\s*\.?"
    r")$",
    re.IGNORECASE,
)
_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


def is_drafting_junk(text: str) -> bool:
    """Reject planning notes, checklists, and fragment tails masquerading as captions."""
    stripped = text.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if any(lower.startswith(p) for p in _DRAFTING_PREFIXES):
        return True
    if any(m in lower for m in _DRAFTING_MARKERS):
        return True
    if _BULLET_LINE_RE.search(stripped) or _NUMBERED_LIST_RE.search(stripped):
        return True
    if len(_NUMBERED_PARAM_RE.findall(stripped)) >= 2:
        return True
    if _INCOMPLETE_END_RE.search(stripped):
        return True
    if _IN_THE_TAIL_RE.search(stripped):
        return True
    if _MARKDOWN_HEADER_RE.search(stripped):
        return True
    if "**analyze" in lower or "**draft" in lower:
        return True
    if stripped.count('"') % 2 == 1:
        return True
    checklist_tags = sum(
        1 for tag in ("(color)", "(action)", "(setting)", "(surface)", "(background)")
        if tag in lower
    )
    if checklist_tags >= 2:
        return True
    parts = _sentence_parts(stripped)
    if parts:
        tail_words = parts[-1].lower().strip().rstrip(".!?")
        if _FRAGMENT_TAIL_RE.match(tail_words):
            return True
    if len(parts) >= 2 and len(parts[-1].split()) <= 3:
        tail = parts[-1].lower()
        if tail.startswith(("the ", "and ", "but ", "or ", "a ", "an ")):
            return True
    return False


def _sentence_parts(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    out: list[str] = []
    for part in parts:
        if part[-1] not in ".!?":
            part = part + "."
        out.append(part)
    return out


def _strip_meta_lines(text: str) -> str | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kept = [
        ln
        for ln in lines
        if not any(ln.lower().startswith(p) for p in _META_LEAK_PREFIXES)
    ]
    if not kept:
        return None
    merged = " ".join(kept).strip()
    return merged if merged else None


def _strip_label_prefix(text: str) -> str | None:
    stripped = text.strip()
    lower = stripped.lower()
    for prefix in _LABEL_PREFIXES:
        if lower.startswith(prefix):
            rest = stripped[len(prefix) :].lstrip()
            return rest if rest else None
    return None


def _paragraph_tails(text: str) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    if len(blocks) <= 1:
        return []
    return list(reversed(blocks))


def _sentence_tail_candidates(text: str, *, max_sentences: int = 3) -> list[str]:
    parts = _sentence_parts(text)
    if not parts:
        return []
    out: list[str] = []
    for n in range(1, min(max_sentences, len(parts)) + 1):
        out.append(" ".join(parts[-n:]))
    return out


def _json_caption_candidate(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                cap = payload.get("caption")
                if isinstance(cap, str) and cap.strip():
                    return cap.strip()
        except json.JSONDecodeError:
            pass
    start = stripped.find('{"caption"')
    if start < 0:
        start = stripped.find('{"caption":')
    if start >= 0:
        end = stripped.rfind("}")
        if end > start:
            try:
                payload = json.loads(stripped[start : end + 1])
                if isinstance(payload, dict):
                    cap = payload.get("caption")
                    if isinstance(cap, str) and cap.strip():
                        return cap.strip()
            except json.JSONDecodeError:
                pass
    return None


def iter_salvage_candidates(raw: str) -> list[str]:
    """Ordered candidates from raw model output (cheapest heuristics first)."""
    text = raw.strip()
    if not text:
        return []

    seen: set[str] = set()
    ordered: list[str] = []

    def add(candidate: str | None) -> None:
        if not candidate:
            return
        norm = candidate.strip()
        if not norm or norm in seen:
            return
        seen.add(norm)
        ordered.append(norm)

    add(_json_caption_candidate(text))
    add(_strip_label_prefix(text))
    add(_strip_meta_lines(text))
    for block in _paragraph_tails(text):
        add(block)
        add(_strip_label_prefix(block))
        add(_strip_meta_lines(block))
        for tail in _sentence_tail_candidates(block):
            add(tail)
    for tail in _sentence_tail_candidates(text):
        add(tail)

    return ordered


def pick_valid_candidate(
    raw: str,
    *,
    style: str,
    is_valid: Callable[[str, str], tuple[bool, str]],
) -> tuple[str | None, str]:
    for candidate in iter_salvage_candidates(raw):
        bad, reason = is_valid(candidate, style=style)
        if not bad:
            return candidate, ""
    return None, ""


def pick_two_sentence_fit(
    raw: str,
    *,
    style: str,
    hard_limit: int,
    is_valid: Callable[[str, str], tuple[bool, str]],
) -> tuple[str | None, str]:
    """Prefer the longest valid 1–2 sentence tail under the word hard limit."""
    best: str | None = None
    best_words = 0
    for candidate in iter_salvage_candidates(raw):
        for tail in _sentence_tail_candidates(candidate, max_sentences=2):
            words = len(tail.split())
            if words > hard_limit:
                continue
            bad, _ = is_valid(tail, style=style)
            if bad:
                continue
            if words > best_words:
                best = tail
                best_words = words
    if best:
        return best, ""
    return None, ""


_COMPRESS_USER = (
    "Rewrite the draft below as exactly 2 short sentences, under {target} words total. "
    "Keep the same tone, humor, and scene facts. Output ONLY the rewritten caption.\n\n"
    "Draft:\n{draft}"
)


def compress_caption_call(
    *,
    client: OpenAI,
    model: str,
    style: str,
    draft: str,
    system_prompt: str,
    temperature: float = 0.15,
    max_tokens: int = 120,
) -> str:
    if not get_int_env("STYLE_COMPRESS_ON_LONG", 1):
        return ""
    user = _COMPRESS_USER.format(draft=draft.strip(), target=_STYLE_TARGET_WORDS)
    request_kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if get_int_env("STYLE_JSON_MODE", 0):
        request_kwargs["response_format"] = {"type": "json_object"}
        user = (
            f"{user}\n\nReply with JSON only: "
            '{"caption":"your rewritten caption here"}'
        )
        request_kwargs["messages"][1]["content"] = user
    resp = client.chat.completions.create(**request_kwargs)
    return (resp.choices[0].message.content or "").strip()
