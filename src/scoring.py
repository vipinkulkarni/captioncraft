"""Caption quality checks shared by unit tests and local eval scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.caption_salvage import is_drafting_junk

_TECH_JOKE_PATTERN = (
    r"\b(api|bugs?|deploy(?:s|ment|ing|ed)?|production|server|git|code|stack overflow|"
    r"debug(?:ging)?|runtime|ci|merge|packet|ddos|sprint|backend|frontend|cpu|gpu|"
    r"outage|hotfix(?:es|ed)?|rollback|staging|sandbox|exception|async|callback|sla|"
    r"refactor|syntax error|load balancing|try[- ]catch|pr\b|unhandled|"
    r"single[- ]threaded|multi[- ]threaded|cache[d]?|regression|qa|commit(?:s|ted)?|"
    r"database|flush(?:ing|ed)?|thread(?:s|ed|ing)?|bottleneck|payload|latency|"
    r"benchmark|pipeline|container|microservice|lint(?:er|ing)?|"
    r"deadlock|throughput|downtime|dev(?:ops)?|standup|retry(?:ing|ed)?|"
    r"for loop|non[- ]blocking|idle\b|loading|render(?:ing)?|loop\b|"
    r"oauth|kubernetes|k8s|npm|docker|vm\b|ssh|terraform|serverless|"
    r"bandwidth|firewall|proxy|segfault|compiler|transpil(?:e|er|ing)?|break)\b"
)
TECH_JOKE_MARKERS = re.compile(_TECH_JOKE_PATTERN, re.I)
TECH_JARGON_MARKERS = re.compile(
    _TECH_JOKE_PATTERN.replace("|staging|", "|").replace("|code|", "|"),
    re.I,
)
VALID_END_CHARS = ".!?)\""

STYLE_WORD_HARD_LIMIT: dict[str, int] = {
    "formal": 58,
    "humorous_non_tech": 62,
    "sarcastic": 68,
    "humorous_tech": 68,
}
DEFAULT_WORD_HARD_LIMIT = 72
SIMILARITY_WARN_THRESHOLD = 0.65
NEUTRAL_PREFIXES = (
    "the video shows",
    "the video takes place",
    "the video captures",
    "the scene is set",
    "failed to",
    "video caption (",
)

META_LEAK_MARKERS = (
    "we need to",
    "the user wants",
    "the user asks",
    "using only these facts",
    "must be 2 short sentences",
    "completely new wording",
    "write a funny",
    "write a dry",
    "write a polished",
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 2}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cross_style_similarity_warnings(captions: dict[str, str], task_id: str) -> list[str]:
    """Warn when two styles for the same clip reuse too much wording."""
    styles = sorted(captions)
    warnings: list[str] = []
    for i, s1 in enumerate(styles):
        words1 = _content_words(captions[s1])
        for s2 in styles[i + 1 :]:
            sim = _jaccard_similarity(words1, _content_words(captions[s2]))
            if sim >= SIMILARITY_WARN_THRESHOLD:
                warnings.append(f"{task_id}/{s1} vs {s2}: high similarity ({sim:.0%})")
    return warnings


def is_structural_failure(text: str) -> tuple[bool, str]:
    """Hard failures only — for skipping judge API calls, not quality scoring."""
    if not text or not text.strip():
        return True, "empty"
    if text.lower().startswith("failed to"):
        return True, "error"
    if text.lower().startswith("video caption ("):
        return True, "placeholder"
    if is_drafting_junk(text):
        return True, "meta-leak"
    return False, ""


def score_caption(text: str, style: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty"
    if text.lower().startswith("failed to"):
        return False, "error"
    if text.lower().startswith("video caption ("):
        return False, "placeholder"
    if is_drafting_junk(text):
        return False, "meta-leak"
    lower = text.lower()
    if any(lower.startswith(p) for p in NEUTRAL_PREFIXES):
        return False, "neutral-copy"
    if sum(1 for m in META_LEAK_MARKERS if m in lower) >= 2:
        return False, "meta-leak"
    if lower.startswith("the user ") or lower.startswith("we need to"):
        return False, "meta-leak"
    if text[-1] not in VALID_END_CHARS:
        return False, "truncated"
    words = text.split()
    if len(words) < 5:
        return False, "too-short"
    hard_limit = STYLE_WORD_HARD_LIMIT.get(style, DEFAULT_WORD_HARD_LIMIT)
    if len(words) > hard_limit:
        return False, "too-long"
    if style == "humorous_tech" and not TECH_JOKE_MARKERS.search(text):
        return False, "no-tech-joke"
    if style == "humorous_non_tech" and TECH_JARGON_MARKERS.search(text):
        return False, "tech-jargon"
    return True, "ok"


def score_file(path: Path) -> tuple[int, int, list[str], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    passes = 0
    total = 0
    fails: list[str] = []
    warnings: list[str] = []
    for task in data:
        tid = task["task_id"]
        captions = task["captions"]
        warnings.extend(cross_style_similarity_warnings(captions, tid))
        for style, caption in captions.items():
            total += 1
            ok, reason = score_caption(caption, style)
            if ok:
                passes += 1
            else:
                fails.append(f"{tid}/{style}: {reason}")
    return passes, total, fails, warnings
