"""Extract and compress captions from leaky or over-long model output."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from openai import OpenAI

from src.env import get_int_env

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_DIR = _REPO_ROOT / "prompts"
_STYLE_PROMPT_NAMES = (
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
)
_EXAMPLE_CAPTION_RE = re.compile(
    r"(?im)^Example \(different scene.*?:\s*\n(?:Facts:.*\n)?Caption:\s*(.+)$"
)
_CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "its",
        "it",
        "this",
        "that",
        "into",
        "over",
        "under",
        "through",
        "like",
        "then",
        "than",
        "primary",
        "subject",
        "colors",
        "scene",
    }
)

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
    "write:",
    "example:",
    "example: \"",
    "possible caption:",
    "possible metaphor:",
    "possible angle:",
    "possibly:",
    "maybe:",
    "facts say",
    "but facts say",
    "kicker:",
    "relatable:",
    "like relatable",
    "let's craft:",
    "let's go with",
    "let's write:",
    "let's refine",
    "let's try:",
    "let's adjust",
    "but need to",
    "but we need to",
    "need to use",
    "need to follow",
    "need to match",
    "need to incorporate",
    "need to include",
    "but ensure",
    "keep simple:",
    "keep it simple:",
    "notable moments:",
    "absurd comparison:",
    "setting outdoor",
    "could say.",
    "punchline:",
    "actually:",
    "that uses ",
    "then undercut",
    "drafting",
    "we can adapt",
    "we can use",
    "we can include",
    "but we can include",
    "so we can",
    "first option",
    "that's a ",
    "could say ",
    "i'll try:",
    "alternative:",
    "alternatively:",
    "alternatively,",
    "the previous caption",
    "maybe another",
    "but careful",
    "revised:",
    "use \"",
    "use '",
    "use the given",
    "actions:",
    "action:",
    "final:",
    "then it adds",
    "mention ",
    "but must ",
    "but check",
    "but does it",
    "but we also",
    "we need ",
    "then punchline",
    "not necessary",
    "let's think",
    "also \"",
    "also uses",
    "metaphor:",
    "metaphor/joke",
    "metaphor/joke style",
    "another:",
    "shape:",
    "shape (do not",
    "tone:",
    "idea:",
    "note:",
    "draft:",
    "option:",
    "hook:",
    "concept:",
)

_DRAFTING_MARKERS = (
    "also need",
    "also need to",
    "could say",
    "could say ",
    "maybe \"",
    "maybe '",
    "maybe:",
    "possible angle",
    "facts say",
    "but facts say",
    "use \"",
    "use '",
    "as detail",
    "background:",
    "notable moments:",
    "actions (early)",
    "actions (late)",
    "primary subject:",
    "caption focus:",
    "on-screen text:",
    "reference at least",
    "at least one color",
    "count words:",
    "check word count:",
    "includes colors:",
    "under 50 words",
    "match setting",
    "no emojis",
    "drafting the caption",
    "[later action",
    "[subject]",
    "[visible action",
    "[punchline",
    "[relatable absurd",
    "metaphor/joke style",
    "shape: \"[",
    "let me count:",
    "is dev reference",
    '"merge conflict" is',
    "is a metaphor",
    "is metaphor",
    "doesn't introduce new",
    "does not introduce new",
    "don't invent",
    "do not invent",
    "maybe not needed",
    "let's try",
    "absurd comparisons are tone",
    "must not add new physical",
    "no inventing brunch",
    "props not in the facts",
    "closed world:",
    "keep it natural",
    "first sentence",
    "second sentence",
    "punchline exagger",
    "total 27",
    "also mention",
    "types might not",
    "might not be accurate",
    "punchline could",
    "could add flavor",
    "let's think",
    "let's adjust",
    "is deadpan",
    "must include color",
    "also uses \"",
    "then undercut",
    "but need to match",
    "but need to tie",
    "uses late action",
    "uses early action",
    "then punchline:",
    "foliage implied",
    "setting: indoor",
    "setting: outdoor",
    "actions: early",
    "actions: late",
    "maybe another angle",
    "different angle",
    "if it fits the joke",
    "fits the joke",
    "is a bit cliché",
    "is a bit cliche",
    "the previous caption",
    "we need a different",
    "not a physical object",
    "in scene, it's a comparison",
    "should be okay",
    "it's a comparison",
    "but careful",
    "don't invent",
    "do not invent",
    "use the given",
    "revised:",
    "punchline:",
)

# NOTE: excluded on purpose — legitimate caption tails exist for "in"
# ("soaking it all in."), "with" ("a force to be reckoned with."), "at"
# ("where it's at."), "her"/"his" (object/possessive pronouns), "is"
# ("that's just how it is."), "then" ("every now and then.").
_FRAGMENT_TAIL_RE = re.compile(
    r"\b(?:the|and|to|a|an|or|but|like|its|their|of|from|as)\.?$",
    re.IGNORECASE,
)

# Single-word evaluative sentences the model appends after judging its own
# draft, e.g. '"...caption..." Hmm.' or 'Punchline: absurd suggestions. Good.'
# Kept minimal: words like "Nice."/"Perfect." can be legit deadpan sarcasm.
_EVALUATIVE_TAILS = frozenset({"hmm", "good", "that works", "also", "that"})
# Same idea, but robust to a closing quote ending the previous sentence
# (the sentence splitter can't split on '."<space>').
_EVALUATIVE_TAIL_RE = re.compile(
    r"[.!?][\"'”’]?\s+(?:hmm+|good|that works|also|that)\s*[.!?]?$",
    re.IGNORECASE,
)

# A colon immediately followed by a period ("Let's adjust:.") only appears in
# truncated planning notes.
_COLON_PERIOD_RE = re.compile(r":\s*\.")
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
    r"\b(?:like a|lands on|the incoming|awaiting merge|are the|and late)\s*\.|"
    # Dangling simile / contraction cutoffs: "chops cucumber like they're."
    r"\blike\s+(?:they(?:'re| are)|it(?:'s| is)|he(?:'s| is)|she(?:'s| is)|we(?:'re| are))\s*\.|"
    r"\b(?:they(?:'re)|it(?:'s)|he(?:'s)|she(?:'s)|we(?:'re))\s*\."
    r")$",
    re.IGNORECASE,
)
_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


_DESCRIBE_FIELD_PREFIXES = (
    "background:",
    "notable moments:",
    "actions (early):",
    "actions (late):",
    "primary subject:",
    "subject 1:",
    "subject 2:",
    "setting:",
    "camera:",
    "caption focus:",
    "on-screen text:",
)

# Observed mid-token cutoffs that still look like finished sentences.
_TRUNCATED_END_STEMS = frozenset(
    {"mult", "monito", "displa", "keybo", "contin", "throug", "witho"}
)
_FRAGMENT_LAST_SENTENCE = frozenset(
    {
        "could",
        "maybe",
        "also",
        "but",
        "and",
        "or",
        "so",
        "then",
        "well",
        "wait",
        "yes",
        "no",
        "ok",
        "hmm",
        "example",
        "write",
        "draft",
        "note",
    }
)


def is_describe_field_dump(text: str) -> bool:
    """True when the caption is pasted describe-schema fields, not a caption."""
    lower = text.strip().lower()
    if not lower:
        return False
    if any(lower.startswith(p) for p in _DESCRIBE_FIELD_PREFIXES):
        return True
    # Multi-field dumps often appear mid-string after a weak opener.
    hits = sum(1 for p in _DESCRIBE_FIELD_PREFIXES if p in lower)
    return hits >= 2


def is_incomplete_caption(text: str) -> bool:
    """True for truncated / fragment captions. Finished 1–2 sentence lines are OK."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[-1] not in ".!?\"')":
        return True
    parts = _sentence_parts(stripped)
    if not parts:
        return True
    # Official Track 2 refs are often one punchy sentence; allow that when complete.
    if len(parts) == 1 and len(parts[0].split()) < 8:
        return True
    if _INCOMPLETE_END_RE.search(stripped):
        return True
    last = parts[-1].rstrip(".!?\"')")
    words = last.split()
    if not words:
        return True
    if len(words) == 1 and words[0].lower().strip("\"'") in _FRAGMENT_LAST_SENTENCE:
        return True
    end = re.sub(r"[^a-z0-9']+", "", words[-1].lower())
    if not end:
        return True
    return end in _TRUNCATED_END_STEMS


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
    if _COLON_PERIOD_RE.search(stripped):
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
        # search (not match): catches truncations anywhere in the final
        # sentence, e.g. "it flies off, leaving the water to."
        if _FRAGMENT_TAIL_RE.search(tail_words):
            return True
        if tail_words in _EVALUATIVE_TAILS:
            return True
    if _EVALUATIVE_TAIL_RE.search(stripped):
        return True
    if len(parts) >= 2 and len(parts[-1].split()) <= 3:
        tail = parts[-1].lower()
        if tail.startswith(("the ", "and ", "but ", "or ", "a ", "an ")):
            return True
    return False


def caption_hard_fail_reason(
    text: str, *, style: str = "", description: str = ""
) -> str:
    """Deterministic fail reason the LLM judge must not override. Empty if OK."""
    _ = style  # reserved for style-specific hard fails
    if not text or not text.strip():
        return "empty"
    if is_prompt_example_echo(text):
        return "meta-leak"
    if is_describe_field_dump(text):
        return "describe-dump"
    if is_drafting_junk(text):
        return "meta-leak"
    if is_incomplete_caption(text):
        return "incomplete"
    if description and scene_subject_mismatch(description, text):
        return "scene-mismatch"
    return ""


def _sentence_parts(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    out: list[str] = []
    for part in parts:
        if part[-1] not in ".!?":
            part = part + "."
        out.append(part)
    return out


def _normalize_caption_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" \"'")


# Historical few-shot captions (older prompt revisions) that still get echoed.
_HARDCODED_EXAMPLE_CAPTIONS = frozenset(
    {
        "the orange kitten marches through the foliage like it owns the lease. tail raised, it approaches the camera as if we should be honored.",
        "an orange kitten walks through green foliage toward the camera. its tail stays raised as it steps closer.",
        "the orange kitten advances through green foliage like a canary deploy. tail raised, it fills the frame — production is live.",
        "an orange kitten struts through the green foliage like a tiny vip late for brunch. tail raised, it fills the frame as if the camera owes it a close-up.",
    }
)


@lru_cache(maxsize=1)
def load_prompt_example_captions() -> frozenset[str]:
    """Captions shown as few-shot examples in style prompts — never ship these."""
    found: set[str] = set(_HARDCODED_EXAMPLE_CAPTIONS)
    for name in _STYLE_PROMPT_NAMES:
        path = _PROMPT_DIR / f"{name}.txt"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        in_example = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("example ("):
                in_example = True
                continue
            if in_example and stripped.lower().startswith("caption:"):
                key = _normalize_caption_key(stripped.split(":", 1)[1])
                if key:
                    found.add(key)
                in_example = False
    return frozenset(found)


def is_prompt_example_echo(text: str) -> bool:
    """True when the model pasted a style-prompt few-shot caption verbatim."""
    key = _normalize_caption_key(text)
    if not key:
        return False
    examples = load_prompt_example_captions()
    if key in examples:
        return True
    # Near-exact: allow tiny punctuation drift.
    for example in examples:
        if abs(len(key) - len(example)) > 12:
            continue
        if key in example or example in key:
            return True
    return False


def _content_tokens(text: str) -> set[str]:
    return {
        t
        for t in _CONTENT_TOKEN_RE.findall(text.lower())
        if len(t) > 2 and t not in _STOPWORDS
    }


def primary_subject_tokens(description: str) -> set[str]:
    for line in description.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("primary subject:") or lower.startswith("subject 1:"):
            body = stripped.split(":", 1)[-1]
            name = body.split("(", 1)[0].split("[", 1)[0]
            return _content_tokens(name)
    return set()


_ANIMAL_ENTITY_TOKENS = frozenset(
    {
        "kitten",
        "cat",
        "cats",
        "kitty",
        "puppy",
        "dog",
        "dogs",
        "rabbit",
        "bunny",
        "bird",
        "birds",
        "pigeon",
        "dove",
        "horse",
        "cow",
        "deer",
        "fox",
        "squirrel",
        "fish",
        "whale",
        "dolphin",
        "penguin",
        "duck",
        "goose",
        "chicken",
        "hen",
        "parrot",
        "owl",
        "bear",
        "lion",
        "tiger",
        "monkey",
        "ape",
        "goat",
        "sheep",
        "pig",
        "mouse",
        "rat",
        "hamster",
        "turtle",
        "snake",
        "lizard",
        "frog",
        "insect",
        "butterfly",
        "bee",
        "spider",
    }
)

# Treat close animal terms as the same entity for mismatch checks.
_ANIMAL_SYNONYMS: dict[str, frozenset[str]] = {
    "kitten": frozenset({"cat", "cats", "kitty"}),
    "kitty": frozenset({"cat", "cats", "kitten"}),
    "cat": frozenset({"kitten", "cats", "kitty"}),
    "cats": frozenset({"cat", "kitten", "kitty"}),
    "puppy": frozenset({"dog", "dogs"}),
    "dog": frozenset({"puppy", "dogs"}),
    "dogs": frozenset({"dog", "puppy"}),
    "bunny": frozenset({"rabbit"}),
    "rabbit": frozenset({"bunny"}),
}


def _expand_animal_synonyms(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for token in list(tokens):
        out |= _ANIMAL_SYNONYMS.get(token, frozenset())
    return out


def foreign_entity_leak(description: str, caption: str) -> bool:
    """True when caption asserts animals not present in scene facts."""
    desc = _expand_animal_synonyms(_content_tokens(description))
    cap = _content_tokens(caption)
    return bool((cap & _ANIMAL_ENTITY_TOKENS) - desc)


def scene_subject_mismatch(description: str, caption: str) -> bool:
    """True when the caption ignores the primary subject and barely grounds in facts.

    Catches catastrophic wrong-scene outputs (e.g. prompt-example kitten on a
    coding clip) without a vision call.
    """
    if not description.strip() or not caption.strip():
        return False
    if is_prompt_example_echo(caption):
        return True
    if foreign_entity_leak(description, caption):
        return True
    primary = _expand_animal_synonyms(primary_subject_tokens(description))
    cap = _expand_animal_synonyms(_content_tokens(caption))
    if not primary:
        return False
    if primary & cap:
        return False
    # No primary-subject overlap: allow only if the caption still hits several
    # other describe anchors (color/setting/action words from the facts).
    desc = _expand_animal_synonyms(_content_tokens(description))
    overlap = desc & cap
    return len(overlap) < 2


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


_QUOTED_SPAN_RE = re.compile(r'"([^"]{40,400})"')


def _quoted_span_candidates(text: str) -> list[str]:
    """Full captions the model quoted inside its own drafting notes.

    Leaks like 'Let\'s refine: "Ginger kitten sits ..." That uses.' usually
    contain the finished caption verbatim between double quotes. Only spans
    that look like complete sentences (start uppercase, end with terminal
    punctuation) are returned, longest first.
    """
    spans = []
    for match in _QUOTED_SPAN_RE.finditer(text):
        span = match.group(1).strip()
        if not span or not span[0].isupper():
            continue
        if span[-1] not in ".!?":
            continue
        if len(span.split()) < 8:
            continue
        spans.append(span)
    spans.sort(key=lambda s: len(s.split()), reverse=True)
    return spans


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
    for span in _quoted_span_candidates(text):
        add(span)
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
    "Rewrite the draft below as 1 punchy sentence or 2 short ones, under {target} words total. "
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
