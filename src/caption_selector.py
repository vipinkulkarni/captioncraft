"""Best-of-N caption selection using the regex structural scorer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from openai import OpenAI

from src.caption import (
    STYLES,
    _FRIENDLY_CAPTION_FAILURE,
    generate_styled_caption_from_text,
    public_caption_result,
)
from src.results import CaptionResult
from src.scoring import score_caption


@dataclass(frozen=True)
class CaptionCandidate:
    text: str
    model: str
    label: str
    result: CaptionResult
    score_rank: int
    score_reason: str


def is_friendly_placeholder(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    for message in _FRIENDLY_CAPTION_FAILURE.values():
        if stripped == message:
            return True
    return False


def rank_caption(text: str, style: str) -> tuple[int, str]:
    if is_friendly_placeholder(text):
        return 0, "friendly-placeholder"
    ok, reason = score_caption(text, style)
    if ok:
        words = len(text.split())
        # Prefer captions near the prompt target (50 words), not ultra-short.
        length_bonus = max(0, 10 - abs(words - 42) // 3)
        return 100 + length_bonus, "ok"
    penalties = {
        "too-long": 45,
        "meta-leak": 25,
        "truncated": 30,
        "too-short": 20,
        "no-tech-joke": 35,
        "tech-jargon": 35,
        "neutral-copy": 15,
        "error": 0,
        "placeholder": 0,
        "empty": 0,
    }
    return penalties.get(reason, 10), reason


def select_best_candidate(candidates: list[CaptionCandidate]) -> CaptionCandidate | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda c: (c.score_rank, c.result.ok, len(c.text.split())),
    )


def generate_candidate(
    *,
    client: OpenAI,
    model: str,
    label: str,
    style: str,
    description: str,
) -> CaptionCandidate:
    result = generate_styled_caption_from_text(
        client=client,
        model=model,
        style=style,
        description=description,
    )
    text = public_caption_result(result, style=style)
    rank, reason = rank_caption(text, style)
    return CaptionCandidate(
        text=text,
        model=model,
        label=label,
        result=result,
        score_rank=rank,
        score_reason=reason,
    )


def generate_best_of_n_caption(
    *,
    client: OpenAI,
    models: list[tuple[str, str]],
    style: str,
    description: str,
    parallel: bool = True,
) -> CaptionCandidate:
    if style not in STYLES:
        raise ValueError(f"unsupported style: {style}")
    if not models:
        raise ValueError("models must not be empty")
    if len(models) == 1:
        return generate_candidate(
            client=client,
            model=models[0][1],
            label=models[0][0],
            style=style,
            description=description,
        )

    candidates: list[CaptionCandidate] = []

    def _one(entry: tuple[str, str]) -> CaptionCandidate:
        label, model = entry
        return generate_candidate(
            client=client,
            model=model,
            label=label,
            style=style,
            description=description,
        )

    if parallel:
        with ThreadPoolExecutor(max_workers=min(len(models), 4)) as pool:
            futures = [pool.submit(_one, entry) for entry in models]
            for fut in as_completed(futures):
                candidates.append(fut.result())
    else:
        for entry in models:
            candidates.append(_one(entry))

    best = select_best_candidate(candidates)
    assert best is not None
    return best
