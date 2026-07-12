"""Best-of-N caption selection using the regex structural scorer."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from openai import OpenAI

from src.caption import (
    STYLES,
    _FRIENDLY_CAPTION_FAILURE,
    generate_styled_caption_from_text,
    public_caption_result,
    resolve_style_temperature,
)
from src.env import get_float_env, get_int_env
from src.results import CaptionResult
from src.scoring import score_caption
from src.caption_grounding import grounding_bonus


@dataclass(frozen=True)
class CaptionCandidate:
    text: str
    model: str
    label: str
    result: CaptionResult
    score_rank: int
    score_reason: str


def pool_candidate_temperature(style: str, index: int) -> float | None:
    """Candidate 0 uses style default; later candidates get +CAPTION_POOL_TEMP_DELTA."""
    if index <= 0:
        return None
    delta = get_float_env("CAPTION_POOL_TEMP_DELTA", 0.15)
    if delta <= 0:
        return None
    base = resolve_style_temperature(style)
    return min(base + delta, 0.97)


def is_friendly_placeholder(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    for message in _FRIENDLY_CAPTION_FAILURE.values():
        if stripped == message:
            return True
    return False


def rank_caption(
    text: str,
    style: str,
    *,
    description: str | None = None,
) -> tuple[int, str]:
    if is_friendly_placeholder(text):
        return 0, "friendly-placeholder"
    ok, reason = score_caption(text, style)
    if ok:
        words = len(text.split())
        # Prefer captions near the prompt target (50 words), not ultra-short.
        length_bonus = max(0, 10 - abs(words - 42) // 3)
        rank = 100 + length_bonus
        if description:
            g_bonus, g_reason = grounding_bonus(description, text)
            rank += g_bonus
            return rank, f"ok+{g_reason}"
        return rank, "ok"
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


def select_best_candidate(
    candidates: list[CaptionCandidate],
    *,
    style: str = "",
    description: str | None = None,
    client: OpenAI | None = None,
    task_id: str = "tiebreak",
) -> CaptionCandidate | None:
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda c: (c.score_rank, c.result.ok, len(c.text.split())),
        reverse=True,
    )
    best = ranked[0]
    if (
        len(ranked) < 2
        or get_int_env("CAPTION_JUDGE_TIEBREAK", 0) != 1
        or not description
        or not client
        or not style
    ):
        return best

    margin = get_int_env("CAPTION_TIEBREAK_MARGIN", 8)
    runner_up = ranked[1]
    if best.score_rank - runner_up.score_rank > margin:
        return best
    if best.score_rank < 100 or runner_up.score_rank < 100:
        return best

    judge_model = os.environ.get(
        "JUDGE_MODEL",
        os.environ.get("CAPTION_MODEL", "accounts/fireworks/models/deepseek-v4-flash"),
    )
    from src.llm_judge import judge_tiebreak_pick

    pick = judge_tiebreak_pick(
        client=client,
        model=judge_model,
        task_id=task_id,
        style=style,
        description=description,
        left_caption=best.text,
        right_caption=runner_up.text,
    )
    if pick == 1:
        return runner_up
    return best


def generate_candidate(
    *,
    client: OpenAI,
    model: str,
    label: str,
    style: str,
    description: str,
    diversity_retry: bool = False,
    temperature_override: float | None = None,
    judge_feedback: str | None = None,
) -> CaptionCandidate:
    result = generate_styled_caption_from_text(
        client=client,
        model=model,
        style=style,
        description=description,
        diversity_retry=diversity_retry,
        temperature_override=temperature_override,
        judge_feedback=judge_feedback,
    )
    text = public_caption_result(result, style=style)
    rank, reason = rank_caption(text, style, description=description)
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
    diversity_retry: bool = False,
    task_id: str = "tiebreak",
    judge_feedback: str | None = None,
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
            diversity_retry=diversity_retry,
            judge_feedback=judge_feedback,
        )

    candidates: list[CaptionCandidate] = []

    def _one(index: int, entry: tuple[str, str]) -> CaptionCandidate:
        label, model = entry
        return generate_candidate(
            client=client,
            model=model,
            label=label,
            style=style,
            description=description,
            diversity_retry=diversity_retry,
            temperature_override=pool_candidate_temperature(style, index),
            judge_feedback=judge_feedback,
        )

    indexed = list(enumerate(models))
    if parallel:
        with ThreadPoolExecutor(max_workers=min(len(models), 4)) as pool:
            futures = [pool.submit(_one, index, entry) for index, entry in indexed]
            for fut in as_completed(futures):
                candidates.append(fut.result())
    else:
        for index, entry in indexed:
            candidates.append(_one(index, entry))

    best = select_best_candidate(
        candidates,
        style=style,
        description=description,
        client=client,
        task_id=task_id,
    )
    assert best is not None
    return best
