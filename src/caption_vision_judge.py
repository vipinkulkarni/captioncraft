"""Vision-backed caption accuracy (caption vs video frames)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.caption import _to_data_url
from src.env import get_float_env, get_int_env
from src.llm_clients import fireworks_extra_body, is_google_ai_model, resolve_llm_client

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_PATH = _REPO_ROOT / "prompts" / "judge_caption_accuracy.txt"

_DEFAULT_VISION_JUDGE = "accounts/fireworks/models/minimax-m3"
_ACCURACY_RE = re.compile(r'"accuracy"\s*:\s*([0-9]*\.?[0-9]+)', re.I)
_ISSUE_RE = re.compile(r'"issue"\s*:\s*"(.*?)"', re.I | re.S)


@dataclass
class CaptionVisionAccuracy:
    accuracy: float
    issue: str = ""
    judge_model: str = ""
    parse_error: str = ""

    @property
    def ok(self) -> bool:
        return not self.parse_error


def resolve_caption_vision_judge_model() -> str:
    return (
        os.environ.get("CAPTION_VISION_JUDGE_MODEL", "").strip()
        or _DEFAULT_VISION_JUDGE
    )


def caption_vision_accuracy_enabled() -> bool:
    """Pipeline flag: only on when explicitly enabled after composition-gap diagnostic."""
    return get_int_env("CAPTION_VISION_ACCURACY", 0) == 1


def load_judge_caption_accuracy_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def _extract_json_object(raw: str) -> str:
    text = raw.strip()
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


def _parse_unit_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return round(score, 3)


def parse_caption_vision_accuracy_response(
    raw: str,
    *,
    judge_model: str = "",
) -> CaptionVisionAccuracy:
    if not (raw or "").strip():
        return CaptionVisionAccuracy(
            accuracy=0.0,
            judge_model=judge_model,
            parse_error="EmptyResponse",
        )
    try:
        payload = json.loads(_extract_json_object(raw))
        if isinstance(payload, dict):
            accuracy = _parse_unit_score(payload.get("accuracy"))
            if accuracy is not None:
                return CaptionVisionAccuracy(
                    accuracy=accuracy,
                    issue=str(payload.get("issue") or "").strip(),
                    judge_model=judge_model,
                )
    except json.JSONDecodeError:
        pass

    match = _ACCURACY_RE.search(raw)
    if match:
        accuracy = _parse_unit_score(match.group(1))
        if accuracy is not None:
            issue_m = _ISSUE_RE.search(raw)
            issue = issue_m.group(1).strip() if issue_m else ""
            return CaptionVisionAccuracy(
                accuracy=accuracy,
                issue=issue,
                judge_model=judge_model,
            )
    return CaptionVisionAccuracy(
        accuracy=0.0,
        judge_model=judge_model,
        parse_error="InvalidJSON",
    )


def judge_caption_vision_accuracy(
    *,
    client: OpenAI | None,
    model: str,
    frames_jpeg: list[bytes],
    caption: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> CaptionVisionAccuracy:
    """Score one caption against video frames (leaderboard-aligned accuracy)."""
    if is_google_ai_model(model):
        return CaptionVisionAccuracy(
            accuracy=0.0,
            judge_model=model,
            parse_error="GoogleCaptionVisionUnsupported",
        )
    resolved = client or resolve_llm_client(model)
    if resolved is None:
        return CaptionVisionAccuracy(
            accuracy=0.0,
            judge_model=model,
            parse_error="MissingClient",
        )

    system_prompt = load_judge_caption_accuracy_prompt()
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": _to_data_url(b)}} for b in frames_jpeg
    ]
    content.append(
        {
            "type": "text",
            "text": (
                f"Caption:\n{caption.strip()}\n\n"
                "Score accuracy against the frames above."
            ),
        }
    )
    tok = (
        max_tokens
        if max_tokens is not None
        else max(get_int_env("CAPTION_VISION_JUDGE_MAX_TOKENS", 256), 64)
    )
    temp = (
        temperature
        if temperature is not None
        else get_float_env("CAPTION_VISION_JUDGE_TEMPERATURE", 0.1)
    )
    last_raw = ""
    for use_json_mode in (True, False):
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": tok,
            "temperature": temp,
        }
        if use_json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}
        extra = fireworks_extra_body(model)
        if extra:
            request_kwargs["extra_body"] = extra
        try:
            resp = resolved.chat.completions.create(**request_kwargs)
            last_raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            return CaptionVisionAccuracy(
                accuracy=0.0,
                judge_model=model,
                parse_error=type(exc).__name__,
            )
        score = parse_caption_vision_accuracy_response(last_raw, judge_model=model)
        if score.ok:
            return score
    return parse_caption_vision_accuracy_response(last_raw, judge_model=model)
