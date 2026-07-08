import os
import base64
import time
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI

from src.env import get_float_env, get_int_env

load_dotenv()

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")

_STYLE_TEMPERATURE: dict[str, float] = {
    "formal": 0.5,
    "sarcastic": 0.88,
    "humorous_tech": 0.88,
    "humorous_non_tech": 0.82,
}

_META_LEAK_PREFIXES = (
    "we need to",
    "the user wants",
    "the user asks",
    "i need to",
    "i will",
    "here is",
    "here's",
    "caption:",
)

_META_LEAK_MARKERS = (
    "we need to",
    "the user wants",
    "the user asks",
    "using only these facts",
    "must be 2 short sentences",
    "completely new wording",
    "video description:",
    "output contract",
)


def load_prompt(style: str) -> str:
    path = Path(__file__).resolve().parent.parent / "prompts" / f"{style}.txt"
    return path.read_text(encoding="utf-8")


def _to_data_url(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        return text[1:-1].strip()
    return text


def _looks_truncated(text: str, finish_reason: str | None) -> bool:
    if finish_reason == "length":
        return True
    if not text:
        return False
    return text[-1] not in ".!?"


def _is_meta_leak(output: str) -> bool:
    lower = output.strip().lower()
    if any(lower.startswith(p) for p in _META_LEAK_PREFIXES):
        return True
    return sum(1 for m in _META_LEAK_MARKERS if m in lower) >= 2


def _is_bad_output(output: str) -> tuple[bool, str]:
    """Basic sanity check on a generated caption. Used as a validation
    signal (for retry-once and for scoring), not as the driver of a
    cascading fallback-prompt chain."""
    if not output.strip():
        return True, "EmptyResponse"
    if _is_meta_leak(output):
        return True, "MetaLeak"
    words = output.split()
    if len(words) < 5:
        return True, "TooShort"
    if len(words) > 70:
        return True, "TooLong"
    return False, ""


def generate_factual_description(
    *,
    client: OpenAI,
    model: str,
    frames_jpeg: Iterable[bytes],
    retry: int = 0,
) -> str:
    system_prompt = load_prompt("describe")
    # Do not fall back to MAX_TOKENS — that env var is often 220 for style captions
    # and starves reasoning/vision models (Gemma, GPT-OSS) into empty responses.
    base_max = max(get_int_env("DESCRIBE_MAX_TOKENS", 1000), 64)
    max_tokens = base_max + (300 if retry else 0)
    temperature = get_float_env("DESCRIBE_TEMPERATURE", 0.2)

    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": _to_data_url(b)}} for b in frames_jpeg
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = resp.choices[0]
    text = _strip_wrapping_quotes((choice.message.content or "").strip())
    finish_reason = choice.finish_reason

    if (not text or _looks_truncated(text, finish_reason)) and retry < 1:
        return generate_factual_description(
            client=client,
            model=model,
            frames_jpeg=frames_jpeg,
            retry=retry + 1,
        )

    return text


def generate_styled_caption_from_text(
    *,
    client: OpenAI,
    model: str,
    style: str,
    description: str,
    retry: int = 0,
) -> str:
    """Generate one styled caption from a factual description.

    Message structure is a strict system/user split: the system message
    carries the persona, style rules, and an explicit "output only the
    caption" contract; the user message carries only the raw description.
    This keeps reasoning-style models from echoing instructions back
    instead of answering (see misc/eval/model_eval_plan.md).

    At most 2 attempts total. On persistent failure, returns a clearly
    marked "Failed to caption: <reason>" string rather than papering over
    it with emergency prompts or truncated fallbacks.
    """
    system_prompt = load_prompt(style)
    base_max = max(get_int_env("STYLE_MAX_TOKENS", get_int_env("MAX_TOKENS", 140)), 32)
    max_tokens = base_max + (60 if retry else 0)
    base_temp = _STYLE_TEMPERATURE.get(style, get_float_env("TEMPERATURE", 0.75))
    temperature = min(base_temp + retry * 0.15, 0.97)

    user_prompt = f"Video description:\n{description}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = resp.choices[0]
    out = _strip_wrapping_quotes((choice.message.content or "").strip())
    finish_reason = choice.finish_reason

    truncated = _looks_truncated(out, finish_reason)
    is_bad, reason = _is_bad_output(out)

    if (truncated or is_bad) and retry < 1:
        if is_bad and reason == "EmptyResponse":
            time.sleep(1)
        return generate_styled_caption_from_text(
            client=client,
            model=model,
            style=style,
            description=description,
            retry=retry + 1,
        )

    if truncated:
        return "Failed to caption: Truncated"
    if is_bad:
        return f"Failed to caption: {reason}"

    return out


def dry_run_captions(task_id: str, styles: list[str]) -> dict[str, str]:
    return {s: f"[DRY_RUN] {task_id} - {s}" for s in styles}


def get_fireworks_client() -> OpenAI:
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing FIREWORKS_API_KEY")

    return OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
    )
