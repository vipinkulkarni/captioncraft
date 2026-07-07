import base64
import os
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


def load_prompt(style: str) -> str:
    path = Path(__file__).resolve().parent.parent / "prompts" / f"{style}.txt"
    return path.read_text(encoding="utf-8")


def _to_data_url(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _looks_truncated(text: str, finish_reason: str | None) -> bool:
    if finish_reason == "length":
        return True
    if not text:
        return False
    return text[-1] not in ".!?"


def _word_overlap_ratio(a: str, b: str) -> float:
    wa = {w for w in a.lower().split() if w}
    wb = {w for w in b.lower().split() if w}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _is_verbatim_copy(output: str, description: str) -> bool:
    if not output or not description:
        return False
    out = output.strip().lower()
    desc = description.strip().lower()
    if out == desc:
        return True
    if out in desc or desc in out:
        return True
    return _word_overlap_ratio(output, description) >= 0.65


def _is_bad_style_output(output: str, description: str) -> bool:
    if not output.strip():
        return True
    if len(output.split()) > 45:
        return True
    return _is_verbatim_copy(output, description)


def generate_factual_description(
    *,
    client: OpenAI,
    model: str,
    frames_jpeg: Iterable[bytes],
    max_tokens_override: int | None = None,
    truncation_retries: int = 0,
) -> str:
    prompt = load_prompt("describe")
    base_max = max(get_int_env("DESCRIBE_MAX_TOKENS", get_int_env("MAX_TOKENS", 550)), 64)
    max_tokens = max_tokens_override if max_tokens_override is not None else base_max
    temperature = get_float_env("DESCRIBE_TEMPERATURE", 0.2)

    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": _to_data_url(b)}} for b in frames_jpeg
    ]
    content.append({"type": "text", "text": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    finish_reason = choice.finish_reason

    if _looks_truncated(text, finish_reason) and truncation_retries < 2:
        bumped = base_max + (truncation_retries + 1) * 150
        return generate_factual_description(
            client=client,
            model=model,
            frames_jpeg=frames_jpeg,
            max_tokens_override=bumped,
            truncation_retries=truncation_retries + 1,
        )

    return text


def _truncate_to_words(text: str, max_words: int = 40) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    trimmed = " ".join(words[:max_words])
    if trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


def _ensure_non_empty(caption: str, description: str, style: str) -> str:
    if caption.strip():
        return caption.strip()
    short = _truncate_to_words(description, 40)
    if short.strip():
        return short
    return f"Video caption ({style})."


def _emergency_styled_caption(
    *,
    client: OpenAI,
    model: str,
    style: str,
    description: str,
) -> str:
    style_hint = {
        "formal": "polished, neutral, professional",
        "sarcastic": "dry, ironic, subtly mocking",
        "humorous_tech": "funny with tech/developer jokes",
        "humorous_non_tech": "funny everyday humor, no tech jargon",
    }.get(style, style)

    prompt = (
        f"Write a {style_hint} video caption using ONLY these facts:\n{description}\n\n"
        "Rules: 2 short sentences max, under 50 words, completely new wording, "
        "do not copy any sentence from the facts. English only. No emojis."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=90,
        temperature=0.95,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_styled_caption_from_text(
    *,
    client: OpenAI,
    model: str,
    style: str,
    description: str,
    retry: int = 0,
) -> str:
    prompt_template = load_prompt(style)
    max_tokens = max(
        get_int_env("STYLE_MAX_TOKENS", get_int_env("MAX_TOKENS", 140)),
        32,
    )
    base_temp = _STYLE_TEMPERATURE.get(style, get_float_env("TEMPERATURE", 0.75))
    temperature = min(base_temp + retry * 0.1, 0.97)

    if retry >= 2:
        content_block = (
            f"Factual description of the video:\n{description}\n\n"
            "Your last attempt copied the source. Rewrite in your style with completely fresh wording. "
            "Maximum 2 short sentences, under 50 words. English only. No emojis."
        )
        max_tokens = min(max_tokens, 90)
        temperature = 0.95
    else:
        content_block = (
            f"Factual description of the video:\n{description}\n\n"
            "Rewrite this into your style. Do not copy the description verbatim or reuse its phrases. "
            "Pick only the most important details. 2 short sentences, under 55 words. "
            "English only. No emojis."
        )

    user_prompt = prompt_template.format(content=content_block)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    out = (resp.choices[0].message.content or "").strip()

    if not out and retry < 2:
        time.sleep(1)
        return generate_styled_caption_from_text(
            client=client,
            model=model,
            style=style,
            description=description,
            retry=retry + 1,
        )

    if _is_bad_style_output(out, description) and retry < 2:
        return generate_styled_caption_from_text(
            client=client,
            model=model,
            style=style,
            description=description,
            retry=retry + 1,
        )

    if _is_bad_style_output(out, description):
        emergency = _emergency_styled_caption(
            client=client,
            model=model,
            style=style,
            description=description,
        )
        if emergency and not _is_bad_style_output(emergency, description):
            return _ensure_non_empty(emergency, description, style)
        return _ensure_non_empty(_truncate_to_words(emergency or out, 40), description, style)

    return _ensure_non_empty(out or description, description, style)


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
