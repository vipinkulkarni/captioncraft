import base64
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI

from src.env import get_float_env, get_int_env

load_dotenv()

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")


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
    return _word_overlap_ratio(output, description) >= 0.8


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
        get_int_env("STYLE_MAX_TOKENS", get_int_env("MAX_TOKENS", 180)),
        32,
    )
    base_temp = get_float_env("TEMPERATURE", 0.75)
    temperature = min(base_temp + retry * 0.12, 0.95)

    if retry >= 2:
        content_block = (
            f"Factual description of the video:\n{description}\n\n"
            "Your last attempt copied the source. Rewrite in your style with fresh wording. "
            "Maximum 2 short sentences, under 55 words. English only. No emojis."
        )
        max_tokens = min(max_tokens, 100)
        temperature = 0.9
    else:
        content_block = (
            f"Factual description of the video:\n{description}\n\n"
            "Rewrite this into your style. Do not copy the description verbatim. "
            "Pick only the most important details. 2-3 short sentences, under 70 words. "
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

    if _is_verbatim_copy(out, description) and retry < 2:
        return generate_styled_caption_from_text(
            client=client,
            model=model,
            style=style,
            description=description,
            retry=retry + 1,
        )

    if _is_verbatim_copy(out, description):
        sentences = [s.strip() for s in out.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        short = ". ".join(sentences[:2])
        if short and not short.endswith("."):
            short += "."
        return short or description

    return out or description


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
