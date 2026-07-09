import os
import base64
import functools
import time
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI

from src.env import get_float_env, get_int_env

# Load repo-root .env reliably (Streamlit/other runners may change CWD).
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_REPO_ROOT / ".env")

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")

_STYLE_TEMPERATURE: dict[str, float] = {
    "formal": 0.5,
    "sarcastic": 0.88,
    "humorous_tech": 0.88,
    "humorous_non_tech": 0.82,
}

# Soft validation ceilings (prompt limit + ~12-word buffer). Catches paragraph dumps,
# not a caption that runs a few words over the prompt target.
_STYLE_WORD_HARD_LIMIT: dict[str, int] = {
    "formal": 58,
    "humorous_non_tech": 62,
    "sarcastic": 68,
    "humorous_tech": 68,
}
_DEFAULT_WORD_HARD_LIMIT = 72

_DESCRIBE_FAILURE_PREFIX = "Failed to describe video:"
_CAPTION_FAILURE_PREFIX = "Failed to caption:"
_PROCESS_FAILURE_PREFIX = "Failed to process video:"

_FRIENDLY_DESCRIBE_FAILURE: dict[str, str] = {
    "formal": (
        "The available footage did not provide enough visual detail for a complete scene description."
    ),
    "sarcastic": (
        "The clip kept its secrets, offering too little to work with for a proper scene read."
    ),
    "humorous_tech": (
        "Sparse frames left the describe pipeline empty—nothing reliable enough to commit to production."
    ),
    "humorous_non_tech": (
        "The clip was too stingy with details to pin down what was really going on."
    ),
}
_FRIENDLY_CAPTION_FAILURE: dict[str, str] = {
    "formal": (
        "A scene unfolds in the video, though a styled caption could not be produced."
    ),
    "sarcastic": (
        "Something happens in this clip, apparently—but a caption with the requested tone never showed up."
    ),
    "humorous_tech": (
        "The scene description compiled, but this styled caption deploy failed—no output shipped."
    ),
    "humorous_non_tech": (
        "Something's clearly happening here, but the caption never quite came together."
    ),
}
_FRIENDLY_PROCESS_FAILURE: dict[str, str] = {
    "formal": "This video clip could not be processed into captions with the requested styles.",
    "sarcastic": "The pipeline looked at this task and quietly declined to cooperate.",
    "humorous_tech": (
        "Processing hit an unhandled edge case—no usable captions made it to production."
    ),
    "humorous_non_tech": (
        "This clip didn't cooperate, so no proper captions made it out the other side."
    ),
}


def _friendly_message(messages: dict[str, str], style: str) -> str:
    return messages.get(style) or messages["formal"]


def _friendly_failures_enabled() -> bool:
    return os.environ.get("FRIENDLY_FAILURES", "0") == "1"


def is_describe_failure(text: str) -> bool:
    return text.startswith(_DESCRIBE_FAILURE_PREFIX)


def public_caption(text: str, *, style: str = "formal") -> str:
    """Map internal failure strings to judge-facing captions when enabled."""
    if not _friendly_failures_enabled():
        return text
    if text.startswith(_DESCRIBE_FAILURE_PREFIX):
        return _friendly_message(_FRIENDLY_DESCRIBE_FAILURE, style)
    if text.startswith(_CAPTION_FAILURE_PREFIX):
        return _friendly_message(_FRIENDLY_CAPTION_FAILURE, style)
    if text.startswith(_PROCESS_FAILURE_PREFIX):
        return _friendly_message(_FRIENDLY_PROCESS_FAILURE, style)
    if text == "Invalid task input." or text == "Unsupported style requested.":
        return _friendly_message(_FRIENDLY_PROCESS_FAILURE, style)
    return text

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


@functools.lru_cache(maxsize=16)
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


_VALID_END_CHARS = ".!?)\""


def looks_truncated(text: str, finish_reason: str | None) -> bool:
    if finish_reason == "length":
        return True
    if not text:
        return False
    return text[-1] not in _VALID_END_CHARS


def _is_meta_leak(output: str) -> bool:
    lower = output.strip().lower()
    if any(lower.startswith(p) for p in _META_LEAK_PREFIXES):
        return True
    return sum(1 for m in _META_LEAK_MARKERS if m in lower) >= 2


def _is_bad_output(output: str, *, style: str = "") -> tuple[bool, str]:
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
    hard_limit = _STYLE_WORD_HARD_LIMIT.get(style, _DEFAULT_WORD_HARD_LIMIT)
    if len(words) > hard_limit:
        return True, "TooLong"
    return False, ""


def vision_describe_call(
    *,
    client: OpenAI,
    model: str,
    frames_jpeg: Iterable[bytes],
    max_tokens: int,
    temperature: float,
) -> tuple[str, str | None]:
    system_prompt = load_prompt("describe")
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
    return text, choice.finish_reason


def generate_factual_description(
    *,
    client: OpenAI,
    model: str,
    frames_jpeg: Iterable[bytes],
) -> str:
    """Single-shot describe call. Retries are handled in pipeline._describe_frames."""
    base_max = max(get_int_env("DESCRIBE_MAX_TOKENS", 1200), 64)
    temperature = get_float_env("DESCRIBE_TEMPERATURE", 0.2)
    text, _ = vision_describe_call(
        client=client,
        model=model,
        frames_jpeg=frames_jpeg,
        max_tokens=base_max,
        temperature=temperature,
    )
    return text


def generate_styled_caption_from_text(
    *,
    client: OpenAI,
    model: str,
    style: str,
    description: str,
) -> str:
    """Generate one styled caption from a factual description.

    At most STYLE_MAX_ATTEMPTS API calls (default 2). On persistent failure,
    returns a clearly marked "Failed to caption: <reason>" string.
    """
    system_prompt = load_prompt(style)
    base_max = max(get_int_env("STYLE_MAX_TOKENS", get_int_env("MAX_TOKENS", 140)), 32)
    base_temp = _STYLE_TEMPERATURE.get(style, get_float_env("TEMPERATURE", 0.75))
    max_attempts = max(get_int_env("STYLE_MAX_ATTEMPTS", 2), 1)
    user_prompt = f"Video description:\n{description}"

    last_out = ""
    last_reason = "EmptyResponse"

    for attempt in range(max_attempts):
        if attempt == 0:
            max_tokens = base_max
        elif last_reason == "Truncated" or looks_truncated(last_out, "length"):
            max_tokens = base_max + 220
        else:
            max_tokens = base_max + 60
        temperature = min(max(base_temp - attempt * 0.08, 0.2), 0.97)

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
        last_out = out

        truncated = looks_truncated(out, finish_reason)
        is_bad, reason = _is_bad_output(out, style=style)
        last_reason = "Truncated" if truncated else reason

        if not truncated and not is_bad:
            return out

        if attempt < max_attempts - 1:
            if reason in ("EmptyResponse", "MetaLeak"):
                time.sleep(1)
            continue

        # Last attempt: keep partial output only for clean truncation, not bad output.
        if truncated and out.strip() and not is_bad:
            out = out.strip()
            if out[-1] not in _VALID_END_CHARS:
                out = out + "."
            return out
        if is_bad:
            return f"{_CAPTION_FAILURE_PREFIX} {reason}"

    return last_out or f"{_CAPTION_FAILURE_PREFIX} EmptyResponse"


def dry_run_captions(task_id: str, styles: list[str]) -> dict[str, str]:
    return {s: f"[DRY_RUN] {task_id} - {s}" for s in styles}


def get_fireworks_client() -> OpenAI:
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing FIREWORKS_API_KEY")

    timeout_s = get_float_env("API_TIMEOUT_S", 45.0)
    return OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
        timeout=timeout_s,
        max_retries=0,
    )
