import os
import base64
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI

from src.env import get_float_env, get_int_env
from src.describe_schema import parse_describe_json
from src.results import (
    CAPTION_FAILURE_PREFIX,
    DESCRIBE_FAILURE_PREFIX,
    PROCESS_FAILURE_PREFIX,
    CaptionError,
    CaptionResult,
    DescribeResult,
    ProcessError,
    caption_error_from_reason,
    process_failure_string,
)
from src.llm_clients import (
    get_fireworks_client,
    get_openrouter_client,
    google_api_key as _google_api_key,
    is_google_ai_model,
    is_openrouter_model,
    resolve_google_model_id as _resolve_google_model_id,
    resolve_llm_client,
)
from src.retry import RetryPolicy, call_with_retry
from src.caption_salvage import (
    compress_caption_call,
    is_drafting_junk,
    iter_salvage_candidates,
    pick_two_sentence_fit,
    pick_valid_candidate,
)

# Load repo-root .env reliably (Streamlit/other runners may change CWD).
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_REPO_ROOT / ".env")

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")

_STYLE_TEMPERATURE: dict[str, float] = {
    "formal": 0.5,
    "sarcastic": 0.78,
    "humorous_tech": 0.78,
    "humorous_non_tech": 0.72,
}

_META_LEAK_SALVAGE_TEMP = 0.12

# Soft validation ceilings (prompt limit + ~12-word buffer). Catches paragraph dumps,
# not a caption that runs a few words over the prompt target.
_STYLE_WORD_HARD_LIMIT: dict[str, int] = {
    "formal": 58,
    "humorous_non_tech": 62,
    "sarcastic": 68,
    "humorous_tech": 68,
}
_DEFAULT_WORD_HARD_LIMIT = 72

_DESCRIBE_FAILURE_PREFIX = DESCRIBE_FAILURE_PREFIX
_CAPTION_FAILURE_PREFIX = CAPTION_FAILURE_PREFIX
_PROCESS_FAILURE_PREFIX = PROCESS_FAILURE_PREFIX

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


def structured_describe_enabled() -> bool:
    return os.environ.get("STRUCTURED_DESCRIBE", "1") == "1"


def _describe_prompt_name() -> str:
    return "describe" if structured_describe_enabled() else "describe_prose"


_STYLE_STRUCTURED_HINT = (
    "Use at least one color or marking from subjects and at least one action "
    "from actions_early or actions_late. Match setting and surfaces exactly "
    "(water vs ground, indoor vs outdoor). Do not invent details not present below."
)

_META_LEAK_RETRY_NUDGE = (
    "Your last reply restated instructions or rules. "
    "Output ONLY the final caption — two sentences, no preamble, "
    "no mention of rules, the user, or your task."
)

_DIVERSITY_RETRY_NUDGE = (
    "Write a fresh variant: keep the same scene facts but change wording and humor angle."
)


def _build_style_user_prompt(
    description: str,
    *,
    meta_leak_retry: bool = False,
    diversity_retry: bool = False,
) -> str:
    if structured_describe_enabled():
        prompt = f"{_STYLE_STRUCTURED_HINT}\n\nScene facts:\n{description}"
    else:
        prompt = f"Scene facts:\n{description}"
    if meta_leak_retry:
        prompt = f"{prompt}\n\n{_META_LEAK_RETRY_NUDGE}"
    if diversity_retry:
        prompt = f"{prompt}\n\n{_DIVERSITY_RETRY_NUDGE}"
    return prompt


def _friendly_failures_enabled() -> bool:
    return os.environ.get("FRIENDLY_FAILURES", "0") == "1"


def is_describe_failure(text: str) -> bool:
    return text.startswith(DESCRIBE_FAILURE_PREFIX)


def public_describe_result(result: DescribeResult, *, style: str = "formal") -> str:
    if result.ok:
        return result.text or ""
    if not _friendly_failures_enabled():
        return result.to_failure_string()
    return _friendly_message(_FRIENDLY_DESCRIBE_FAILURE, style)


def public_caption_result(result: CaptionResult, *, style: str = "formal") -> str:
    if result.ok:
        return result.text or ""
    if not _friendly_failures_enabled():
        return result.to_failure_string()
    return _friendly_message(_FRIENDLY_CAPTION_FAILURE, style)


def public_process_failure(
    error: ProcessError,
    *,
    style: str = "formal",
    detail: str = "",
) -> str:
    raw = process_failure_string(error, detail=detail)
    if not _friendly_failures_enabled():
        return raw
    return _friendly_message(_FRIENDLY_PROCESS_FAILURE, style)


def public_caption(text: str, *, style: str = "formal") -> str:
    """Map internal failure strings to judge-facing captions when enabled."""
    if not _friendly_failures_enabled():
        return text
    if text.startswith(DESCRIBE_FAILURE_PREFIX):
        return _friendly_message(_FRIENDLY_DESCRIBE_FAILURE, style)
    if text.startswith(CAPTION_FAILURE_PREFIX):
        return _friendly_message(_FRIENDLY_CAPTION_FAILURE, style)
    if text.startswith(PROCESS_FAILURE_PREFIX):
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


_PROMPTS_DIR = _REPO_ROOT / "prompts"
_OUTPUT_CONTRACT_PATH = _PROMPTS_DIR / "_output_contract.txt"


@functools.lru_cache(maxsize=16)
def _read_prompt_file(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _output_contract() -> str:
    return _OUTPUT_CONTRACT_PATH.read_text(encoding="utf-8").strip()


@functools.lru_cache(maxsize=16)
def load_prompt(style: str) -> str:
    prompt_name = _describe_prompt_name() if style == "describe" else style
    body = _read_prompt_file(prompt_name)
    if style not in STYLES:
        return body
    role, _, rest = body.partition("\n\n")
    return f"{role}\n\n{_output_contract()}\n\n{rest}"


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


def _maybe_salvage_meta_leak_preamble(output: str) -> str | None:
    """Recover a caption buried after a label preamble, e.g. 'Caption: ...'."""
    from src.caption_salvage import _strip_label_prefix

    return _strip_label_prefix(output)


def _salvage_output(raw: str, *, style: str) -> str:
    """Apply heuristic extraction before validation."""
    salvaged, _ = pick_valid_candidate(raw, style=style, is_valid=_is_bad_output)
    if salvaged:
        return salvaged
    hard_limit = _STYLE_WORD_HARD_LIMIT.get(style, _DEFAULT_WORD_HARD_LIMIT)
    fit, _ = pick_two_sentence_fit(
        raw,
        style=style,
        hard_limit=hard_limit,
        is_valid=_is_bad_output,
    )
    if fit:
        return fit
    return raw


def _normalize_style_output(output: str, *, style: str = "") -> str:
    stripped = _strip_wrapping_quotes(output.strip())
    if not stripped:
        return stripped
    if style:
        return _salvage_output(stripped, style=style)
    salvaged = _maybe_salvage_meta_leak_preamble(stripped)
    return salvaged if salvaged else stripped


def _is_bad_output(output: str, *, style: str = "") -> tuple[bool, str]:
    """Basic sanity check on a generated caption. Used as a validation
    signal (for retry-once and for scoring), not as the driver of a
    cascading fallback-prompt chain."""
    if not output.strip():
        return True, "EmptyResponse"
    if is_drafting_junk(output):
        return True, "MetaLeak"
    if _is_meta_leak(output):
        return True, "MetaLeak"
    words = output.split()
    if len(words) < 5:
        return True, "TooShort"
    hard_limit = _STYLE_WORD_HARD_LIMIT.get(style, _DEFAULT_WORD_HARD_LIMIT)
    if len(words) > hard_limit:
        return True, "TooLong"
    if looks_truncated(output, None):
        return True, "Truncated"
    return False, ""


@dataclass
class _StyleAttempt:
    out: str
    finish_reason: str | None
    is_bad: bool
    bad_reason: str
    truncated: bool


def vision_describe_call(
    *,
    client: OpenAI | None,
    model: str,
    frames_jpeg: Iterable[bytes],
    max_tokens: int,
    temperature: float,
    json_mode: bool = False,
) -> tuple[str, str | None]:
    if is_google_ai_model(model):
        return _google_vision_describe_call(
            model=model,
            frames_jpeg=frames_jpeg,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
    if client is None:
        raise RuntimeError("OpenAI-compatible client required for this vision model")
    system_prompt = load_prompt(_describe_prompt_name())
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": _to_data_url(b)}} for b in frames_jpeg
    ]
    request_kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        request_kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**request_kwargs)
    choice = resp.choices[0]
    text = _strip_wrapping_quotes((choice.message.content or "").strip())
    return text, choice.finish_reason


def generate_factual_description(
    *,
    client: OpenAI | None,
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
    temperature_override: float | None = None,
    diversity_retry: bool = False,
) -> CaptionResult:
    """Generate one styled caption from a factual description.

    At most STYLE_MAX_ATTEMPTS API calls (default 3). On persistent failure,
    returns a CaptionResult with a typed error.
    """
    system_prompt = load_prompt(style)
    base_max = max(get_int_env("STYLE_MAX_TOKENS", get_int_env("MAX_TOKENS", 140)), 32)
    if temperature_override is not None:
        base_temp = temperature_override
    else:
        base_temp = _STYLE_TEMPERATURE.get(style, get_float_env("TEMPERATURE", 0.75))
    policy = RetryPolicy(
        max_attempts=max(get_int_env("STYLE_MAX_ATTEMPTS", 3), 1),
        base_sleep_s=1.0,
        jitter_s=get_float_env("RETRY_JITTER_S", 0.5),
    )
    last_reason = "EmptyResponse"

    def style_attempt(
        *,
        max_tokens: int,
        temperature: float,
        meta_leak_retry: bool,
        json_mode: bool = False,
    ) -> _StyleAttempt:
        user_prompt = _build_style_user_prompt(
            description,
            meta_leak_retry=meta_leak_retry,
            diversity_retry=diversity_retry,
        )
        if json_mode:
            user_prompt = (
                f"{user_prompt}\n\nReply with JSON only: "
                '{"caption":"your two-sentence caption here"}'
            )
        request_kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**request_kwargs)
        choice = resp.choices[0]
        raw = (choice.message.content or "").strip()
        out = _normalize_style_output(_strip_wrapping_quotes(raw), style=style)
        finish_reason = choice.finish_reason
        truncated = looks_truncated(out, finish_reason)
        is_bad, reason = _is_bad_output(out, style=style)
        if is_bad and reason in ("MetaLeak", "TooLong"):
            salvaged, _ = pick_valid_candidate(raw, style=style, is_valid=_is_bad_output)
            if salvaged:
                out = salvaged
                is_bad, reason = _is_bad_output(out, style=style)
            if is_bad and reason == "TooLong":
                hard_limit = _STYLE_WORD_HARD_LIMIT.get(style, _DEFAULT_WORD_HARD_LIMIT)
                fit, _ = pick_two_sentence_fit(
                    raw,
                    style=style,
                    hard_limit=hard_limit,
                    is_valid=_is_bad_output,
                )
                if fit:
                    out = fit
                    is_bad, reason = _is_bad_output(out, style=style)
        return _StyleAttempt(
            out=out,
            finish_reason=finish_reason,
            is_bad=is_bad,
            bad_reason=reason,
            truncated=truncated,
        )

    use_json_first = get_int_env("STYLE_JSON_MODE", 0) == 1

    def attempt_fn(attempt: int) -> _StyleAttempt:
        nonlocal last_reason
        if attempt == 1:
            max_tokens = base_max
        elif last_reason == "Truncated":
            max_tokens = base_max + 220
        elif last_reason == "TooLong":
            max_tokens = base_max + 40
        else:
            max_tokens = base_max + 60
        meta_retry = attempt > 1 and last_reason == "MetaLeak"
        too_long_retry = attempt > 1 and last_reason == "TooLong"
        if meta_retry or too_long_retry:
            temperature = min(base_temp - (attempt - 1) * 0.08, 0.25)
        else:
            temperature = min(max(base_temp - (attempt - 1) * 0.08, 0.2), 0.97)
        result = style_attempt(
            max_tokens=max_tokens,
            temperature=temperature,
            meta_leak_retry=meta_retry,
            json_mode=use_json_first and attempt == 1,
        )
        last_reason = "Truncated" if result.truncated else result.bad_reason
        return result

    def classify(attempt: int, result: _StyleAttempt) -> str | None:
        if not result.truncated and not result.is_bad:
            return None
        if (
            attempt == policy.max_attempts
            and result.truncated
            and result.out.strip()
            and not result.is_bad
        ):
            return None
        return "Truncated" if result.truncated else result.bad_reason or "EmptyResponse"

    def should_sleep(_attempt: int, reason: str) -> bool:
        return reason in ("EmptyResponse", "MetaLeak", "TooLong")

    def _finalize(out: str, *, truncated: bool) -> str:
        text = out.strip()
        if truncated and text and text[-1] not in _VALID_END_CHARS:
            text = text + "."
        return text

    attempts_holder: list[int] = []

    def _try_post_fail_salvage(reason: str, draft: str) -> str | None:
        if reason == "MetaLeak":
            salvaged, _ = pick_valid_candidate(draft, style=style, is_valid=_is_bad_output)
            if salvaged:
                bad, _ = _is_bad_output(salvaged, style=style)
                if not bad:
                    return salvaged
        if reason in ("MetaLeak", "TooLong") and get_int_env("STYLE_META_LEAK_SALVAGE", 1):
            salvage = style_attempt(
                max_tokens=base_max + 60,
                temperature=_META_LEAK_SALVAGE_TEMP,
                meta_leak_retry=True,
            )
            attempts_holder.append(1)
            if not salvage.is_bad:
                return _finalize(salvage.out, truncated=salvage.truncated)
        if reason == "TooLong" and get_int_env("STYLE_COMPRESS_ON_LONG", 1):
            compressed_raw = compress_caption_call(
                client=client,
                model=model,
                style=style,
                draft=draft,
                system_prompt=system_prompt,
            )
            attempts_holder.append(1)
            if compressed_raw:
                compressed = _normalize_style_output(
                    _strip_wrapping_quotes(compressed_raw),
                    style=style,
                )
                bad, _ = _is_bad_output(compressed, style=style)
                if not bad:
                    return _finalize(compressed, truncated=False)
                salvaged, _ = pick_valid_candidate(
                    compressed_raw,
                    style=style,
                    is_valid=_is_bad_output,
                )
                if salvaged:
                    return _finalize(salvaged, truncated=False)
        return None

    last, reasons = call_with_retry(
        policy=policy,
        attempt=attempt_fn,
        classify=classify,
        should_sleep=should_sleep,
    )
    attempts = len(reasons) + 1 if not reasons else len(reasons)
    attempts += sum(attempts_holder)

    if not reasons:
        out = last.out
        if last.truncated and out.strip() and out[-1] not in _VALID_END_CHARS:
            out = out.strip() + "."
        return CaptionResult(text=out, error=None, attempts=attempts)

    if last.truncated and last.out.strip() and not last.is_bad:
        out = last.out.strip()
        if out[-1] not in _VALID_END_CHARS:
            out = out + "."
        return CaptionResult(text=out, error=None, attempts=attempts)

    if last.is_bad:
        reason = last.bad_reason or reasons[-1]
        salvaged_text = _try_post_fail_salvage(reason, last.out or "")
        if salvaged_text:
            return CaptionResult(text=salvaged_text, error=None, attempts=attempts)
        return CaptionResult(
            text=None,
            error=caption_error_from_reason(reason),
            error_detail=reason,
            attempts=attempts,
        )

    if last.out:
        return CaptionResult(text=last.out, error=None, attempts=attempts)
    return CaptionResult(
        text=None,
        error=CaptionError.EMPTY,
        error_detail="EmptyResponse",
        attempts=attempts,
    )


def dry_run_captions(task_id: str, styles: list[str]) -> dict[str, str]:
    return {s: f"[DRY_RUN] {task_id} - {s}" for s in styles}


def resolve_caption_model_pool() -> list[tuple[str, str]]:
    """Parse CAPTION_MODEL_POOL (comma-separated slugs or label=model)."""
    raw = os.environ.get("CAPTION_MODEL_POOL", "").strip()
    if not raw:
        return []
    out: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, model = item.split("=", 1)
            out.append((label.strip(), model.strip()))
        elif item.startswith("accounts/"):
            out.append((item.rsplit("/", 1)[-1], item))
        else:
            out.append((item, f"accounts/fireworks/models/{item}"))
    return out


def caption_pool_enabled() -> bool:
    return len(resolve_caption_model_pool()) > 1


def _google_finish_reason(response) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None
    name = getattr(reason, "name", None) or str(reason)
    upper = str(name).upper()
    if "MAX" in upper or upper == "LENGTH":
        return "length"
    return str(name).lower()


def _google_vision_describe_call(
    *,
    model: str,
    frames_jpeg: Iterable[bytes],
    max_tokens: int,
    temperature: float,
    json_mode: bool = False,
) -> tuple[str, str | None]:
    from google import genai
    from google.genai import types

    system_prompt = load_prompt(_describe_prompt_name())
    # Bound hung requests: a stalled Google call must fail fast into the M3 fallback.
    timeout_ms = int(get_float_env("GOOGLE_API_TIMEOUT_S", 60.0) * 1000)
    client = genai.Client(
        api_key=_google_api_key(),
        http_options=types.HttpOptions(timeout=timeout_ms),
    )
    model_id = _resolve_google_model_id(model)

    parts: list[types.Part] = [
        types.Part.from_bytes(data=frame, mime_type="image/jpeg")
        for frame in frames_jpeg
    ]
    parts.append(
        types.Part.from_text(
            text="These images are evenly spaced frames from a short video clip, "
            "shown in chronological order."
        )
    )

    output_tokens = max(max_tokens, get_int_env("GOOGLE_DESCRIBE_MAX_TOKENS", 2048))
    config_kwargs: dict = {
        "system_instruction": system_prompt,
        "temperature": temperature,
        "max_output_tokens": output_tokens,
        "thinking_config": types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.MINIMAL,
            include_thoughts=False,
        ),
    }
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=model_id,
        contents=parts,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    text = _google_response_text(response)
    text = _strip_wrapping_quotes(text.strip())
    return text, _google_finish_reason(response)


def _google_response_text(response) -> str:
    """Prefer public text; fall back to non-thought candidate parts."""
    text = (getattr(response, "text", None) or "").strip()
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    chunks: list[str] = []
    for part in parts:
        if getattr(part, "thought", False):
            continue
        chunk = getattr(part, "text", None)
        if chunk:
            chunks.append(chunk.strip())
    return "\n".join(chunks).strip()
