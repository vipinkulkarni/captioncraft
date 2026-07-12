"""Vision-backed describe faithfulness judge and dual-describe pick-best."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.caption import _to_data_url
from src.env import get_float_env, get_int_env
from src.llm_clients import fireworks_extra_body, is_google_ai_model, resolve_llm_client
from src.results import DescribeResult

_REPO_ROOT = Path(__file__).resolve().parent.parent
_JUDGE_DESCRIBE_PROMPT_PATH = _REPO_ROOT / "prompts" / "judge_describe.txt"

_DEFAULT_VISION_ALT = "accounts/fireworks/models/qwen3p7-plus"

_FAITHFULNESS_RE = re.compile(r'"faithfulness"\s*:\s*([0-9]*\.?[0-9]+)', re.I)
_COVERAGE_RE = re.compile(r'"coverage"\s*:\s*([0-9]*\.?[0-9]+)', re.I)
_ISSUE_RE = re.compile(r'"issue"\s*:\s*"(.*?)"', re.I | re.S)


@dataclass
class DescribeJudgeScore:
    faithfulness: float
    coverage: float
    issue: str = ""
    judge_model: str = ""
    parse_error: str = ""

    @property
    def proxy(self) -> float:
        return round((self.faithfulness + self.coverage) / 2.0, 3)

    @property
    def ok(self) -> bool:
        return not self.parse_error


def dual_describe_enabled() -> bool:
    return get_int_env("DESCRIBE_DUAL", 1) == 1


def resolve_vision_alt_model() -> str:
    return os.environ.get("VISION_ALT_MODEL", _DEFAULT_VISION_ALT).strip()


def resolve_describe_judge_min() -> float:
    return max(0.0, min(1.0, get_float_env("DESCRIBE_JUDGE_MIN", 0.85)))


def load_judge_describe_prompt() -> str:
    return _JUDGE_DESCRIBE_PROMPT_PATH.read_text(encoding="utf-8").strip()


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


def parse_describe_judge_response(
    raw: str,
    *,
    judge_model: str = "",
) -> DescribeJudgeScore:
    if not (raw or "").strip():
        return DescribeJudgeScore(
            faithfulness=0.0,
            coverage=0.0,
            judge_model=judge_model,
            parse_error="EmptyResponse",
        )
    try:
        payload = json.loads(_extract_json_object(raw))
        if isinstance(payload, dict):
            faithfulness = _parse_unit_score(payload.get("faithfulness"))
            coverage = _parse_unit_score(payload.get("coverage"))
            if faithfulness is not None and coverage is not None:
                return DescribeJudgeScore(
                    faithfulness=faithfulness,
                    coverage=coverage,
                    issue=str(payload.get("issue") or "").strip(),
                    judge_model=judge_model,
                )
    except json.JSONDecodeError:
        pass

    faith_m = _FAITHFULNESS_RE.search(raw)
    cov_m = _COVERAGE_RE.search(raw)
    if faith_m and cov_m:
        faithfulness = _parse_unit_score(faith_m.group(1))
        coverage = _parse_unit_score(cov_m.group(1))
        if faithfulness is not None and coverage is not None:
            issue_m = _ISSUE_RE.search(raw)
            issue = issue_m.group(1).strip() if issue_m else ""
            return DescribeJudgeScore(
                faithfulness=faithfulness,
                coverage=coverage,
                issue=issue,
                judge_model=judge_model,
            )
    return DescribeJudgeScore(
        faithfulness=0.0,
        coverage=0.0,
        judge_model=judge_model,
        parse_error="InvalidJSON",
    )


def judge_describe_call(
    *,
    client: OpenAI | None,
    model: str,
    frames_jpeg: list[bytes],
    description: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> DescribeJudgeScore:
    """Score one description against video frames with a vision model."""
    if is_google_ai_model(model):
        return DescribeJudgeScore(
            faithfulness=0.0,
            coverage=0.0,
            judge_model=model,
            parse_error="GoogleDescribeJudgeUnsupported",
        )
    resolved = client or resolve_llm_client(model)
    if resolved is None:
        return DescribeJudgeScore(
            faithfulness=0.0,
            coverage=0.0,
            judge_model=model,
            parse_error="MissingClient",
        )

    system_prompt = load_judge_describe_prompt()
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": _to_data_url(b)}} for b in frames_jpeg
    ]
    content.append(
        {
            "type": "text",
            "text": (
                "Candidate description:\n"
                f"{description.strip()}\n\n"
                "Score faithfulness and coverage against the frames above."
            ),
        }
    )
    tok = max_tokens if max_tokens is not None else max(get_int_env("DESCRIBE_JUDGE_MAX_TOKENS", 256), 64)
    temp = (
        temperature
        if temperature is not None
        else get_float_env("DESCRIBE_JUDGE_TEMPERATURE", 0.1)
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
        except Exception as exc:  # noqa: BLE001 — surface as parse_error for pick logic
            return DescribeJudgeScore(
                faithfulness=0.0,
                coverage=0.0,
                judge_model=model,
                parse_error=type(exc).__name__,
            )
        score = parse_describe_judge_response(last_raw, judge_model=model)
        if score.ok:
            return score
    return parse_describe_judge_response(last_raw, judge_model=model)


def pick_best_describe(
    *,
    primary: DescribeResult,
    primary_model: str,
    alternate: DescribeResult,
    alternate_model: str,
    primary_score: DescribeJudgeScore | None,
    alternate_score: DescribeJudgeScore | None,
) -> tuple[DescribeResult, str, DescribeJudgeScore | None, DescribeJudgeScore | None]:
    """Pick the describe with higher cross-judge proxy; ties prefer primary."""
    if primary.ok and not alternate.ok:
        return primary, primary_model, primary_score, alternate_score
    if alternate.ok and not primary.ok:
        return alternate, alternate_model, primary_score, alternate_score
    if not primary.ok and not alternate.ok:
        return primary, primary_model, primary_score, alternate_score

    p_proxy = primary_score.proxy if primary_score and primary_score.ok else -1.0
    a_proxy = alternate_score.proxy if alternate_score and alternate_score.ok else -1.0
    if a_proxy > p_proxy:
        return alternate, alternate_model, primary_score, alternate_score
    return primary, primary_model, primary_score, alternate_score


def cross_judge_describes(
    *,
    frames_jpeg: list[bytes],
    primary_text: str,
    primary_model: str,
    alternate_text: str,
    alternate_model: str,
    client: OpenAI | None = None,
    parallel: bool = True,
) -> tuple[DescribeJudgeScore, DescribeJudgeScore]:
    """Cross-score: alternate model judges primary text; primary judges alternate."""

    def _judge_primary() -> DescribeJudgeScore:
        # Qwen (alt) rates M3 (primary) text
        judge_client = resolve_llm_client(alternate_model, fallback=client)
        return judge_describe_call(
            client=judge_client,
            model=alternate_model,
            frames_jpeg=frames_jpeg,
            description=primary_text,
        )

    def _judge_alternate() -> DescribeJudgeScore:
        # M3 (primary) rates Qwen (alt) text
        judge_client = resolve_llm_client(primary_model, fallback=client)
        return judge_describe_call(
            client=judge_client,
            model=primary_model,
            frames_jpeg=frames_jpeg,
            description=alternate_text,
        )

    if parallel:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_p = pool.submit(_judge_primary)
            fut_a = pool.submit(_judge_alternate)
            return fut_p.result(), fut_a.result()
    return _judge_primary(), _judge_alternate()
