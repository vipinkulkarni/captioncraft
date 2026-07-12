"""Vision-backed caption accuracy (caption vs video frames)."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.caption import _to_data_url
from src.env import get_float_env, get_int_env
from src.llm_clients import fireworks_extra_body, is_google_ai_model, resolve_llm_client

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_PATH = _REPO_ROOT / "prompts" / "judge_caption_accuracy.txt"

_DEFAULT_VISION_JUDGE = "accounts/fireworks/models/minimax-m3"
_DEFAULT_VISION_JUDGE_ALT = "accounts/fireworks/models/kimi-k2p6"
_ACCURACY_RE = re.compile(r'"accuracy"\s*:\s*([0-9]*\.?[0-9]+)', re.I)
_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', re.I)
_ISSUE_RE = re.compile(r'"issue"\s*:\s*"(.*?)"', re.I | re.S)


@dataclass
class CaptionVisionAccuracy:
    accuracy: float
    issue: str = ""
    confidence: float = 1.0
    judge_model: str = ""
    parse_error: str = ""

    @property
    def ok(self) -> bool:
        return not self.parse_error

    @property
    def usable(self) -> bool:
        """True when parse succeeded and confidence clears the eval threshold."""
        if not self.ok:
            return False
        return self.confidence >= vision_judge_min_confidence()


@dataclass
class CaptionVisionPanelScore:
    accuracy: float
    confidence: float
    issue: str = ""
    disagreement: float = 0.0
    members: list[CaptionVisionAccuracy] = field(default_factory=list)
    parse_error: str = ""

    @property
    def ok(self) -> bool:
        """True when at least one judge produced a score."""
        return bool(self.members) and any(m.ok for m in self.members)

    @property
    def usable(self) -> bool:
        """Decision-grade: full panel, high confidence, low disagreement."""
        if not self.members or not all(m.ok for m in self.members):
            return False
        if self.confidence < vision_judge_min_confidence():
            return False
        if self.disagreement > vision_judge_max_disagreement():
            return False
        return True


def resolve_caption_vision_judge_model() -> str:
    return (
        os.environ.get("CAPTION_VISION_JUDGE_MODEL", "").strip()
        or _DEFAULT_VISION_JUDGE
    )


def resolve_caption_vision_judge_alt_model() -> str:
    return (
        os.environ.get("CAPTION_VISION_JUDGE_ALT", "").strip()
        or _DEFAULT_VISION_JUDGE_ALT
    )


def resolve_caption_vision_judge_panel() -> list[str]:
    """Models for multi-judge panel (eval). Empty env → primary+alt if distinct."""
    raw = os.environ.get("CAPTION_VISION_JUDGE_PANEL", "").strip()
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
        seen: set[str] = set()
        out: list[str] = []
        for m in models:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out
    primary = resolve_caption_vision_judge_model()
    alt = resolve_caption_vision_judge_alt_model()
    if alt and alt != primary:
        return [primary, alt]
    return [primary]


def caption_vision_accuracy_enabled() -> bool:
    """Pipeline flag: only on when explicitly enabled after composition-gap diagnostic."""
    return get_int_env("CAPTION_VISION_ACCURACY", 0) == 1


def vision_judge_min_confidence() -> float:
    return min(max(get_float_env("CAPTION_VISION_JUDGE_MIN_CONFIDENCE", 0.7), 0.0), 1.0)


def vision_judge_max_disagreement() -> float:
    return min(max(get_float_env("CAPTION_VISION_JUDGE_MAX_DISAGREE", 0.25), 0.0), 1.0)


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
            confidence=0.0,
            judge_model=judge_model,
            parse_error="EmptyResponse",
        )
    try:
        payload = json.loads(_extract_json_object(raw))
        if isinstance(payload, dict):
            accuracy = _parse_unit_score(payload.get("accuracy"))
            if accuracy is not None:
                conf_raw = payload.get("confidence", payload.get("conf"))
                confidence = _parse_unit_score(conf_raw)
                # Legacy responses without confidence: treat as high confidence.
                if confidence is None:
                    confidence = 1.0
                return CaptionVisionAccuracy(
                    accuracy=accuracy,
                    confidence=confidence,
                    issue=str(payload.get("issue") or "").strip(),
                    judge_model=judge_model,
                )
    except json.JSONDecodeError:
        pass

    match = _ACCURACY_RE.search(raw)
    if match:
        accuracy = _parse_unit_score(match.group(1))
        if accuracy is not None:
            conf_m = _CONFIDENCE_RE.search(raw)
            confidence = _parse_unit_score(conf_m.group(1)) if conf_m else 1.0
            if confidence is None:
                confidence = 1.0
            issue_m = _ISSUE_RE.search(raw)
            issue = issue_m.group(1).strip() if issue_m else ""
            return CaptionVisionAccuracy(
                accuracy=accuracy,
                confidence=confidence,
                issue=issue,
                judge_model=judge_model,
            )
    return CaptionVisionAccuracy(
        accuracy=0.0,
        confidence=0.0,
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
            confidence=0.0,
            judge_model=model,
            parse_error="GoogleCaptionVisionUnsupported",
        )
    resolved = client or resolve_llm_client(model)
    if resolved is None:
        return CaptionVisionAccuracy(
            accuracy=0.0,
            confidence=0.0,
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
                "Score accuracy and confidence against the frames above."
            ),
        }
    )
    tok = (
        max_tokens
        if max_tokens is not None
        else max(get_int_env("CAPTION_VISION_JUDGE_MAX_TOKENS", 320), 64)
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
                confidence=0.0,
                judge_model=model,
                parse_error=type(exc).__name__,
            )
        score = parse_caption_vision_accuracy_response(last_raw, judge_model=model)
        if score.ok:
            return score
    return parse_caption_vision_accuracy_response(last_raw, judge_model=model)


def aggregate_vision_panel(
    members: list[CaptionVisionAccuracy],
) -> CaptionVisionPanelScore:
    """Mean accuracy/confidence across successful judges; track disagreement."""
    ok_members = [m for m in members if m.ok]
    if not ok_members:
        err = members[0].parse_error if members else "EmptyPanel"
        return CaptionVisionPanelScore(
            accuracy=0.0,
            confidence=0.0,
            members=list(members),
            parse_error=err or "AllJudgesFailed",
        )
    accuracies = [m.accuracy for m in ok_members]
    confidences = [m.confidence for m in ok_members]
    disagreement = max(accuracies) - min(accuracies) if len(accuracies) > 1 else 0.0
    issues = [m.issue for m in ok_members if m.issue]
    issue = " | ".join(dict.fromkeys(issues))
    parse_error = ""
    if len(ok_members) < len(members):
        failed = [m.parse_error or m.judge_model for m in members if not m.ok]
        parse_error = f"PartialPanel:{','.join(failed)}"
    return CaptionVisionPanelScore(
        accuracy=round(sum(accuracies) / len(accuracies), 3),
        confidence=round(sum(confidences) / len(confidences), 3),
        disagreement=round(disagreement, 3),
        issue=issue,
        members=list(members),
        parse_error=parse_error,
    )


def judge_caption_vision_panel(
    *,
    frames_jpeg: list[bytes],
    caption: str,
    models: list[str] | None = None,
    client: OpenAI | None = None,
    parallel: bool = True,
) -> CaptionVisionPanelScore:
    """Score caption with one or more vision judges and aggregate."""
    panel = models or resolve_caption_vision_judge_panel()
    if not panel:
        return CaptionVisionPanelScore(
            accuracy=0.0,
            confidence=0.0,
            parse_error="EmptyPanel",
        )

    def _one(model: str) -> CaptionVisionAccuracy:
        return judge_caption_vision_accuracy(
            client=resolve_llm_client(model, fallback=client),
            model=model,
            frames_jpeg=frames_jpeg,
            caption=caption,
        )

    members: list[CaptionVisionAccuracy]
    if parallel and len(panel) > 1:
        with ThreadPoolExecutor(max_workers=len(panel)) as pool:
            futs = {pool.submit(_one, m): m for m in panel}
            by_model = {futs[f]: f.result() for f in as_completed(futs)}
            members = [by_model[m] for m in panel]
    else:
        members = [_one(m) for m in panel]
    return aggregate_vision_panel(members)
