"""LLM-as-judge scoring for caption eval (style fit, accuracy, specificity)."""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import APIConnectionError, APITimeoutError, OpenAI

from src.caption import STYLES, get_fireworks_client
from src.env import get_int_env
from src.eval_paths import DESCRIPTIONS_FULL
from src.pipeline import load_descriptions_cache
from src.scoring import is_structural_failure

_REPO_ROOT = Path(__file__).resolve().parent.parent
_JUDGE_STYLE_PROMPT_PATH = _REPO_ROOT / "prompts" / "judge_style.txt"
_JUDGE_DISTINCTNESS_PROMPT_PATH = _REPO_ROOT / "prompts" / "judge_distinctness.txt"
_DEFAULT_DESCRIPTIONS_PATH = DESCRIPTIONS_FULL

DEFAULT_JUDGE_PANEL_MODELS = (
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/glm-5p1",
    "accounts/fireworks/models/deepseek-v4-flash",
)

DIMENSIONS = ("style_fit", "accuracy", "specificity")
_SCORE_FIELD_RE = re.compile(
    r'"(style_fit|accuracy|specificity)"\s*:\s*(\d+)',
    re.I,
)
_ISSUE_FIELD_RE = re.compile(r'"issue"\s*:\s*"(.*?)"', re.I | re.S)
_DISTINCTNESS_RE = re.compile(r'"cross_style_distinctness"\s*:\s*(\d+)', re.I)


@dataclass
class CaptionJudgeScore:
    style: str
    style_fit: int
    accuracy: int
    specificity: int
    issue: str = ""
    skipped: bool = False
    skip_reason: str = ""

    @property
    def average(self) -> float:
        return (self.style_fit + self.accuracy + self.specificity) / 3.0

    def passes(self, *, min_score: int) -> bool:
        if self.skipped:
            return False
        return (
            self.style_fit >= min_score
            and self.accuracy >= min_score
            and self.specificity >= min_score
        )


@dataclass
class ClipJudgeResult:
    task_id: str
    captions: dict[str, CaptionJudgeScore] = field(default_factory=dict)
    cross_style_distinctness: int = 0
    distinctness_note: str = ""
    parse_error: str = ""

    def passing_styles(self, *, min_score: int) -> int:
        return sum(1 for c in self.captions.values() if c.passes(min_score=min_score))

    def total_styles(self) -> int:
        return len(self.captions)


@dataclass
class JudgeFileResult:
    clips: list[ClipJudgeResult]
    model: str
    min_score: int
    descriptions_provided: bool
    panel_models: list[str] = field(default_factory=list)
    per_judge: dict[str, "JudgeFileResult"] = field(default_factory=dict)

    @property
    def is_panel(self) -> bool:
        return bool(self.panel_models)

    @property
    def passes(self) -> int:
        return sum(c.passing_styles(min_score=self.min_score) for c in self.clips)

    @property
    def total(self) -> int:
        return sum(c.total_styles() for c in self.clips)

    def low_distinctness(self) -> list[str]:
        threshold = self.min_score
        out: list[str] = []
        for clip in self.clips:
            if clip.cross_style_distinctness and clip.cross_style_distinctness < threshold:
                note = clip.distinctness_note or "low distinctness"
                out.append(f"{clip.task_id}: distinctness={clip.cross_style_distinctness} ({note})")
        return out

    def failures(self) -> list[str]:
        fails: list[str] = []
        for clip in self.clips:
            for style, score in clip.captions.items():
                if score.skipped:
                    fails.append(f"{clip.task_id}/{style}: {score.skip_reason}")
                    continue
                if not score.passes(min_score=self.min_score):
                    weak = [
                        name
                        for name in DIMENSIONS
                        if getattr(score, name) < self.min_score
                    ]
                    detail = score.issue or ", ".join(weak)
                    dims = (
                        f"style_fit={score.style_fit} "
                        f"accuracy={score.accuracy} "
                        f"specificity={score.specificity}"
                    )
                    fails.append(f"{clip.task_id}/{style}: judge {dims} ({detail})")
        return fails


def load_judge_style_prompt() -> str:
    return _JUDGE_STYLE_PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_judge_distinctness_prompt() -> str:
    return _JUDGE_DISTINCTNESS_PROMPT_PATH.read_text(encoding="utf-8").strip()


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


def default_descriptions_path() -> Path | None:
    return _DEFAULT_DESCRIPTIONS_PATH if _DEFAULT_DESCRIPTIONS_PATH.is_file() else None


def resolve_descriptions_path(path: Path | None) -> Path | None:
    if path is not None:
        return path
    return default_descriptions_path()


def load_descriptions(path: Path | None) -> dict[str, str]:
    resolved = resolve_descriptions_path(path)
    if resolved is None:
        return {}
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "descriptions" in data:
        return load_descriptions_cache(resolved)
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items() if v}
    raise ValueError("descriptions file must be a task_id -> text map or cache with descriptions key")


def _clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(1, min(5, score))


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))


def resolve_judge_models(*, panel: bool = False, override: list[str] | None = None) -> list[str]:
    if override:
        models = [m.strip() for m in override if m.strip()]
        if models:
            return models
    env_panel = os.environ.get("JUDGE_MODELS", "").strip()
    if env_panel:
        models = [m.strip() for m in env_panel.split(",") if m.strip()]
        if models:
            return models
    if panel:
        return list(DEFAULT_JUDGE_PANEL_MODELS)
    single = os.environ.get("JUDGE_MODEL", "").strip()
    if not single:
        single = os.environ.get("CAPTION_MODEL", "accounts/fireworks/models/deepseek-v4-flash")
    return [single]


def _model_short_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def aggregate_clip_judges(per_judge: dict[str, ClipJudgeResult]) -> ClipJudgeResult:
    if not per_judge:
        raise ValueError("per_judge must not be empty")
    task_id = next(iter(per_judge.values())).task_id
    aggregated = ClipJudgeResult(task_id=task_id)

    for style in STYLES:
        active_scores: list[CaptionJudgeScore] = []
        skip_reasons: list[str] = []
        for clip in per_judge.values():
            score = clip.captions.get(style)
            if score is None:
                continue
            if score.skipped:
                if score.skip_reason:
                    skip_reasons.append(score.skip_reason)
                continue
            active_scores.append(score)

        if not active_scores:
            reason = skip_reasons[0] if skip_reasons else "judge-missing-style"
            aggregated.captions[style] = CaptionJudgeScore(
                style=style,
                style_fit=0,
                accuracy=0,
                specificity=0,
                issue=reason,
                skipped=True,
                skip_reason=reason,
            )
            continue

        issues = [s.issue for s in active_scores if s.issue]
        aggregated.captions[style] = CaptionJudgeScore(
            style=style,
            style_fit=_median_int([s.style_fit for s in active_scores]),
            accuracy=_median_int([s.accuracy for s in active_scores]),
            specificity=_median_int([s.specificity for s in active_scores]),
            issue="; ".join(dict.fromkeys(issues))[:240],
        )

    distinctness_vals = [
        clip.cross_style_distinctness
        for clip in per_judge.values()
        if clip.cross_style_distinctness
    ]
    aggregated.cross_style_distinctness = _median_int(distinctness_vals)
    notes = [clip.distinctness_note for clip in per_judge.values() if clip.distinctness_note]
    aggregated.distinctness_note = notes[0] if notes else ""
    parse_errors = [clip.parse_error for clip in per_judge.values() if clip.parse_error]
    aggregated.parse_error = "; ".join(dict.fromkeys(parse_errors))
    return aggregated


def _auto_skip_caption(text: str, style: str) -> CaptionJudgeScore | None:
    is_fail, reason = is_structural_failure(text)
    if not is_fail:
        return None
    return CaptionJudgeScore(
        style=style,
        style_fit=0,
        accuracy=0,
        specificity=0,
        issue=reason,
        skipped=True,
        skip_reason=reason,
    )


def parse_judge_response(raw: str, *, styles: tuple[str, ...] = STYLES) -> tuple[dict[str, CaptionJudgeScore], int, str, str]:
    """Parse batch clip judge JSON (legacy)."""
    try:
        payload = json.loads(_extract_json_object(raw))
    except json.JSONDecodeError as exc:
        return {}, 0, "", f"InvalidJSON: {exc}"

    if not isinstance(payload, dict):
        return {}, 0, "", "InvalidJSON: root must be object"

    captions_raw = payload.get("captions")
    if not isinstance(captions_raw, dict):
        return {}, 0, "", "InvalidJSON: missing captions object"

    scores: dict[str, CaptionJudgeScore] = {}
    for style in styles:
        entry = captions_raw.get(style)
        if not isinstance(entry, dict):
            scores[style] = CaptionJudgeScore(
                style=style,
                style_fit=0,
                accuracy=0,
                specificity=0,
                issue="missing from judge response",
                skipped=True,
                skip_reason="judge-missing-style",
            )
            continue
        scores[style] = CaptionJudgeScore(
            style=style,
            style_fit=_clamp_score(entry.get("style_fit")),
            accuracy=_clamp_score(entry.get("accuracy")),
            specificity=_clamp_score(entry.get("specificity")),
            issue=str(entry.get("issue") or "").strip(),
        )

    distinctness = _clamp_score(payload.get("cross_style_distinctness"))
    note = str(payload.get("distinctness_note") or "").strip()
    return scores, distinctness, note, ""


def _payload_from_lenient_json(raw: str) -> dict[str, Any] | None:
    text = _extract_json_object(raw)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fields = {name.lower(): int(value) for name, value in _SCORE_FIELD_RE.findall(text)}
    if len(fields) < 3:
        return None
    issue_match = _ISSUE_FIELD_RE.search(text)
    if issue_match:
        fields["issue"] = issue_match.group(1).strip()
    else:
        fields["issue"] = ""
    return fields


def parse_style_judge_response(raw: str, *, style: str) -> tuple[CaptionJudgeScore | None, str]:
    if not raw.strip():
        return None, "EmptyResponse"
    payload = _payload_from_lenient_json(raw)
    if payload is None:
        return None, "InvalidJSON: could not parse judge scores"
    return CaptionJudgeScore(
        style=style,
        style_fit=_clamp_score(payload.get("style_fit")),
        accuracy=_clamp_score(payload.get("accuracy")),
        specificity=_clamp_score(payload.get("specificity")),
        issue=str(payload.get("issue") or "").strip(),
    ), ""


def parse_distinctness_response(raw: str) -> tuple[int, str, str]:
    if not raw.strip():
        return 0, "", "EmptyResponse"
    try:
        payload = json.loads(_extract_json_object(raw))
        if isinstance(payload, dict):
            return (
                _clamp_score(payload.get("cross_style_distinctness")),
                str(payload.get("distinctness_note") or "").strip(),
                "",
            )
    except json.JSONDecodeError:
        pass
    match = _DISTINCTNESS_RE.search(raw)
    if match:
        return _clamp_score(match.group(1)), "", ""
    return 0, "", "InvalidJSON: could not parse distinctness"


def _chat_json(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    last = ""
    for use_json_mode in (True, False):
        for attempt in range(3):
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if use_json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = client.chat.completions.create(**request_kwargs)
            except (APIConnectionError, APITimeoutError):
                time.sleep(2.0 + attempt * 1.5)
                continue
            last = (resp.choices[0].message.content or "").strip()
            if last:
                return last
            time.sleep(1.0 + attempt * 0.5)
    return last


def _judge_single_style(
    *,
    client: OpenAI,
    model: str,
    task_id: str,
    style: str,
    caption: str,
    description: str | None,
    temperature: float,
) -> tuple[CaptionJudgeScore | None, str]:
    system_prompt = load_judge_style_prompt()
    lines = [f"Task: {task_id}", f"Style: {style}", f"Caption: {caption}"]
    if description:
        lines.extend(["", "Scene facts:", description.strip()])
    else:
        lines.append("Scene facts: (not provided — score accuracy from plausibility only)")
    user_prompt = "\n".join(lines)

    last_error = ""
    for attempt in range(3):
        raw = _chat_json(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=512,
            temperature=max(temperature - attempt * 0.05, 0.05),
        )
        score, err = parse_style_judge_response(raw, style=style)
        if score is not None:
            return score, ""
        if not raw:
            last_error = "EmptyResponse"
        else:
            last_error = err
    return None, last_error


def _judge_distinctness(
    *,
    client: OpenAI,
    model: str,
    task_id: str,
    captions: dict[str, str],
    temperature: float,
) -> tuple[int, str, str]:
    system_prompt = load_judge_distinctness_prompt()
    lines = [f"Task: {task_id}", "", "Captions:"]
    for style in STYLES:
        lines.append(f"- {style}: {captions.get(style, '').strip()}")
    user_prompt = "\n".join(lines)

    last_error = ""
    for attempt in range(2):
        raw = _chat_json(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=180,
            temperature=max(temperature - attempt * 0.05, 0.05),
        )
        distinctness, note, err = parse_distinctness_response(raw)
        if not err:
            return distinctness, note, ""
        last_error = err
    return 0, "", last_error


def judge_clip_call(
    *,
    client: OpenAI,
    model: str,
    task_id: str,
    captions: dict[str, str],
    description: str | None = None,
    temperature: float = 0.2,
    skip_distinctness: bool = False,
    parallel_styles: bool = False,
) -> ClipJudgeResult:
    """Judge each style separately, optionally in parallel; distinctness optional."""
    result = ClipJudgeResult(task_id=task_id)

    def _judge_one(style: str) -> tuple[str, CaptionJudgeScore]:
        text = captions.get(style, "")
        skipped = _auto_skip_caption(text, style)
        if skipped is not None:
            return style, skipped

        score, err = _judge_single_style(
            client=client,
            model=model,
            task_id=task_id,
            style=style,
            caption=text,
            description=description,
            temperature=temperature,
        )
        if score is None:
            return style, CaptionJudgeScore(
                style=style,
                style_fit=0,
                accuracy=0,
                specificity=0,
                skipped=True,
                skip_reason=err or "judge-parse-error",
            )
        return style, score

    if parallel_styles and len(STYLES) > 1:
        with ThreadPoolExecutor(max_workers=min(len(STYLES), 4)) as pool:
            futures = [pool.submit(_judge_one, style) for style in STYLES]
            for fut in as_completed(futures):
                style, score = fut.result()
                result.captions[style] = score
    else:
        for style in STYLES:
            s, score = _judge_one(style)
            result.captions[s] = score

    if (
        not skip_distinctness
        and any(not score.skipped for score in result.captions.values())
    ):
        distinctness, note, err = _judge_distinctness(
            client=client,
            model=model,
            task_id=task_id,
            captions={s: captions.get(s, "") for s in STYLES},
            temperature=temperature,
        )
        result.cross_style_distinctness = distinctness
        result.distinctness_note = note
        if err:
            result.parse_error = err

    return result


def judge_results_data(
    data: list[dict],
    *,
    descriptions: dict[str, str] | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
    min_score: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> JudgeFileResult:
    descriptions = descriptions or {}
    judge_client = client or get_fireworks_client()
    judge_model = model or os.environ.get(
        "JUDGE_MODEL",
        os.environ.get("CAPTION_MODEL", "accounts/fireworks/models/deepseek-v4-flash"),
    )
    threshold = min_score if min_score is not None else get_int_env("JUDGE_MIN_SCORE", 3)

    clips: list[ClipJudgeResult] = []
    total = len(data)
    for index, task in enumerate(data, start=1):
        task_id = str(task["task_id"])
        if on_progress:
            on_progress(
                f"judge {_model_short_name(judge_model)} clip {index}/{total}: {task_id}"
            )
        captions = task.get("captions") or {}
        if not isinstance(captions, dict):
            captions = {}
        clip = judge_clip_call(
            client=judge_client,
            model=judge_model,
            task_id=task_id,
            captions={str(k): str(v) for k, v in captions.items()},
            description=descriptions.get(task_id) or None,
            skip_distinctness=get_int_env("JUDGE_SKIP_DISTINCTNESS", 0) == 1,
            parallel_styles=get_int_env("JUDGE_PARALLEL_STYLES", 0) == 1,
        )
        clips.append(clip)
        if on_progress:
            passing = clip.passing_styles(min_score=threshold)
            on_progress(
                f"judge {_model_short_name(judge_model)} clip {index}/{total}: "
                f"{task_id} -> {passing}/{clip.total_styles()} pass"
            )
        sleep_s = get_int_env("JUDGE_SLEEP_MS", 0) / 1000.0
        if sleep_s > 0:
            time.sleep(sleep_s)

    result = JudgeFileResult(
        clips=clips,
        model=judge_model,
        min_score=threshold,
        descriptions_provided=bool(descriptions),
    )
    if on_progress:
        on_progress(
            f"judge {_model_short_name(judge_model)} done: {result.passes}/{result.total}"
        )
    return result


def judge_results_data_panel(
    data: list[dict],
    *,
    descriptions: dict[str, str] | None = None,
    client: OpenAI | None = None,
    models: list[str] | None = None,
    min_score: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> JudgeFileResult:
    panel_models = resolve_judge_models(panel=True, override=models)
    if on_progress:
        names = ", ".join(_model_short_name(m) for m in panel_models)
        on_progress(f"panel: {len(panel_models)} judges ({names})")
    per_judge: dict[str, JudgeFileResult] = {}
    for judge_index, judge_model in enumerate(panel_models, start=1):
        if on_progress:
            on_progress(
                f"panel judge {judge_index}/{len(panel_models)}: {_model_short_name(judge_model)}"
            )

        def _judge_progress(msg: str, *, _jm: str = judge_model) -> None:
            if on_progress:
                on_progress(msg)

        per_judge[judge_model] = judge_results_data(
            data,
            descriptions=descriptions,
            client=client,
            model=judge_model,
            min_score=min_score,
            on_progress=_judge_progress,
        )

    if on_progress:
        on_progress("panel: aggregating median scores")
    aggregated_clips: list[ClipJudgeResult] = []
    for index in range(len(data)):
        per_clip = {model: result.clips[index] for model, result in per_judge.items()}
        aggregated_clips.append(aggregate_clip_judges(per_clip))

    threshold = min_score if min_score is not None else get_int_env("JUDGE_MIN_SCORE", 3)
    model_label = "panel(median): " + ", ".join(_model_short_name(m) for m in panel_models)
    result = JudgeFileResult(
        clips=aggregated_clips,
        model=model_label,
        min_score=threshold,
        descriptions_provided=bool(descriptions),
        panel_models=panel_models,
        per_judge=per_judge,
    )
    if on_progress:
        on_progress(f"panel median: {result.passes}/{result.total}")
    return result


def judge_file(
    path: Path,
    *,
    descriptions_path: Path | None = None,
    panel: bool = False,
    judge_models: list[str] | None = None,
) -> JudgeFileResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    resolved = resolve_descriptions_path(descriptions_path)
    descriptions = load_descriptions(resolved) if resolved else {}
    if panel:
        return judge_results_data_panel(data, descriptions=descriptions, models=judge_models)
    return judge_results_data(data, descriptions=descriptions)


def judge_result_to_dict(result: JudgeFileResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": result.model,
        "min_score": result.min_score,
        "descriptions_provided": result.descriptions_provided,
        "passes": result.passes,
        "total": result.total,
        "panel_models": list(result.panel_models),
        "clips": [
            {
                "task_id": clip.task_id,
                "cross_style_distinctness": clip.cross_style_distinctness,
                "distinctness_note": clip.distinctness_note,
                "parse_error": clip.parse_error,
                "captions": {
                    style: asdict(score) for style, score in clip.captions.items()
                },
            }
            for clip in result.clips
        ],
    }
    if result.per_judge:
        payload["per_judge"] = {
            model: {
                "passes": sub.passes,
                "total": sub.total,
                "model": sub.model,
            }
            for model, sub in result.per_judge.items()
        }
    return payload


def collect_calibration_samples(
    result: JudgeFileResult,
    data: list[dict],
    *,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Borderline captions near the pass threshold for human spot-check."""
    by_id = {str(task["task_id"]): task for task in data}
    samples: list[tuple[float, dict[str, Any]]] = []

    for clip in result.clips:
        task = by_id.get(clip.task_id, {})
        captions = task.get("captions") or {}
        for style, score in clip.captions.items():
            if score.skipped:
                continue
            min_dim = min(score.style_fit, score.accuracy, score.specificity)
            passed = score.passes(min_score=result.min_score)
            near_threshold = (
                min_dim in {result.min_score - 1, result.min_score, result.min_score + 1}
                or abs(score.average - result.min_score) <= 0.75
            )
            if not near_threshold:
                continue
            distance = abs(min_dim - result.min_score) + abs(score.average - result.min_score) * 0.25
            if not passed:
                distance -= 0.1
            samples.append(
                (
                    distance,
                    {
                        "task_id": clip.task_id,
                        "style": style,
                        "passed": passed,
                        "style_fit": score.style_fit,
                        "accuracy": score.accuracy,
                        "specificity": score.specificity,
                        "issue": score.issue,
                        "caption": str(captions.get(style, "")),
                    },
                )
            )

    samples.sort(key=lambda item: item[0])
    return [item[1] for item in samples[:limit]]


def format_calibration_report(samples: list[dict[str, Any]]) -> str:
    if not samples:
        return "No borderline captions found near the current threshold."
    lines = [f"Borderline captions for human spot-check ({len(samples)}):", ""]
    for index, sample in enumerate(samples, start=1):
        verdict = "PASS" if sample["passed"] else "FAIL"
        lines.append(f"{index}. [{verdict}] {sample['task_id']}/{sample['style']}")
        lines.append(
            f"   scores: style_fit={sample['style_fit']} "
            f"accuracy={sample['accuracy']} specificity={sample['specificity']}"
        )
        if sample.get("issue"):
            lines.append(f"   issue: {sample['issue']}")
        lines.append(f"   caption: {sample['caption']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_judge_summary(result: JudgeFileResult) -> str:
    lines = [
        f"{result.passes}/{result.total}",
        f"  (judge min={result.min_score}, model={result.model})",
    ]
    if result.is_panel and result.per_judge:
        lines.append("  per-judge:")
        for model, sub in result.per_judge.items():
            lines.append(f"    {_model_short_name(model)}: {sub.passes}/{sub.total}")
    if not result.descriptions_provided:
        lines.append("  (no scene facts — accuracy is plausibility-only)")
    for fail in result.failures():
        lines.append(f"  {fail}")
    warnings = result.low_distinctness()
    if warnings:
        lines.append("distinctness warnings:")
        for w in warnings:
            lines.append(f"  {w}")
    return "\n".join(lines)
