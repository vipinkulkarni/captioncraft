"""Pipelined LLM judge + time-budgeted caption retries (per clip, overlaps caption path)."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from src.caption import (
    generate_styled_caption_from_text,
    get_fireworks_client,
    public_caption_result,
    resolve_caption_model_pool,
)
from src.caption_selector import generate_best_of_n_caption
from src.caption_vision_judge import (
    caption_vision_accuracy_enabled,
    caption_vision_accuracy_mode,
    judge_caption_vision_accuracy,
    resolve_caption_vision_judge_model,
    vision_accuracy_target_styles,
)
from src.env import get_float_env, get_int_env
from src.llm_judge import (
    CaptionJudgeScore,
    ClipJudgeResult,
    _judge_single_style,
    judge_clip_call,
    resolve_judge_min_score,
    resolve_judge_quality_floor,
)
from src.results import DescribeResult
from src.run_log import log_human
from src.scoring import score_caption


def judge_retry_enabled() -> bool:
    return get_int_env("JUDGE_RETRY", 0) == 1


def remember_description(store: dict[str, str], task_id: str, describe: DescribeResult) -> None:
    if describe.ok and describe.text:
        store[task_id] = describe.text.strip()


def remember_frames(store: dict[str, list[bytes]], task_id: str, frames: list[bytes]) -> None:
    if task_id and frames:
        store[task_id] = frames


def list_judge_failures(
    clip: ClipJudgeResult,
    *,
    min_score: float,
    captions: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    quality_floor = resolve_judge_quality_floor()
    check_regex = get_int_env("JUDGE_RETRY_REGEX", 1) == 1

    for style, score in clip.captions.items():
        key = (clip.task_id, style)
        if score.skipped:
            if key not in seen:
                failures.append(key)
                seen.add(key)
            continue
        if not score.passes(min_score=min_score):
            if key not in seen:
                failures.append(key)
                seen.add(key)
            continue
        if score.accuracy < quality_floor or score.style_match < quality_floor:
            if key not in seen:
                failures.append(key)
                seen.add(key)

    if check_regex and captions:
        for style, text in captions.items():
            key = (clip.task_id, str(style))
            if key in seen:
                continue
            ok, _reason = score_caption(str(text), str(style))
            if not ok:
                failures.append(key)
                seen.add(key)

    return failures


def style_still_fails(
    score: CaptionJudgeScore,
    *,
    task_id: str,
    style: str,
    caption: str,
    min_score: float,
) -> bool:
    clip = ClipJudgeResult(task_id=task_id, captions={style: score})
    return bool(
        list_judge_failures(clip, min_score=min_score, captions={style: caption})
    )


def judge_feedback_nudge(score: CaptionJudgeScore) -> str:
    issue = score.issue or score.skip_reason or "did not meet accuracy/style bar"
    if score.meta_leak:
        return (
            f"Previous output was drafting/meta-leak (not a finished caption). "
            f"Issue: {issue}. "
            "Output ONLY the finished caption in the requested tone — no planning, "
            "self-critique, 'but careful', 'revised:', or instructions about writing."
        )
    return (
        f"Previous caption scored accuracy={score.accuracy:.2f} "
        f"style_match={score.style_match:.2f}. Issue: {issue}. "
        "Rewrite using only scene facts; do not invent or rename objects, species, "
        "UI, code, or emotions not listed. Metaphors must not add new scene objects."
    )


def _apply_vision_accuracy(
    *,
    clip: ClipJudgeResult,
    captions: dict[str, str],
    frames: list[bytes],
    client: OpenAI,
    model: str,
    styles: list[str] | None = None,
) -> None:
    """Overwrite text accuracy with min(text, vision) when vision judge succeeds."""
    target = styles if styles is not None else list(clip.captions.keys())
    work: list[tuple[str, CaptionJudgeScore, str]] = []
    for style in target:
        score = clip.captions.get(style)
        if score is None or score.skipped:
            continue
        text = captions.get(style, "")
        if not text.strip():
            continue
        work.append((style, score, text))

    if not work:
        return

    def _one(style: str, score: CaptionJudgeScore, text: str):
        vis = judge_caption_vision_accuracy(
            client=client,
            model=model,
            frames_jpeg=frames,
            caption=text,
        )
        return style, score, vis

    results = []
    if len(work) == 1:
        results.append(_one(*work[0]))
    else:
        with ThreadPoolExecutor(max_workers=min(len(work), 3)) as pool:
            futs = [pool.submit(_one, *item) for item in work]
            for fut in as_completed(futs):
                results.append(fut.result())

    for style, score, vis in results:
        if not vis.ok:
            log_human(
                f"judge retry: vision accuracy skip {clip.task_id}/{style} "
                f"({vis.parse_error})"
            )
            continue
        if not vis.usable:
            log_human(
                f"judge retry: vision accuracy skip {clip.task_id}/{style} "
                f"(low confidence={vis.confidence:.2f})"
            )
            continue
        if vis.accuracy < score.accuracy:
            # Mild 0.85-style demotions create retry storms without helping quality.
            demote_max = get_float_env("CAPTION_VISION_DEMOTE_MAX", 0.75)
            if vis.accuracy >= demote_max:
                log_human(
                    f"judge retry: vision accuracy soft {clip.task_id}/{style} "
                    f"{score.accuracy:.2f}->{vis.accuracy:.2f} (no demote ≥{demote_max:.2f})"
                )
                continue
            note = vis.issue or "vision accuracy below text accuracy"
            merged_issue = score.issue
            if note:
                merged_issue = (
                    f"{merged_issue}; vision: {note}" if merged_issue else f"vision: {note}"
                )
            clip.captions[style] = CaptionJudgeScore(
                style=style,
                accuracy=vis.accuracy,
                style_match=score.style_match,
                issue=merged_issue,
            )
            log_human(
                f"judge retry: vision accuracy {clip.task_id}/{style} "
                f"{score.accuracy:.2f}->{vis.accuracy:.2f}"
            )


def _regenerate_style_caption(
    *,
    client: OpenAI,
    model: str,
    style: str,
    description: str,
    task_id: str = "",
    judge_feedback: str | None = None,
) -> str:
    diversity = get_int_env("JUDGE_RETRY_DIVERSITY", 1) == 1
    retry_temp = get_float_env("JUDGE_RETRY_TEMPERATURE", 0.92) if diversity else None
    pool = resolve_caption_model_pool()
    if len(pool) > 1:
        candidate = generate_best_of_n_caption(
            client=client,
            models=pool,
            style=style,
            description=description,
            diversity_retry=diversity,
            task_id=task_id or "retry",
            judge_feedback=judge_feedback,
        )
        return candidate.text
    result = generate_styled_caption_from_text(
        client=client,
        model=model,
        style=style,
        description=description,
        temperature_override=retry_temp,
        diversity_retry=diversity,
        judge_feedback=judge_feedback,
    )
    return public_caption_result(result, style=style)


def _resolve_judge_model() -> str:
    return os.environ.get(
        "JUDGE_MODEL",
        os.environ.get("CAPTION_MODEL", "accounts/fireworks/models/deepseek-v4-flash"),
    )


class PipelinedJudgeRetry:
    """Judge each clip as soon as it is captioned; retry failures while the pipeline runs."""

    def __init__(
        self,
        *,
        results: list[dict],
        results_path: Path,
        descriptions: dict[str, str],
        caption_client: OpenAI,
        caption_model: str,
        run_start: float,
        time_budget_s: float,
        total_clips: int,
        judge_client: OpenAI | None = None,
        frames_by_task: dict[str, list[bytes]] | None = None,
    ) -> None:
        self._results = results
        self._results_path = results_path
        self._descriptions = descriptions
        self._frames_by_task = frames_by_task if frames_by_task is not None else {}
        self._caption_client = caption_client
        self._caption_model = caption_model
        self._run_start = run_start
        self._time_budget_s = time_budget_s
        self._total_clips = total_clips
        self._lock = threading.Lock()
        self._retries_done = 0
        self._logged_estimate = False
        self._judge_client = judge_client
        self._judge_model = _resolve_judge_model()
        self._vision_judge_model = resolve_caption_vision_judge_model()
        self._vision_accuracy = caption_vision_accuracy_enabled()
        self._min_score = resolve_judge_min_score()
        self._reserve_s = get_float_env("JUDGE_RETRY_RESERVE_S", 18.0)
        self._min_remaining = get_float_env("JUDGE_MIN_REMAINING_S", 90.0)
        self._est_per_clip = get_float_env("JUDGE_ESTIMATE_PER_CLIP_S", 12.0)
        self._skip_distinctness = get_int_env("JUDGE_SKIP_DISTINCTNESS", 1) == 1
        self._parallel_styles = get_int_env("JUDGE_PARALLEL_STYLES", 1) == 1
        self._max_per_style = max(get_int_env("JUDGE_RETRY_MAX_PER_STYLE", 2), 0)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge-retry")
        self._futures: list[Future[None]] = []

    def _get_judge_client(self) -> OpenAI:
        if self._judge_client is None:
            self._judge_client = get_fireworks_client()
        return self._judge_client

    def remaining_s(self) -> float:
        if self._time_budget_s <= 0:
            return float("inf")
        return self._time_budget_s - (time.monotonic() - self._run_start)

    def submit(self, result: dict, *, clip_index: int) -> None:
        if not judge_retry_enabled():
            return
        task_id = str(result.get("task_id", ""))
        if not task_id or not self._descriptions.get(task_id, ""):
            return
        future = self._executor.submit(self._process_clip, result, clip_index)
        self._futures.append(future)

    def _log_budget_estimate(self, *, clip_index: int) -> None:
        if self._logged_estimate or self._time_budget_s <= 0:
            return
        self._logged_estimate = True
        remaining = self.remaining_s()
        clips_left = max(self._total_clips - clip_index, 0)
        tail_est = clips_left * self._est_per_clip
        log_human(
            f"judge retry: {remaining:.0f}s budget left, "
            f"~{self._est_per_clip:.0f}s/clip est, "
            f"~{tail_est:.0f}s judge tail if caption pipeline finishes now"
        )

    def _persist_caption(self, result: dict, style: str, text: str) -> None:
        with self._lock:
            caps = result.get("captions")
            if isinstance(caps, dict):
                caps[style] = text
            self._results_path.parent.mkdir(parents=True, exist_ok=True)
            self._results_path.write_text(
                json.dumps(self._results, indent=2),
                encoding="utf-8",
            )

    def _process_clip(self, result: dict, clip_index: int) -> None:
        task_id = str(result.get("task_id", ""))
        description = self._descriptions.get(task_id, "")
        if not description:
            return

        remaining = self.remaining_s()
        if self._time_budget_s > 0 and remaining < self._min_remaining:
            log_human(
                f"judge retry: skip {task_id} "
                f"({remaining:.0f}s left < {self._min_remaining:.0f}s min)"
            )
            return

        self._log_budget_estimate(clip_index=clip_index)
        log_human(
            f"judge retry: scoring {task_id} ({clip_index + 1}/{self._total_clips})..."
        )

        captions = result.get("captions") or {}
        if not isinstance(captions, dict):
            captions = {}

        clip = judge_clip_call(
            client=self._get_judge_client(),
            model=self._judge_model,
            task_id=task_id,
            captions={str(k): str(v) for k, v in captions.items()},
            description=description,
            skip_distinctness=self._skip_distinctness,
            parallel_styles=self._parallel_styles,
        )
        frames = self._frames_by_task.get(task_id) or []
        caption_map = {str(k): str(v) for k, v in captions.items()}
        if self._vision_accuracy and frames:
            text_failures = {
                style
                for _tid, style in list_judge_failures(
                    clip,
                    min_score=self._min_score,
                    captions=caption_map,
                )
            }
            target_styles = vision_accuracy_target_styles(
                list(clip.captions.keys()),
                failing_styles=text_failures,
            )
            if target_styles:
                log_human(
                    f"judge retry: vision accuracy mode="
                    f"{caption_vision_accuracy_mode()} "
                    f"styles={','.join(target_styles)}"
                )
                _apply_vision_accuracy(
                    clip=clip,
                    captions=caption_map,
                    frames=frames,
                    client=self._get_judge_client(),
                    model=self._vision_judge_model,
                    styles=target_styles,
                )
        passing = clip.passing_styles(min_score=self._min_score)
        log_human(
            f"judge retry: {task_id} -> {passing}/{clip.total_styles()} pass"
        )

        for _task_id, style in list_judge_failures(
            clip,
            min_score=self._min_score,
            captions={str(k): str(v) for k, v in captions.items()},
        ):
            best_text = str(captions.get(style, ""))
            best_score = clip.captions.get(style) or CaptionJudgeScore(
                style=style,
                accuracy=0.0,
                style_match=0.0,
                skipped=True,
                skip_reason="missing-score",
            )
            stopped_budget = False
            for attempt in range(1, self._max_per_style + 1):
                if not style_still_fails(
                    best_score,
                    task_id=task_id,
                    style=style,
                    caption=best_text,
                    min_score=self._min_score,
                ):
                    break
                if not self._try_retry():
                    log_human("judge retry: stopping retries (budget exhausted)")
                    stopped_budget = True
                    break
                feedback = judge_feedback_nudge(best_score)
                log_human(
                    f"judge retry: re-caption {_task_id}/{style} "
                    f"attempt {attempt}/{self._max_per_style}..."
                )
                new_text = _regenerate_style_caption(
                    client=self._caption_client,
                    model=self._caption_model,
                    style=style,
                    description=description,
                    task_id=task_id,
                    judge_feedback=feedback,
                )
                new_score, err = _judge_single_style(
                    client=self._get_judge_client(),
                    model=self._judge_model,
                    task_id=task_id,
                    style=style,
                    caption=new_text,
                    description=description,
                    temperature=0.2,
                )
                if new_score is None:
                    log_human(
                        f"judge retry: re-judge failed {_task_id}/{style} "
                        f"({err or 'parse-error'}); keeping prior caption"
                    )
                    continue
                if self._vision_accuracy and frames:
                    temp_clip = ClipJudgeResult(
                        task_id=task_id,
                        captions={style: new_score},
                    )
                    _apply_vision_accuracy(
                        clip=temp_clip,
                        captions={style: new_text},
                        frames=frames,
                        client=self._get_judge_client(),
                        model=self._vision_judge_model,
                    )
                    new_score = temp_clip.captions[style]
                if (
                    best_score.skipped
                    or new_score.average > best_score.average
                    or not style_still_fails(
                        new_score,
                        task_id=task_id,
                        style=style,
                        caption=new_text,
                        min_score=self._min_score,
                    )
                ):
                    best_text = new_text
                    best_score = new_score
                    clip.captions[style] = best_score
                    self._persist_caption(result, style, best_text)
                    log_human(
                        f"judge retry: updated {_task_id}/{style} "
                        f"mean={best_score.average:.3f} "
                        f"(acc={best_score.accuracy:.2f} "
                        f"style={best_score.style_match:.2f})"
                    )
                else:
                    log_human(
                        f"judge retry: kept prior {_task_id}/{style} "
                        f"(new mean={new_score.average:.3f} "
                        f"< best={best_score.average:.3f})"
                    )
            if stopped_budget:
                break
            if style_still_fails(
                best_score,
                task_id=task_id,
                style=style,
                caption=best_text,
                min_score=self._min_score,
            ):
                log_human(
                    f"judge retry: capped {_task_id}/{style} "
                    f"best mean={best_score.average:.3f}"
                )

    def _try_retry(self) -> bool:
        with self._lock:
            if self._time_budget_s > 0 and self.remaining_s() < self._reserve_s:
                return False
            self._retries_done += 1
            return True

    def finish(self) -> None:
        self._executor.shutdown(wait=True)
        for future in self._futures:
            future.result()
        if self._retries_done:
            log_human(
                f"judge retry: done ({self._retries_done} caption(s) retried) "
                f"-> {self._results_path}"
            )


def create_pipelined_judge_retry(
    *,
    results: list[dict],
    results_path: Path,
    descriptions: dict[str, str],
    caption_client: OpenAI,
    caption_model: str,
    run_start: float,
    time_budget_s: float,
    total_clips: int,
    frames_by_task: dict[str, list[bytes]] | None = None,
) -> PipelinedJudgeRetry | None:
    if not judge_retry_enabled():
        return None
    if total_clips <= 0:
        log_human("judge retry: skipped (no clips)")
        return None
    if time_budget_s <= 0:
        log_human("judge retry: skipped (TIME_BUDGET_S not set)")
        return None
    return PipelinedJudgeRetry(
        results=results,
        results_path=results_path,
        descriptions=descriptions,
        caption_client=caption_client,
        caption_model=caption_model,
        run_start=run_start,
        time_budget_s=time_budget_s,
        total_clips=total_clips,
        frames_by_task=frames_by_task,
    )
