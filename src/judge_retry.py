"""Pipelined LLM judge + time-budgeted caption retries (per clip, overlaps caption path)."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from src.caption import (
    generate_styled_caption_from_text,
    get_fireworks_client,
    public_caption_result,
    resolve_caption_model_pool,
)
from src.caption_selector import generate_best_of_n_caption
from src.env import get_float_env, get_int_env
from src.llm_judge import (
    ClipJudgeResult,
    judge_clip_call,
)
from src.results import DescribeResult
from src.run_log import log_human


def judge_retry_enabled() -> bool:
    return get_int_env("JUDGE_RETRY", 0) == 1


def remember_description(store: dict[str, str], task_id: str, describe: DescribeResult) -> None:
    if describe.ok and describe.text:
        store[task_id] = describe.text.strip()


def list_judge_failures(clip: ClipJudgeResult, *, min_score: int) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    for style, score in clip.captions.items():
        if not score.passes(min_score=min_score):
            failures.append((clip.task_id, style))
    return failures


def _regenerate_style_caption(
    *,
    client: OpenAI,
    model: str,
    style: str,
    description: str,
) -> str:
    pool = resolve_caption_model_pool()
    if len(pool) > 1:
        candidate = generate_best_of_n_caption(
            client=client,
            models=pool,
            style=style,
            description=description,
        )
        return candidate.text
    result = generate_styled_caption_from_text(
        client=client,
        model=model,
        style=style,
        description=description,
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
    ) -> None:
        self._results = results
        self._results_path = results_path
        self._descriptions = descriptions
        self._caption_client = caption_client
        self._caption_model = caption_model
        self._run_start = run_start
        self._time_budget_s = time_budget_s
        self._total_clips = total_clips
        self._lock = threading.Lock()
        self._retries_done = 0
        self._logged_estimate = False
        self._judge_client = get_fireworks_client()
        self._judge_model = _resolve_judge_model()
        self._min_score = get_int_env("JUDGE_MIN_SCORE", 3)
        self._reserve_s = get_float_env("JUDGE_RETRY_RESERVE_S", 18.0)
        self._min_remaining = get_float_env("JUDGE_MIN_REMAINING_S", 90.0)
        self._est_per_clip = get_float_env("JUDGE_ESTIMATE_PER_CLIP_S", 12.0)
        self._skip_distinctness = get_int_env("JUDGE_SKIP_DISTINCTNESS", 1) == 1
        self._parallel_styles = get_int_env("JUDGE_PARALLEL_STYLES", 1) == 1
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge-retry")
        self._futures: list[Future[None]] = []

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
            client=self._judge_client,
            model=self._judge_model,
            task_id=task_id,
            captions={str(k): str(v) for k, v in captions.items()},
            description=description,
            skip_distinctness=self._skip_distinctness,
            parallel_styles=self._parallel_styles,
        )
        passing = clip.passing_styles(min_score=self._min_score)
        log_human(
            f"judge retry: {task_id} -> {passing}/{clip.total_styles()} pass"
        )

        for _task_id, style in list_judge_failures(clip, min_score=self._min_score):
            if not self._try_retry():
                log_human("judge retry: stopping retries (budget exhausted)")
                break
            log_human(f"judge retry: re-caption {_task_id}/{style}...")
            new_text = _regenerate_style_caption(
                client=self._caption_client,
                model=self._caption_model,
                style=style,
                description=description,
            )
            with self._lock:
                caps = result.get("captions")
                if isinstance(caps, dict):
                    caps[style] = new_text
                self._results_path.parent.mkdir(parents=True, exist_ok=True)
                self._results_path.write_text(
                    json.dumps(self._results, indent=2),
                    encoding="utf-8",
                )
            log_human(f"judge retry: updated {_task_id}/{style}")

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
    )
