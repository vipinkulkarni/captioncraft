from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from concurrent.futures import Future, ThreadPoolExecutor, as_completed

import cv2
import httpx
from openai import OpenAI

from src.caption import (
    STYLES,
    dry_run_captions,
    generate_styled_caption_from_text,
    public_caption_result,
    public_describe_result,
    public_process_failure,
    resolve_caption_model_pool,
    resolve_llm_client,
    looks_truncated,
    structured_describe_enabled,
    vision_describe_call,
)
from src.caption_selector import generate_best_of_n_caption
from src.describe_schema import parse_describe_json
from src.describe_quality import describe_quality_issue
from src.env import get_float_env, get_frame_config, get_int_env, resolve_frame_count
from src.results import DescribeResult, ProcessError, describe_error_from_reason
from src.retry import RetryPolicy, call_with_retry
from src.run_log import emit_config_event, emit_event, log_human


def _store_live_description(store: dict[str, str], task_id: str, describe: DescribeResult) -> None:
    if describe.ok and describe.text:
        store[task_id] = describe.text.strip()


def write_descriptions_cache(
    path: Path,
    descriptions: dict[str, str],
    *,
    meta: dict | None = None,
) -> None:
    payload = {
        "descriptions": {str(k): str(v) for k, v in descriptions.items() if v},
        "meta": meta or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def merge_descriptions_into_file(path: Path, live: dict[str, str]) -> None:
    """Merge live task descriptions into an existing cache file (preserve other keys)."""
    existing: dict[str, str] = {}
    meta: dict = {}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "descriptions" in data:
            raw = data.get("descriptions") or {}
            if isinstance(raw, dict):
                existing = {str(k): str(v) for k, v in raw.items() if v}
            if isinstance(data.get("meta"), dict):
                meta = dict(data["meta"])
        elif isinstance(data, dict):
            existing = {str(k): str(v) for k, v in data.items() if v}
    existing.update({str(k): str(v) for k, v in live.items() if v})
    meta = {**meta, "updated_from_live": True}
    write_descriptions_cache(path, existing, meta=meta)


def persist_live_descriptions(
    results_path: Path,
    descriptions: dict[str, str],
    *,
    update_fixture: bool,
) -> None:
    """Write run-local descriptions_live.json; optionally merge into frozen fixture."""
    if not descriptions:
        return
    live_path = results_path.parent / "descriptions_live.json"
    write_descriptions_cache(
        live_path,
        descriptions,
        meta={"source": "live_pipeline"},
    )
    if not update_fixture:
        return
    update_raw = os.environ.get("DESCRIPTIONS_UPDATE_PATH", "").strip()
    if update_raw:
        update_path = Path(update_raw)
    else:
        from src.eval_paths import DESCRIPTIONS_FULL

        update_path = DESCRIPTIONS_FULL
    merge_descriptions_into_file(update_path, descriptions)


def read_tasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tasks.json must be a JSON array")
    return data


def load_descriptions_cache(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    descriptions = data.get("descriptions", {})
    if not isinstance(descriptions, dict):
        raise ValueError("descriptions cache must contain a descriptions object")
    return {str(k): str(v) for k, v in descriptions.items() if v}


def write_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


_BAD_DOWNLOAD_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "application/json",
        "text/plain",
        "application/xml",
        "text/xml",
    }
)


def _is_bad_download_content_type(content_type: str) -> bool:
    """Reject obvious non-video responses; allow missing/generic headers."""
    base = content_type.split(";", 1)[0].strip().lower()
    if not base:
        return False
    if base in _BAD_DOWNLOAD_CONTENT_TYPES:
        return True
    return base.startswith("text/")


def download_video(url: str, dest: Path) -> None:
    read_timeout = float(os.environ.get("DOWNLOAD_READ_TIMEOUT", "180"))
    timeout = httpx.Timeout(connect=15.0, read=read_timeout, write=15.0, pool=15.0)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        content_type = (r.headers.get("content-type") or "").lower()
        if _is_bad_download_content_type(content_type):
            raise RuntimeError(f"Non-video response content-type={content_type}")
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)


def _frame_indices(frame_count: int, max_frames: int) -> list[int]:
    if max_frames <= 1:
        return [0]
    if frame_count <= 0:
        return []
    last = max(frame_count - 1, 0)
    indices = [round(i * last / (max_frames - 1)) for i in range(max_frames)]
    return sorted(set(indices))


def _frame_times_ms(duration_s: float, max_frames: int) -> list[float]:
    if max_frames <= 1:
        return [0.0]
    if duration_s <= 0:
        return []
    last_ms = max(duration_s * 1000.0, 0.0)
    return sorted(
        {round(i * last_ms / (max_frames - 1), 1) for i in range(max_frames)}
    )


def _encode_frame_jpeg(frame, width: int) -> bytes:
    h, w = frame.shape[:2]
    if w > width:
        new_h = max(int(h * (width / w)), 1)
        frame = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buf.tobytes()


def extract_frames_jpeg(
    video_path: Path,
    *,
    max_frames: int | None = None,
    width: int = 512,
    duration_hint_s: float = 0.0,
) -> tuple[list[bytes], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    duration_s = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    effective_duration = max(duration_s, max(duration_hint_s, 0.0))
    if max_frames is None:
        max_frames = resolve_frame_count(effective_duration)

    frames: list[bytes] = []
    indices = _frame_indices(frame_count, max_frames)
    if indices:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(_encode_frame_jpeg(frame, width))
    else:
        times_ms = _frame_times_ms(effective_duration, max_frames)
        if times_ms:
            for ms in times_ms:
                cap.set(cv2.CAP_PROP_POS_MSEC, ms)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frames.append(_encode_frame_jpeg(frame, width))
        else:
            for _ in range(max_frames):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(_encode_frame_jpeg(frame, width))

    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted from video")

    return frames, max(duration_s, max(duration_hint_s, 0.0))


@dataclass
class _DescribeAttempt:
    text: str
    finish_reason: str | None
    elapsed_s: float = 0.0
    error: str | None = None


def _describe_output_issue(
    text: str,
    finish_reason: str | None,
    *,
    duration_s: float = 0.0,
) -> str | None:
    if not text.strip():
        return "EmptyResponse"
    if structured_describe_enabled():
        if finish_reason == "length":
            return "Truncated"
        ok, reason, _formatted = parse_describe_json(text)
        if not ok:
            return reason or "InvalidJSON"
        quality_issue = describe_quality_issue(text, duration_s=duration_s)
        if quality_issue:
            return quality_issue
        return None
    if looks_truncated(text, finish_reason):
        return "Truncated" if text else "EmptyResponse"
    return None


def _finalize_describe_text(raw_text: str) -> tuple[str | None, str | None, str | None]:
    if structured_describe_enabled():
        ok, reason, formatted = parse_describe_json(raw_text)
        if not ok:
            return None, reason or "InvalidJSON", None
        return formatted, None, raw_text
    return raw_text, None, None


def _vision_fallback_model() -> str:
    return os.environ.get("VISION_FALLBACK_MODEL", "").strip()


def _describe_max_attempts() -> int:
    base = max(get_int_env("DESCRIBE_MAX_ATTEMPTS", 2), 1)
    if not _vision_fallback_model():
        return base
    with_fallback = get_int_env("DESCRIBE_MAX_ATTEMPTS_WITH_FALLBACK", 1)
    return max(min(with_fallback, base), 1)


def _describe_no_retry_reasons() -> frozenset[str]:
    return frozenset(
        {
            "ReadTimeout",
            "APITimeoutError",
            "Timeout",
            "ServerError",
            "InternalServerError",
        }
    )


def _should_skip_primary_for_duration(duration_s: float) -> bool:
    threshold = get_float_env("DESCRIBE_LONG_SKIP_PRIMARY_S", 0.0)
    return threshold > 0 and duration_s >= threshold


def _describe_frames(
    *,
    client: OpenAI | None,
    model: str,
    task_id: str,
    frames: list[bytes],
    duration_s: float = 0.0,
) -> DescribeResult:
    base_max = max(get_int_env("DESCRIBE_MAX_TOKENS", 1200), 64)
    temperature = get_float_env("DESCRIBE_TEMPERATURE", 0.2)
    policy = RetryPolicy(
        max_attempts=_describe_max_attempts(),
        base_sleep_s=get_float_env("DESCRIBE_RETRY_SLEEP_S", 1.5),
        jitter_s=get_float_env("RETRY_JITTER_S", 0.5),
    )
    t0 = time.perf_counter()

    def attempt_fn(attempt: int) -> _DescribeAttempt:
        t_attempt = time.perf_counter()
        try:
            max_tokens = base_max + (300 if attempt > 1 else 0)
            text, finish_reason = vision_describe_call(
                client=client,
                model=model,
                frames_jpeg=frames,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=structured_describe_enabled(),
            )
            return _DescribeAttempt(
                text=text,
                finish_reason=finish_reason,
                elapsed_s=time.perf_counter() - t_attempt,
            )
        except Exception as e:
            return _DescribeAttempt(
                text="",
                finish_reason=None,
                elapsed_s=time.perf_counter() - t_attempt,
                error=type(e).__name__,
            )

    def classify(attempt: int, result: _DescribeAttempt) -> str | None:
        if result.error:
            return result.error
        issue = _describe_output_issue(
            result.text,
            result.finish_reason,
            duration_s=duration_s,
        )
        if issue is None:
            if attempt > 1:
                log_human(
                    f"  {task_id} describe attempt {attempt} ok in {result.elapsed_s:.1f}s",
                )
            return None
        return issue

    def on_failure(attempt: int, reason: str, result: _DescribeAttempt) -> None:
        log_human(
            f"  {task_id} describe attempt {attempt}/{policy.max_attempts}: "
            f"{reason} in {result.elapsed_s:.1f}s",
        )

    def should_sleep(_attempt: int, reason: str) -> bool:
        return reason not in _describe_no_retry_reasons()

    def should_retry(_attempt: int, reason: str) -> bool:
        return reason not in _describe_no_retry_reasons()

    last, reasons = call_with_retry(
        policy=policy,
        attempt=attempt_fn,
        classify=classify,
        on_failure=on_failure,
        should_sleep=should_sleep,
        should_retry=should_retry,
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    attempts = len(reasons) + 1 if not reasons else len(reasons)

    if not reasons:
        text, _issue, raw_json = _finalize_describe_text(last.text)
        return DescribeResult(
            text=text,
            error=None,
            attempts=attempts,
            total_ms=total_ms,
            raw_json=raw_json,
        )

    last_reason = reasons[-1]
    if last.text and last_reason == "Truncated":
        text, _issue, raw_json = _finalize_describe_text(last.text)
        if text:
            return DescribeResult(
                text=text,
                error=None,
                attempts=attempts,
                total_ms=total_ms,
                raw_json=raw_json,
            )

    return DescribeResult(
        text=None,
        error=describe_error_from_reason(last_reason),
        error_detail=last_reason,
        attempts=attempts,
        total_ms=total_ms,
    )


def _describe_with_fallback(
    *,
    client: OpenAI | None,
    model: str,
    task_id: str,
    frames: list[bytes],
    skip_primary: bool = False,
    duration_s: float = 0.0,
) -> DescribeResult:
    fallback = _vision_fallback_model()
    deadline_skip = skip_primary
    long_skip = _should_skip_primary_for_duration(duration_s)
    skip_primary = deadline_skip or long_skip

    if skip_primary and fallback and fallback != model:
        guard = (
            "deadline guard active"
            if deadline_skip
            else f"long clip ({duration_s:.0f}s)"
        )
        log_human(f"  {task_id}: {guard}, describing with {fallback}")
        fallback_client = resolve_llm_client(fallback, fallback=client)
        result = _describe_frames(
            client=fallback_client,
            model=fallback,
            task_id=task_id,
            frames=frames,
            duration_s=duration_s,
        )
        if result.ok:
            return result
        # Fallback itself failed; fall through to the primary as a last resort.

    result = _describe_frames(
        client=client,
        model=model,
        task_id=task_id,
        frames=frames,
        duration_s=duration_s,
    )
    if result.ok:
        return result

    if not fallback or fallback == model or skip_primary:
        return result

    log_human(
        f"  {task_id}: primary describe failed "
        f"({result.error_detail or result.error}), trying fallback {fallback}",
    )
    fallback_client = resolve_llm_client(fallback, fallback=client)
    fallback_result = _describe_frames(
        client=fallback_client,
        model=fallback,
        task_id=task_id,
        frames=frames,
        duration_s=duration_s,
    )
    if fallback_result.ok:
        return fallback_result
    return result


def _task_duration_hint(task: dict) -> float:
    meta = task.get("meta") or {}
    try:
        return max(float(meta.get("duration_s") or 0.0), 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class _PreparedClip:
    task_id: str
    frames: list[bytes]
    frames_count: int
    video_duration_s: float
    download_s: float


def _prepare_clip_frames(
    *,
    task: dict,
    task_index: int,
    tasks: list[dict],
    ctx: _RunContext,
    prefetch: _ClipPrefetch,
) -> _PreparedClip:
    task_id = str(task.get("task_id", ""))
    video_url = str(task.get("video_url", ""))
    t_dl = time.perf_counter()
    td, video_path = prefetch.take(video_url)
    duration_hint = _task_duration_hint(task)
    try:
        log_human(f"  {task_id}: extracting frames...")
        frames, duration_s = extract_frames_jpeg(
            video_path,
            width=ctx.frame_width,
            duration_hint_s=duration_hint,
        )
        effective_duration = max(duration_s, duration_hint)
        log_human(
            f"  {task_id}: extracted {len(frames)} frames (duration={effective_duration:.1f}s)",
        )
    finally:
        td.cleanup()
    download_s = time.perf_counter() - t_dl

    prefetch.schedule_many(
        _next_prefetch_urls(
            tasks,
            task_index,
            descriptions_cache=ctx.descriptions_cache,
            dry_run=ctx.dry_run,
            depth=prefetch.depth,
        )
    )

    return _PreparedClip(
        task_id=task_id,
        frames=frames,
        frames_count=len(frames),
        video_duration_s=effective_duration,
        download_s=download_s,
    )


def _prepare_and_describe(
    *,
    task: dict,
    task_index: int,
    tasks: list[dict],
    ctx: _RunContext,
    prefetch: _ClipPrefetch,
    vision_client: OpenAI | None,
    vision_model: str,
) -> tuple[_PreparedClip, DescribeResult]:
    prep = _prepare_clip_frames(
        task=task,
        task_index=task_index,
        tasks=tasks,
        ctx=ctx,
        prefetch=prefetch,
    )
    log_human(f"  {prep.task_id}: describing...")
    t0 = time.perf_counter()
    describe = _describe_with_fallback(
        client=vision_client,
        model=vision_model,
        task_id=prep.task_id,
        frames=prep.frames,
        skip_primary=ctx.should_skip_primary_describe(
            remaining_clips=len(tasks) - task_index,
        ),
        duration_s=prep.video_duration_s,
    )
    log_human(f"  {prep.task_id}: describe done in {time.perf_counter() - t0:.1f}s")
    return prep, describe


def _caption_styles_from_description(
    *,
    client: OpenAI,
    model: str,
    describe: DescribeResult,
    requested_styles: list[str],
    parallel: bool,
    task_id: str = "tiebreak",
) -> tuple[dict[str, str], dict[str, int]]:
    captions: dict[str, str] = {}
    style_attempts: dict[str, int] = {}

    if not describe.ok:
        for style in requested_styles:
            captions[style] = public_describe_result(describe, style=style)
            style_attempts[style] = 0
        return captions, style_attempts

    def _one(style: str) -> tuple[str, str, int]:
        if style not in STYLES:
            return (
                style,
                public_process_failure(ProcessError.UNSUPPORTED_STYLE, style=style),
                0,
            )
        try:
            pool = resolve_caption_model_pool()
            if len(pool) > 1:
                candidate = generate_best_of_n_caption(
                    client=client,
                    models=pool,
                    style=style,
                    description=describe.text or "",
                    task_id=task_id,
                )
                return style, candidate.text, candidate.result.attempts
            caption_result = generate_styled_caption_from_text(
                client=client,
                model=model,
                style=style,
                description=describe.text or "",
            )
            return (
                style,
                public_caption_result(caption_result, style=style),
                caption_result.attempts,
            )
        except Exception as e:
            log_human(f"  caption {style} failed: {type(e).__name__}")
            return (
                style,
                public_process_failure(
                    ProcessError.PROCESSING,
                    style=style,
                    detail=type(e).__name__,
                ),
                0,
            )

    if parallel and len(requested_styles) > 1:
        with ThreadPoolExecutor(max_workers=min(len(requested_styles), 4)) as pool:
            futures = [pool.submit(_one, style) for style in requested_styles]
            for fut in as_completed(futures):
                style, caption, attempts = fut.result()
                captions[style] = caption
                style_attempts[style] = attempts
    else:
        for style in requested_styles:
            s, caption, attempts = _one(style)
            captions[s] = caption
            style_attempts[s] = attempts

    return captions, style_attempts


def _task_needs_video_download(
    task: dict,
    *,
    descriptions_cache: dict[str, str],
    dry_run: bool,
) -> str | None:
    if dry_run:
        return None
    task_id = str(task.get("task_id", ""))
    video_url = str(task.get("video_url", ""))
    if not task_id or not video_url:
        return None
    if descriptions_cache.get(task_id, ""):
        return None
    return video_url


def _next_prefetch_urls(
    tasks: list[dict],
    after_index: int,
    *,
    descriptions_cache: dict[str, str],
    dry_run: bool,
    depth: int,
) -> list[str]:
    urls: list[str] = []
    for j in range(after_index + 1, len(tasks)):
        url = _task_needs_video_download(
            tasks[j],
            descriptions_cache=descriptions_cache,
            dry_run=dry_run,
        )
        if url:
            urls.append(url)
            if len(urls) >= depth:
                break
    return urls


class _ClipPrefetch:
    """Download upcoming clips in background threads while the current one is processed.

    Downloads stream to temp files on disk, so memory stays flat regardless of
    clip size; only PREFETCH_DEPTH temp files exist at once.
    """

    def __init__(self, depth: int | None = None) -> None:
        self._depth = max(depth if depth is not None else get_int_env("PREFETCH_DEPTH", 2), 1)
        self._executor = ThreadPoolExecutor(
            max_workers=self._depth, thread_name_prefix="clip-prefetch"
        )
        self._pending: dict[
            str, tuple[tempfile.TemporaryDirectory[str], Path, Future[None]]
        ] = {}

    @property
    def depth(self) -> int:
        return self._depth

    def schedule(self, video_url: str) -> None:
        if video_url in self._pending or len(self._pending) >= self._depth:
            return
        td = tempfile.TemporaryDirectory()
        path = Path(td.name) / "clip.mp4"
        future = self._executor.submit(download_video, video_url, path)
        self._pending[video_url] = (td, path, future)

    def schedule_many(self, video_urls: list[str]) -> None:
        for url in video_urls:
            self.schedule(url)

    def take(self, video_url: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        entry = self._pending.pop(video_url, None)
        if entry is not None:
            td, path, future = entry
            try:
                future.result()
                return td, path
            except Exception:
                # Prefetch failed (transient network error); retry inline below.
                td.cleanup()

        td = tempfile.TemporaryDirectory()
        path = Path(td.name) / "clip.mp4"
        download_video(video_url, path)
        return td, path

    def _cancel_pending(self) -> None:
        for td, _path, future in self._pending.values():
            future.cancel()
            try:
                td.cleanup()
            except OSError:
                pass
        self._pending.clear()

    def close(self) -> None:
        self._cancel_pending()
        self._executor.shutdown(wait=False)

    def __enter__(self) -> _ClipPrefetch:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass
class _RunContext:
    dry_run: bool
    frame_width: int
    parallel_styles: bool
    descriptions_cache: dict[str, str]
    client: OpenAI | None
    caption_client: OpenAI | None
    vision_model: str | None
    caption_model: str | None
    descriptions_cache_label: str | None = None
    run_start: float = 0.0
    time_budget_s: float = 0.0
    deadline_reserve_s: float = 30.0
    descriptions_live: dict[str, str] = field(default_factory=dict)
    judge_retry: object | None = None

    def should_skip_primary_describe(self, *, remaining_clips: int) -> bool:
        """True when the remaining budget only covers fast fallback describes.

        Guards against slow-Gemma runs blowing the container time limit: once
        elapsed time leaves less than reserve x remaining clips, remaining
        describes go straight to the fallback model.
        """
        if self.time_budget_s <= 0 or not _vision_fallback_model():
            return False
        elapsed = time.monotonic() - self.run_start
        remaining_budget = self.time_budget_s - elapsed
        return remaining_budget < remaining_clips * self.deadline_reserve_s


def _build_run_context(
    *,
    client: OpenAI | None,
    caption_client: OpenAI | None = None,
    vision_model: str | None,
    caption_model: str | None,
) -> _RunContext:
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    _, frame_width = get_frame_config()
    parallel_styles = os.environ.get("PARALLEL_STYLES", "1") == "1"
    cache_path_raw = os.environ.get("DESCRIPTIONS_CACHE", "").strip()
    descriptions_cache: dict[str, str] = {}
    cache_label: str | None = None
    if cache_path_raw:
        cache_path = Path(cache_path_raw)
        if not cache_path.is_file():
            raise FileNotFoundError(f"DESCRIPTIONS_CACHE not found: {cache_path}")
        descriptions_cache = load_descriptions_cache(cache_path)
        cache_label = str(cache_path)
    return _RunContext(
        dry_run=dry_run,
        frame_width=frame_width,
        parallel_styles=parallel_styles,
        descriptions_cache=descriptions_cache,
        client=client,
        caption_client=caption_client,
        vision_model=vision_model,
        caption_model=caption_model,
        descriptions_cache_label=cache_label,
        run_start=time.monotonic(),
        time_budget_s=get_float_env("TIME_BUDGET_S", 0.0),
        deadline_reserve_s=get_float_env("DEADLINE_RESERVE_S", 30.0),
    )


def _log_config(ctx: _RunContext) -> None:
    if ctx.descriptions_cache_label:
        log_human(f"Using frozen descriptions from {ctx.descriptions_cache_label}")
    log_human(
        f"config: vision={ctx.vision_model} caption={ctx.caption_model} "
        f"parallel_styles={ctx.parallel_styles} frame_width={ctx.frame_width}px "
        f"frame_interval_s={os.environ.get('FRAME_INTERVAL_S', '4')} "
        f"frame_count_min={os.environ.get('FRAME_COUNT_MIN', '8')} "
        f"frame_count_max={os.environ.get('FRAME_COUNT_MAX', '24')} "
        f"api_timeout_s={os.environ.get('API_TIMEOUT_S', '45')} "
        f"vision_fallback={_vision_fallback_model() or 'none'} "
        f"time_budget_s={ctx.time_budget_s or 'off'} "
        f"judge_retry={'on' if get_int_env('JUDGE_RETRY', 0) else 'off'}",
    )
    emit_config_event(
        vision=ctx.vision_model,
        caption=ctx.caption_model,
        parallel_styles=ctx.parallel_styles,
        frame_width=ctx.frame_width,
        frame_interval_s=os.environ.get("FRAME_INTERVAL_S", "4"),
        frame_count_min=os.environ.get("FRAME_COUNT_MIN", "8"),
        frame_count_max=os.environ.get("FRAME_COUNT_MAX", "24"),
        api_timeout_s=os.environ.get("API_TIMEOUT_S", "45"),
        descriptions_cache=ctx.descriptions_cache_label,
        structured_describe=structured_describe_enabled(),
    )


def _emit_task_event(
    *,
    task_id: str,
    stage: str,
    download_s: float = 0.0,
    describe_s: float = 0.0,
    styles_s: float = 0.0,
    frames: int = 0,
    video_duration_s: float | None = None,
    describe_attempts: int = 0,
    style_attempts: dict[str, int] | None = None,
    describe_error: str | None = None,
    process_error: str | None = None,
) -> None:
    total_s = download_s + describe_s + styles_s
    event: dict[str, object] = {
        "stage": stage,
        "task_id": task_id,
        "download_s": round(download_s, 3),
        "describe_s": round(describe_s, 3),
        "styles_s": round(styles_s, 3),
        "total_s": round(total_s, 3),
        "frames": frames,
        "describe_attempts": describe_attempts,
    }
    if video_duration_s is not None:
        event["video_duration_s"] = round(video_duration_s, 3)
    if style_attempts is not None:
        event["style_attempts"] = style_attempts
    if describe_error is not None:
        event["describe_error"] = describe_error
    if process_error is not None:
        event["process_error"] = process_error
    emit_event(event)


def _normalize_requested_styles(styles: object) -> list[str]:
    if not isinstance(styles, list):
        return []
    return list(dict.fromkeys(s for s in styles if isinstance(s, str)))


def _append_and_persist(
    results: list[dict],
    result: dict,
    results_path: Path,
    *,
    ctx: _RunContext | None = None,
    clip_index: int | None = None,
) -> None:
    results.append(result)
    write_results(results_path, results)
    if ctx is not None and not ctx.descriptions_cache and ctx.descriptions_live:
        persist_live_descriptions(
            results_path,
            ctx.descriptions_live,
            update_fixture=True,
        )
    if ctx is not None and ctx.judge_retry is not None and clip_index is not None:
        ctx.judge_retry.submit(result, clip_index=clip_index)


def _finish_task_caption(
    *,
    task: dict,
    prepared: _PreparedClip,
    describe: DescribeResult,
    describe_s: float,
    ctx: _RunContext,
    caption_client: OpenAI,
) -> dict:
    task_id = str(task.get("task_id", ""))
    requested_styles = _normalize_requested_styles(task.get("styles", []))
    if not requested_styles:
        requested_styles = ["formal"]

    download_s = prepared.download_s
    caption_s = 0.0
    style_attempts: dict[str, int] = {}
    _store_live_description(ctx.descriptions_live, task_id, describe)
    try:
        t1 = time.perf_counter()
        log_human(f"  {task_id}: captioning styles...")
        captions, style_attempts = _caption_styles_from_description(
            client=caption_client,
            model=ctx.caption_model or "",
            describe=describe,
            requested_styles=requested_styles,
            parallel=ctx.parallel_styles,
            task_id=task_id,
        )
        caption_s = time.perf_counter() - t1
        log_human(
            f"  {task_id}: done in {download_s + describe_s + caption_s:.1f}s "
            f"(download={download_s:.1f}s, describe={describe_s:.1f}s, styles={caption_s:.1f}s)",
        )
        _emit_task_event(
            task_id=task_id,
            stage="complete",
            download_s=download_s,
            describe_s=describe_s,
            styles_s=caption_s,
            frames=prepared.frames_count,
            video_duration_s=prepared.video_duration_s,
            describe_attempts=describe.attempts,
            style_attempts=style_attempts,
            describe_error=describe.error.value if describe.error else None,
        )
    except Exception as e:
        log_human(f"  {task_id}: Failed to process video: {type(e).__name__}")
        captions = {
            style: public_process_failure(
                ProcessError.PROCESSING,
                style=style,
                detail=type(e).__name__,
            )
            for style in requested_styles
        }
        _emit_task_event(
            task_id=task_id,
            stage="error",
            download_s=download_s,
            describe_s=describe_s,
            styles_s=caption_s,
            frames=prepared.frames_count,
            video_duration_s=prepared.video_duration_s,
            describe_attempts=describe.attempts,
            style_attempts=style_attempts or None,
            describe_error=describe.error.value if describe.error else None,
            process_error=type(e).__name__,
        )
    return {"task_id": task_id, "captions": captions}


def _process_task_live(
    *,
    task: dict,
    task_index: int,
    tasks: list[dict],
    ctx: _RunContext,
    prefetch: _ClipPrefetch,
    vision_client: OpenAI | None,
    caption_client: OpenAI,
    prepared: _PreparedClip | None = None,
    describe: DescribeResult | None = None,
) -> dict:
    """Process one clip when describe may already be running/completed (overlap path)."""
    task_id = str(task.get("task_id", ""))
    video_url = str(task.get("video_url", ""))
    requested_styles = _normalize_requested_styles(task.get("styles", []))

    if not task_id or not video_url or not requested_styles:
        if not requested_styles:
            requested_styles = ["formal"]
        captions = {
            style: public_process_failure(ProcessError.INVALID_TASK, style=style)
            for style in requested_styles
        }
        _emit_task_event(task_id=task_id or "unknown", stage="invalid")
        return {"task_id": task_id or "unknown", "captions": captions}

    log_human(f"Processing {task_id} (describe + {len(requested_styles)} styles)...")

    describe_s = 0.0
    if describe is None or prepared is None:
        prep, describe = _prepare_and_describe(
            task=task,
            task_index=task_index,
            tasks=tasks,
            ctx=ctx,
            prefetch=prefetch,
            vision_client=vision_client,
            vision_model=ctx.vision_model or "",
        )
        prepared = prep
        describe_s = (describe.total_ms or 0.0) / 1000.0
    else:
        describe_s = (describe.total_ms or 0.0) / 1000.0

    return _finish_task_caption(
        task=task,
        prepared=prepared,
        describe=describe,
        describe_s=describe_s,
        ctx=ctx,
        caption_client=caption_client,
    )


def _process_task(
    *,
    task: dict,
    task_index: int,
    tasks: list[dict],
    ctx: _RunContext,
    prefetch: _ClipPrefetch,
) -> dict:
    task_id = str(task.get("task_id", ""))
    video_url = str(task.get("video_url", ""))
    requested_styles = _normalize_requested_styles(task.get("styles", []))

    if not task_id or not video_url or not requested_styles:
        if not requested_styles:
            requested_styles = ["formal"]
        captions = {
            style: public_process_failure(ProcessError.INVALID_TASK, style=style)
            for style in requested_styles
        }
        _emit_task_event(task_id=task_id or "unknown", stage="invalid")
        return {"task_id": task_id or "unknown", "captions": captions}

    if ctx.dry_run:
        _emit_task_event(task_id=task_id, stage="dry_run")
        return {"task_id": task_id, "captions": dry_run_captions(task_id, requested_styles)}

    if ctx.vision_model is None or ctx.caption_model is None:
        raise RuntimeError("vision_model/caption_model required when DRY_RUN=0")

    vision_client = resolve_llm_client(ctx.vision_model, fallback=ctx.client)
    caption_client = resolve_llm_client(
        ctx.caption_model,
        fallback=ctx.caption_client or ctx.client,
    )
    if caption_client is None:
        raise RuntimeError("caption model requires a Fireworks or OpenRouter client")

    log_human(f"Processing {task_id} (describe + {len(requested_styles)} styles)...")

    download_s = 0.0
    describe_s = 0.0
    caption_s = 0.0
    frames_count = 0
    video_duration_s: float | None = None
    describe = DescribeResult(text=None, error=None)
    style_attempts: dict[str, int] = {}

    try:
        cached_description = ctx.descriptions_cache.get(task_id, "")
        if cached_description:
            log_human(f"  {task_id}: using cached description")
            describe = DescribeResult(text=cached_description, error=None)
            prefetch.schedule_many(
                _next_prefetch_urls(
                    tasks,
                    task_index,
                    descriptions_cache=ctx.descriptions_cache,
                    dry_run=ctx.dry_run,
                    depth=prefetch.depth,
                )
            )
        else:
            t_dl = time.perf_counter()
            td, video_path = prefetch.take(video_url)
            duration_hint = _task_duration_hint(task)
            try:
                log_human(f"  {task_id}: extracting frames...")
                frames, duration_s = extract_frames_jpeg(
                    video_path,
                    width=ctx.frame_width,
                    duration_hint_s=duration_hint,
                )
                frames_count = len(frames)
                video_duration_s = max(duration_s, duration_hint)
                log_human(
                    f"  {task_id}: extracted {frames_count} frames "
                    f"(duration={video_duration_s:.1f}s)",
                )
            finally:
                td.cleanup()
            download_s = time.perf_counter() - t_dl

            prefetch.schedule_many(
                _next_prefetch_urls(
                    tasks,
                    task_index,
                    descriptions_cache=ctx.descriptions_cache,
                    dry_run=ctx.dry_run,
                    depth=prefetch.depth,
                )
            )

            t0 = time.perf_counter()
            log_human(f"  {task_id}: describing...")
            describe = _describe_with_fallback(
                client=vision_client,
                model=ctx.vision_model,
                task_id=task_id,
                frames=frames,
                skip_primary=ctx.should_skip_primary_describe(
                    remaining_clips=len(tasks) - task_index,
                ),
                duration_s=video_duration_s,
            )
            describe_s = time.perf_counter() - t0

        _store_live_description(ctx.descriptions_live, task_id, describe)
        t1 = time.perf_counter()
        log_human(f"  {task_id}: captioning styles...")
        captions, style_attempts = _caption_styles_from_description(
            client=caption_client,
            model=ctx.caption_model,
            describe=describe,
            requested_styles=requested_styles,
            parallel=ctx.parallel_styles,
            task_id=task_id,
        )
        caption_s = time.perf_counter() - t1
        log_human(
            f"  {task_id}: done in {download_s + describe_s + caption_s:.1f}s "
            f"(download={download_s:.1f}s, describe={describe_s:.1f}s, styles={caption_s:.1f}s)",
        )
        _emit_task_event(
            task_id=task_id,
            stage="complete",
            download_s=download_s,
            describe_s=describe_s,
            styles_s=caption_s,
            frames=frames_count,
            video_duration_s=video_duration_s,
            describe_attempts=describe.attempts,
            style_attempts=style_attempts,
            describe_error=describe.error.value if describe.error else None,
        )
    except Exception as e:
        log_human(f"  {task_id}: Failed to process video: {type(e).__name__}")
        captions = {
            style: public_process_failure(
                ProcessError.PROCESSING,
                style=style,
                detail=type(e).__name__,
            )
            for style in requested_styles
        }
        _emit_task_event(
            task_id=task_id,
            stage="error",
            download_s=download_s,
            describe_s=describe_s,
            styles_s=caption_s,
            frames=frames_count,
            video_duration_s=video_duration_s,
            describe_attempts=describe.attempts,
            style_attempts=style_attempts or None,
            describe_error=describe.error.value if describe.error else None,
            process_error=type(e).__name__,
        )

    return {"task_id": task_id, "captions": captions}


def _next_live_task_index(
    tasks: list[dict],
    after_index: int,
    *,
    descriptions_cache: dict[str, str],
) -> int | None:
    for j in range(after_index + 1, len(tasks)):
        task_id = str(tasks[j].get("task_id", ""))
        if task_id and not descriptions_cache.get(task_id, ""):
            return j
    return None


def _run_tasks_with_overlap(
    *,
    tasks: list[dict],
    results: list[dict],
    results_path: Path,
    ctx: _RunContext,
    prefetch: _ClipPrefetch,
    vision_client: OpenAI | None,
    caption_client: OpenAI,
) -> None:
    describe_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="describe-overlap")
    pending_describe: Future[tuple[_PreparedClip, DescribeResult]] | None = None

    try:
        for task_index, task in enumerate(tasks):
            task_id = str(task.get("task_id", ""))
            cached_description = ctx.descriptions_cache.get(task_id, "")

            if cached_description:
                if pending_describe is not None:
                    pending_describe.result()
                    pending_describe = None
                result = _process_task(
                    task=task,
                    task_index=task_index,
                    tasks=tasks,
                    ctx=ctx,
                    prefetch=prefetch,
                )
                _append_and_persist(
                    results, result, results_path, ctx=ctx, clip_index=task_index
                )
                continue

            prepared: _PreparedClip | None = None
            describe: DescribeResult | None = None
            describe_s = 0.0
            if pending_describe is not None:
                prepared, describe = pending_describe.result()
                pending_describe = None
                describe_s = (describe.total_ms or 0.0) / 1000.0
            else:
                log_human(
                    f"Processing {task_id} (describe + "
                    f"{len(_normalize_requested_styles(task.get('styles', [])))} styles)...",
                )
                prepared, describe = _prepare_and_describe(
                    task=task,
                    task_index=task_index,
                    tasks=tasks,
                    ctx=ctx,
                    prefetch=prefetch,
                    vision_client=vision_client,
                    vision_model=ctx.vision_model or "",
                )
                describe_s = (describe.total_ms or 0.0) / 1000.0

            next_index = _next_live_task_index(
                tasks,
                task_index,
                descriptions_cache=ctx.descriptions_cache,
            )
            if next_index is not None:
                next_task = tasks[next_index]
                pending_describe = describe_pool.submit(
                    _prepare_and_describe,
                    task=next_task,
                    task_index=next_index,
                    tasks=tasks,
                    ctx=ctx,
                    prefetch=prefetch,
                    vision_client=vision_client,
                    vision_model=ctx.vision_model or "",
                )

            result = _finish_task_caption(
                task=task,
                prepared=prepared,
                describe=describe,
                describe_s=describe_s,
                ctx=ctx,
                caption_client=caption_client,
            )
            _append_and_persist(
                results, result, results_path, ctx=ctx, clip_index=task_index
            )
    finally:
        describe_pool.shutdown(wait=True)


def run_full_tasks(
    *,
    tasks_path: Path,
    results_path: Path,
    client: OpenAI | None,
    vision_model: str | None,
    caption_model: str | None,
    caption_client: OpenAI | None = None,
) -> None:
    tasks = read_tasks(tasks_path)
    ctx = _build_run_context(
        client=client,
        caption_client=caption_client,
        vision_model=vision_model,
        caption_model=caption_model,
    )
    _log_config(ctx)

    vision_client = resolve_llm_client(vision_model, fallback=client)
    caption_client = resolve_llm_client(
        caption_model,
        fallback=caption_client or client,
    )
    if caption_client is None:
        raise RuntimeError("caption model requires a Fireworks or OpenRouter client")

    from src.judge_retry import create_pipelined_judge_retry

    overlap = os.environ.get("OVERLAP_PIPELINE", "1") == "1"
    results: list[dict] = []
    ctx.judge_retry = create_pipelined_judge_retry(
        results=results,
        results_path=results_path,
        descriptions=ctx.descriptions_live,
        caption_client=caption_client,
        caption_model=ctx.caption_model or "",
        run_start=ctx.run_start,
        time_budget_s=ctx.time_budget_s,
        total_clips=len(tasks),
    )
    with _ClipPrefetch() as prefetch:
        try:
            if overlap and not ctx.descriptions_cache:
                _run_tasks_with_overlap(
                    tasks=tasks,
                    results=results,
                    results_path=results_path,
                    ctx=ctx,
                    prefetch=prefetch,
                    vision_client=vision_client,
                    caption_client=caption_client,
                )
            else:
                for task_index, task in enumerate(tasks):
                    result = _process_task(
                        task=task,
                        task_index=task_index,
                        tasks=tasks,
                        ctx=ctx,
                        prefetch=prefetch,
                    )
                    _append_and_persist(
                        results,
                        result,
                        results_path,
                        ctx=ctx,
                        clip_index=task_index,
                    )
        finally:
            if results:
                write_results(results_path, results)
            if not ctx.descriptions_cache and ctx.descriptions_live:
                persist_live_descriptions(
                    results_path,
                    ctx.descriptions_live,
                    update_fixture=True,
                )
                log_human(
                    f"wrote live descriptions -> {results_path.parent / 'descriptions_live.json'}"
                )
            if ctx.judge_retry is not None:
                ctx.judge_retry.finish()
