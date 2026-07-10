from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
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
    looks_truncated,
    vision_describe_call,
)
from src.env import get_float_env, get_frame_config, get_int_env, resolve_frame_count
from src.results import DescribeResult, ProcessError, describe_error_from_reason
from src.retry import RetryPolicy, call_with_retry
from src.run_log import emit_config_event, emit_event, log_human


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
) -> tuple[list[bytes], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    duration_s = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    if max_frames is None:
        max_frames = resolve_frame_count(duration_s)

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
        times_ms = _frame_times_ms(duration_s, max_frames)
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

    return frames, duration_s


@dataclass
class _DescribeAttempt:
    text: str
    finish_reason: str | None
    elapsed_s: float = 0.0
    error: str | None = None


def _describe_frames(
    *,
    client: OpenAI,
    model: str,
    task_id: str,
    frames: list[bytes],
) -> DescribeResult:
    base_max = max(get_int_env("DESCRIBE_MAX_TOKENS", 1200), 64)
    temperature = get_float_env("DESCRIBE_TEMPERATURE", 0.2)
    policy = RetryPolicy(
        max_attempts=max(get_int_env("DESCRIBE_MAX_ATTEMPTS", 2), 1),
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
        if result.text and not looks_truncated(result.text, result.finish_reason):
            if attempt > 1:
                log_human(
                    f"  {task_id} describe attempt {attempt} ok in {result.elapsed_s:.1f}s",
                )
            return None
        return "Truncated" if result.text else "EmptyResponse"

    def on_failure(attempt: int, reason: str, result: _DescribeAttempt) -> None:
        log_human(
            f"  {task_id} describe attempt {attempt}/{policy.max_attempts}: "
            f"{reason} in {result.elapsed_s:.1f}s",
        )

    last, reasons = call_with_retry(
        policy=policy,
        attempt=attempt_fn,
        classify=classify,
        on_failure=on_failure,
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    attempts = len(reasons) + 1 if not reasons else len(reasons)

    if not reasons:
        return DescribeResult(
            text=last.text,
            error=None,
            attempts=attempts,
            total_ms=total_ms,
        )

    last_reason = reasons[-1]
    if last.text and last_reason == "Truncated":
        return DescribeResult(
            text=last.text,
            error=None,
            attempts=attempts,
            total_ms=total_ms,
        )

    return DescribeResult(
        text=None,
        error=describe_error_from_reason(last_reason),
        error_detail=last_reason,
        attempts=attempts,
        total_ms=total_ms,
    )


def _caption_styles_from_description(
    *,
    client: OpenAI,
    model: str,
    describe: DescribeResult,
    requested_styles: list[str],
    parallel: bool,
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


def _next_prefetch_url(
    tasks: list[dict],
    after_index: int,
    *,
    descriptions_cache: dict[str, str],
    dry_run: bool,
) -> str | None:
    for j in range(after_index + 1, len(tasks)):
        url = _task_needs_video_download(
            tasks[j],
            descriptions_cache=descriptions_cache,
            dry_run=dry_run,
        )
        if url:
            return url
    return None


class _ClipPrefetch:
    """Download the next clip in a background thread while the current one is processed."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip-prefetch")
        self._future: Future[None] | None = None
        self._td: tempfile.TemporaryDirectory[str] | None = None
        self._path: Path | None = None
        self._url: str | None = None

    def schedule(self, video_url: str) -> None:
        self._cancel_pending()
        self._td = tempfile.TemporaryDirectory()
        self._path = Path(self._td.name) / "clip.mp4"
        self._url = video_url
        path = self._path
        self._future = self._executor.submit(download_video, video_url, path)

    def take(self, video_url: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        if self._future is not None and self._url == video_url:
            self._future.result()
            self._future = None
            td, path = self._td, self._path
            self._td = None
            self._path = None
            self._url = None
            if td is None or path is None:
                raise RuntimeError("prefetch state missing after completed download")
            return td, path

        td = tempfile.TemporaryDirectory()
        path = Path(td.name) / "clip.mp4"
        download_video(video_url, path)
        return td, path

    def _cancel_pending(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        if self._td is not None:
            self._td.cleanup()
            self._td = None
        self._path = None
        self._url = None

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
    vision_model: str | None
    caption_model: str | None
    descriptions_cache_label: str | None = None


def _build_run_context(
    *,
    client: OpenAI | None,
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
        vision_model=vision_model,
        caption_model=caption_model,
        descriptions_cache_label=cache_label,
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
        f"api_timeout_s={os.environ.get('API_TIMEOUT_S', '45')}",
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


def _append_and_persist(results: list[dict], result: dict, results_path: Path) -> None:
    results.append(result)
    write_results(results_path, results)


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

    if ctx.client is None or ctx.vision_model is None or ctx.caption_model is None:
        raise RuntimeError("client/vision_model/caption_model required when DRY_RUN=0")

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
            next_url = _next_prefetch_url(
                tasks,
                task_index,
                descriptions_cache=ctx.descriptions_cache,
                dry_run=ctx.dry_run,
            )
            if next_url:
                prefetch.schedule(next_url)
        else:
            t_dl = time.perf_counter()
            td, video_path = prefetch.take(video_url)
            try:
                log_human(f"  {task_id}: extracting frames...")
                frames, duration_s = extract_frames_jpeg(video_path, width=ctx.frame_width)
                frames_count = len(frames)
                video_duration_s = duration_s
                log_human(
                    f"  {task_id}: extracted {frames_count} frames "
                    f"(duration={duration_s:.1f}s)",
                )
            finally:
                td.cleanup()
            download_s = time.perf_counter() - t_dl

            next_url = _next_prefetch_url(
                tasks,
                task_index,
                descriptions_cache=ctx.descriptions_cache,
                dry_run=ctx.dry_run,
            )
            if next_url:
                prefetch.schedule(next_url)

            t0 = time.perf_counter()
            log_human(f"  {task_id}: describing...")
            describe = _describe_frames(
                client=ctx.client,
                model=ctx.vision_model,
                task_id=task_id,
                frames=frames,
            )
            describe_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        log_human(f"  {task_id}: captioning styles...")
        captions, style_attempts = _caption_styles_from_description(
            client=ctx.client,
            model=ctx.caption_model,
            describe=describe,
            requested_styles=requested_styles,
            parallel=ctx.parallel_styles,
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


def run_full_tasks(
    *,
    tasks_path: Path,
    results_path: Path,
    client: OpenAI | None,
    vision_model: str | None,
    caption_model: str | None,
) -> None:
    tasks = read_tasks(tasks_path)
    ctx = _build_run_context(
        client=client,
        vision_model=vision_model,
        caption_model=caption_model,
    )
    _log_config(ctx)

    results: list[dict] = []
    with _ClipPrefetch() as prefetch:
        try:
            for task_index, task in enumerate(tasks):
                result = _process_task(
                    task=task,
                    task_index=task_index,
                    tasks=tasks,
                    ctx=ctx,
                    prefetch=prefetch,
                )
                _append_and_persist(results, result, results_path)
        finally:
            if results:
                write_results(results_path, results)
