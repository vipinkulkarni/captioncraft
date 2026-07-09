import json
import os
import sys
import tempfile
import time
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import httpx
from openai import OpenAI

from src.caption import (
    STYLES,
    _looks_truncated,
    _vision_describe_call,
    dry_run_captions,
    generate_styled_caption_from_text,
    is_describe_failure,
    public_caption,
)
from src.env import get_float_env, get_frame_config, get_int_env, resolve_frame_count


def probe_video_duration_s(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    if frame_count > 0 and fps > 0:
        return frame_count / fps
    return 0.0


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
    max_frames: int = 8,
    width: int = 512,
) -> tuple[list[bytes], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    duration_s = frame_count / fps if frame_count > 0 and fps > 0 else 0.0

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


def _describe_frames(
    *,
    client: OpenAI,
    model: str,
    task_id: str,
    frames: list[bytes],
) -> str:
    base_max = max(get_int_env("DESCRIBE_MAX_TOKENS", 1000), 64)
    temperature = get_float_env("DESCRIBE_TEMPERATURE", 0.2)
    max_attempts = max(get_int_env("DESCRIBE_MAX_ATTEMPTS", 2), 1)
    retry_sleep_s = get_float_env("DESCRIBE_RETRY_SLEEP_S", 1.5)

    last_error = ""
    last_text = ""

    for attempt in range(1, max_attempts + 1):
        max_tokens = base_max + (300 if attempt > 1 else 0)
        t0 = time.perf_counter()
        try:
            text, finish_reason = _vision_describe_call(
                client=client,
                model=model,
                frames_jpeg=frames,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.perf_counter() - t0
            if text and not _looks_truncated(text, finish_reason):
                if attempt > 1:
                    print(
                        f"  {task_id} describe attempt {attempt} ok in {elapsed:.1f}s",
                        file=sys.stderr,
                    )
                return text
            last_text = text
            last_error = "Truncated" if text else "EmptyResponse"
            print(
                f"  {task_id} describe attempt {attempt}/{max_attempts}: "
                f"{last_error} in {elapsed:.1f}s",
                file=sys.stderr,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            last_error = type(e).__name__
            print(
                f"  {task_id} describe attempt {attempt}/{max_attempts} failed: "
                f"{last_error} in {elapsed:.1f}s",
                file=sys.stderr,
            )
        if attempt < max_attempts:
            time.sleep(retry_sleep_s)

    if last_text and last_error == "Truncated":
        return last_text

    return f"Failed to describe video: {last_error or 'EmptyResponse'}"


def _caption_styles_from_description(
    *,
    client: OpenAI,
    model: str,
    description: str,
    requested_styles: list[str],
    parallel: bool,
) -> dict[str, str]:
    captions: dict[str, str] = {}

    if is_describe_failure(description):
        return {
            style: public_caption(description) for style in requested_styles
        }

    def _one(style: str) -> tuple[str, str]:
        if style not in STYLES:
            return style, public_caption("Unsupported style requested.")
        try:
            caption = generate_styled_caption_from_text(
                client=client,
                model=model,
                style=style,
                description=description,
            )
            if not caption.strip():
                caption = "Failed to caption: EmptyResponse"
            return style, public_caption(caption)
        except Exception as e:
            print(f"  caption {style} failed: {type(e).__name__}", file=sys.stderr)
            return style, public_caption(f"Failed to caption: {type(e).__name__}")

    if parallel and len(requested_styles) > 1:
        with ThreadPoolExecutor(max_workers=min(len(requested_styles), 4)) as pool:
            futures = [pool.submit(_one, style) for style in requested_styles]
            for fut in as_completed(futures):
                style, caption = fut.result()
                captions[style] = caption
    else:
        for style in requested_styles:
            s, caption = _one(style)
            captions[s] = caption

    return captions


def run_full_tasks(
    *,
    tasks_path: Path,
    results_path: Path,
    client: OpenAI | None,
    vision_model: str | None,
    caption_model: str | None,
) -> None:
    tasks = read_tasks(tasks_path)

    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    _, frame_width = get_frame_config()
    parallel_styles = os.environ.get("PARALLEL_STYLES", "1") == "1"
    cache_path_raw = os.environ.get("DESCRIPTIONS_CACHE", "").strip()
    descriptions_cache: dict[str, str] = {}
    if cache_path_raw:
        cache_path = Path(cache_path_raw)
        if not cache_path.is_file():
            raise FileNotFoundError(f"DESCRIPTIONS_CACHE not found: {cache_path}")
        descriptions_cache = load_descriptions_cache(cache_path)
        print(f"Using frozen descriptions from {cache_path}", file=sys.stderr)
    results: list[dict] = []

    try:
        for task in tasks:
            task_id = str(task.get("task_id", ""))
            video_url = str(task.get("video_url", ""))
            styles = task.get("styles", [])

            requested_styles = [s for s in styles if isinstance(s, str)]
            requested_styles = list(dict.fromkeys(requested_styles))

            captions: dict[str, str] = {}

            if not task_id or not video_url or not requested_styles:
                if not requested_styles:
                    requested_styles = ["formal"]
                for style in requested_styles:
                    captions[style] = public_caption("Invalid task input.")
                results.append({"task_id": task_id or "unknown", "captions": captions})
                write_results(results_path, results)
                continue

            if dry_run:
                captions = dry_run_captions(task_id, requested_styles)
                results.append({"task_id": task_id, "captions": captions})
                write_results(results_path, results)
                continue

            if client is None or vision_model is None or caption_model is None:
                raise RuntimeError("client/vision_model/caption_model required when DRY_RUN=0")

            print(f"Processing {task_id} (describe + {len(requested_styles)} styles)...", file=sys.stderr)

            try:
                cached_description = descriptions_cache.get(task_id, "")
                if cached_description:
                    print(f"  {task_id}: using cached description", file=sys.stderr)
                    description = cached_description
                    download_s = 0.0
                    describe_s = 0.0
                else:
                    t_dl = time.perf_counter()
                    with tempfile.TemporaryDirectory() as td:
                        video_path = Path(td) / "clip.mp4"
                        download_video(video_url, video_path)
                        frame_count = resolve_frame_count(probe_video_duration_s(video_path))
                        print(
                            f"  {task_id}: extracting {frame_count} frames...",
                            file=sys.stderr,
                        )
                        frames, duration_s = extract_frames_jpeg(
                            video_path, max_frames=frame_count, width=frame_width
                        )
                        if duration_s <= 0:
                            duration_s = probe_video_duration_s(video_path)
                        print(
                            f"  {task_id}: extracted {len(frames)} frames "
                            f"(duration={duration_s:.1f}s)",
                            file=sys.stderr,
                        )
                    download_s = time.perf_counter() - t_dl

                    t0 = time.perf_counter()
                    print(f"  {task_id}: describing...", file=sys.stderr)
                    description = _describe_frames(
                        client=client,
                        model=vision_model,
                        task_id=task_id,
                        frames=frames,
                    )
                    describe_s = time.perf_counter() - t0

                t1 = time.perf_counter()
                print(f"  {task_id}: captioning styles...", file=sys.stderr)
                captions = _caption_styles_from_description(
                    client=client,
                    model=caption_model,
                    description=description,
                    requested_styles=requested_styles,
                    parallel=parallel_styles,
                )
                caption_s = time.perf_counter() - t1
                print(
                    f"  {task_id}: done in {download_s + describe_s + caption_s:.1f}s "
                    f"(download={download_s:.1f}s, describe={describe_s:.1f}s, styles={caption_s:.1f}s)",
                    file=sys.stderr,
                )
            except Exception as e:
                err = f"Failed to process video: {type(e).__name__}"
                print(f"  {task_id}: {err}", file=sys.stderr)
                captions = {
                    style: public_caption(err) for style in requested_styles
                }

            results.append({"task_id": task_id, "captions": captions})
            write_results(results_path, results)
    finally:
        if results:
            write_results(results_path, results)
