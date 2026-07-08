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
    dry_run_captions,
    generate_factual_description,
    generate_styled_caption_from_text,
)
from src.env import get_frame_config


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


def download_video(url: str, dest: Path) -> None:
    read_timeout = float(os.environ.get("DOWNLOAD_READ_TIMEOUT", "180"))
    timeout = httpx.Timeout(connect=15.0, read=read_timeout, write=15.0, pool=15.0)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        content_type = (r.headers.get("content-type") or "").lower()
        if content_type and not content_type.startswith("video/"):
            raise RuntimeError(f"Non-video response content-type={content_type}")
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)


def extract_frames_jpeg(video_path: Path, *, max_frames: int = 8, width: int = 512) -> list[bytes]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0 or max_frames <= 1:
        indices = list(range(max_frames))
    else:
        last = max(frame_count - 1, 0)
        indices = [round(i * last / (max_frames - 1)) for i in range(max_frames)]

    frames: list[bytes] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        if w > width:
            new_h = max(int(h * (width / w)), 1)
            frame = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            frames.append(buf.tobytes())

    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted from video")

    return frames


def _describe_frames(
    *,
    client: OpenAI,
    model: str,
    task_id: str,
    frames: list[bytes],
) -> str:
    description = ""
    last_error = ""
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            description = generate_factual_description(
                client=client,
                model=model,
                frames_jpeg=frames,
            )
            if description:
                return description
            last_error = "EmptyResponse"
            print(f"  {task_id} describe attempt {attempt}: empty response", file=sys.stderr)
        except Exception as e:
            last_error = type(e).__name__
            print(f"  {task_id} describe attempt {attempt} failed: {last_error}", file=sys.stderr)
        if attempt < max_attempts:
            time.sleep(2)

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

    if description.startswith("Failed to describe video:"):
        return {style: description for style in requested_styles}

    def _one(style: str) -> tuple[str, str]:
        if style not in STYLES:
            return style, "Unsupported style requested."
        try:
            caption = generate_styled_caption_from_text(
                client=client,
                model=model,
                style=style,
                description=description,
            )
            if not caption.strip():
                caption = "Failed to caption: EmptyResponse"
            return style, caption
        except Exception as e:
            return style, f"Failed to caption: {type(e).__name__}"

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
    frame_count, frame_width = get_frame_config()
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
                captions[style] = "Invalid task input."
            results.append({"task_id": task_id or "unknown", "captions": captions})
            continue

        if dry_run:
            captions = dry_run_captions(task_id, requested_styles)
            results.append({"task_id": task_id, "captions": captions})
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
                    frames = extract_frames_jpeg(video_path, max_frames=frame_count, width=frame_width)
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
            captions = {style: err for style in requested_styles}

        results.append({"task_id": task_id, "captions": captions})

    write_results(results_path, results)
