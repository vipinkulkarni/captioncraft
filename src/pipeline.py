import json
import os
import tempfile
from pathlib import Path

import cv2
import httpx
from openai import OpenAI

from src.caption import STYLES, dry_run_captions, generate_caption_from_frames


def read_tasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tasks.json must be a JSON array")
    return data


def write_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def download_video(url: str, dest: Path) -> None:
    timeout = httpx.Timeout(connect=15.0, read=60.0, write=15.0, pool=15.0)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
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


def run_tasks(
    *,
    tasks_path: Path,
    results_path: Path,
    client: OpenAI | None,
    model: str | None,
) -> None:
    tasks = read_tasks(tasks_path)

    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    results: list[dict] = []

    for task in tasks:
        task_id = str(task.get("task_id", ""))
        video_url = str(task.get("video_url", ""))
        styles = task.get("styles", [])

        requested_styles = [s for s in styles if isinstance(s, str)]
        requested_styles = list(dict.fromkeys(requested_styles))  # de-dupe, stable order

        captions: dict[str, str] = {}

        if not task_id or not video_url or not requested_styles:
            # Best-effort: still emit something valid; track scorer will penalize bad inputs anyway.
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

        if client is None or model is None:
            raise RuntimeError("client/model required when DRY_RUN=0")

        frames: list[bytes] | None = None
        try:
            with tempfile.TemporaryDirectory() as td:
                video_path = Path(td) / "clip.mp4"
                download_video(video_url, video_path)
                frames = extract_frames_jpeg(video_path, max_frames=8, width=512)
        except Exception as e:
            for style in requested_styles:
                captions[style] = f"Failed to process video: {type(e).__name__}"
            results.append({"task_id": task_id, "captions": captions})
            continue

        assert frames is not None

        for style in requested_styles:
            if style not in STYLES:
                captions[style] = "Unsupported style requested."
                continue
            try:
                captions[style] = generate_caption_from_frames(
                    client=client,
                    model=model,
                    style=style,
                    frames_jpeg=frames,
                )
            except Exception as e:
                captions[style] = f"Failed to caption: {type(e).__name__}"

        results.append({"task_id": task_id, "captions": captions})

    write_results(results_path, results)