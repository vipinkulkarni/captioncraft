import base64
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")


def load_prompt(style: str) -> str:
    path = Path(__file__).resolve().parent.parent / "prompts" / f"{style}.txt"
    return path.read_text(encoding="utf-8")


def _to_data_url(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def generate_caption_from_frames(
    *,
    client: OpenAI,
    model: str,
    style: str,
    frames_jpeg: Iterable[bytes],
) -> str:
    prompt_template = load_prompt(style)

    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": _to_data_url(b)}} for b in frames_jpeg
    ]
    content.append(
        {
            "type": "text",
            "text": prompt_template.format(
                content=(
                    "These are sampled frames from a short video clip. "
                    "Write the requested caption based only on what is visible."
                )
            ),
        }
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=220,
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


def dry_run_captions(task_id: str, styles: list[str]) -> dict[str, str]:
    return {s: f"[DRY_RUN] {task_id} - {s}" for s in styles}


def get_fireworks_client() -> OpenAI:
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing FIREWORKS_API_KEY")

    return OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
    )