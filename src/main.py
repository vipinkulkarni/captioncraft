import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.caption import get_fireworks_client, resolve_llm_client
from src.pipeline import run_full_tasks


def _resolve_vision_model() -> str:
    return (
        os.environ.get("VISION_MODEL", "").strip()
        or os.environ.get("FIREWORKS_MODEL", "").strip()
    )


def _resolve_caption_model() -> str:
    return (
        os.environ.get("CAPTION_MODEL", "").strip()
        or os.environ.get("FIREWORKS_MODEL", "").strip()
        or _resolve_vision_model()
    )


def main() -> None:
    load_dotenv()

    tasks_path = Path(os.environ.get("INPUT_TASKS", "/input/tasks.json"))
    results_path = Path(os.environ.get("OUTPUT_RESULTS", "/output/results.json"))
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    if not tasks_path.exists():
        print(f"Missing tasks file: {tasks_path}", file=sys.stderr)
        sys.exit(2)

    if dry_run:
        run_full_tasks(
            tasks_path=tasks_path,
            results_path=results_path,
            client=None,
            vision_model=None,
            caption_model=None,
        )
        sys.exit(0)

    vision_model = _resolve_vision_model()
    caption_model = _resolve_caption_model()
    if not vision_model:
        print(
            "Missing VISION_MODEL or FIREWORKS_MODEL (e.g. accounts/fireworks/models/minimax-m3)",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        vision_client = resolve_llm_client(vision_model)
        caption_client = resolve_llm_client(caption_model)
        if caption_client is None:
            raise RuntimeError(
                "CAPTION_MODEL requires Fireworks or OpenRouter (not Google AI vision)"
            )
        run_full_tasks(
            tasks_path=tasks_path,
            results_path=results_path,
            client=vision_client or caption_client,
            caption_client=caption_client,
            vision_model=vision_model,
            caption_model=caption_model,
        )
    except Exception as e:
        print(f"Fatal error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
