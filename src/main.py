import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.caption import get_fireworks_client
from src.pipeline import run_tasks


def main() -> None:
    load_dotenv()

    tasks_path = Path(os.environ.get("INPUT_TASKS", "/input/tasks.json"))
    results_path = Path(os.environ.get("OUTPUT_RESULTS", "/output/results.json"))

    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    if not tasks_path.exists():
        print(f"Missing tasks file: {tasks_path}", file=sys.stderr)
        sys.exit(2)

    if dry_run:
        run_tasks(tasks_path=tasks_path, results_path=results_path, client=None, model=None)
        sys.exit(0)

    model = os.environ.get("FIREWORKS_MODEL", "")
    if not model:
        print(
            "Missing FIREWORKS_MODEL (e.g. accounts/fireworks/models/minimax-m3)",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        client = get_fireworks_client()
        run_tasks(tasks_path=tasks_path, results_path=results_path, client=client, model=model)
    except Exception as e:
        print(f"Fatal error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()