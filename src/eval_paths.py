"""Canonical paths for local eval workspace (misc/eval). Tracked in git."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO_ROOT / "misc" / "eval"
DATASETS_DIR = EVAL_ROOT / "datasets"
FIXTURES_DIR = EVAL_ROOT / "fixtures"
SCRIPTS_DIR = EVAL_ROOT / "scripts"
NOTES_DIR = EVAL_ROOT / "notes"
PROBES_DIR = EVAL_ROOT / "probes"
RUNS_ROOT = REPO_ROOT / "misc" / "runs"
RUNS_EVAL = RUNS_ROOT / "eval_dataset"
RUNS_ARCHIVE = RUNS_ROOT / "archive"
RUNS_DEMO = RUNS_ROOT / "demo"
DATA_OUTPUT = REPO_ROOT / "data" / "output"

TASKS_FULL = DATASETS_DIR / "tasks_eval_dataset.json"
TASKS_TRAIN = DATASETS_DIR / "tasks_train10.json"
TASKS_TEST = DATASETS_DIR / "tasks_test6.json"
TASKS_AMD3 = DATASETS_DIR / "tasks_amd_example_vids.json"
TASKS_PILOT = DATASETS_DIR / "tasks_eval_pilot_lengths.json"
TASKS_BLIND = DATASETS_DIR / "tasks_blind10.json"

DESCRIPTIONS_FULL = FIXTURES_DIR / "descriptions_eval16.json"
DESCRIPTIONS_TRAIN = FIXTURES_DIR / "descriptions_train10.json"
DESCRIPTIONS_TEST = FIXTURES_DIR / "descriptions_test6.json"
DESCRIPTIONS_AMD3 = FIXTURES_DIR / "descriptions_amd3.json"

SPLIT_MANIFEST = DATASETS_DIR / "split_manifest.json"
PROMPT_MANIFEST = NOTES_DIR / "prompt_manifest.json"
