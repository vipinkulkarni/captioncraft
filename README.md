# CaptionCraft

Video captioning agent for AMD Developer Hackathon (Track 2).

Given a video URL and a list of styles, CaptionCraft produces one caption per style (`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`) and writes `results.json` in the Track-2 contract format.

## Architecture (high level)

This repo uses a **describe → rewrite** pipeline:

1. Download video and sample frames (dynamic frame count based on duration)
2. Run a single **vision “describe”** call to produce a factual description
3. Rewrite that description into 4 styles using a text model (optionally in parallel)

This keeps the expensive vision step to **one call per clip**, then fans out into cheap caption rewrites.

## Local quickstart (CLI)

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt

$env:FIREWORKS_API_KEY="..."
$env:VISION_MODEL="accounts/fireworks/models/minimax-m3"
$env:CAPTION_MODEL="accounts/fireworks/models/deepseek-v4-flash"
$env:PARALLEL_STYLES="1"

$env:INPUT_TASKS="misc\eval\tasks_amd_example_vids.json"
$env:OUTPUT_RESULTS="misc\runs\demo\results.json"
.\.venv\Scripts\python.exe -m src.main
```

## Demo app (Streamlit)

The repo includes a minimal Streamlit UI (`streamlit_app.py`) that runs the **same pipeline** (`src.pipeline.run_full_tasks`) as the Docker agent, but for a single pasted video URL.

### Run locally

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt

$env:FIREWORKS_API_KEY="..."
$env:VISION_MODEL="accounts/fireworks/models/minimax-m3"
$env:CAPTION_MODEL="accounts/fireworks/models/deepseek-v4-flash"
$env:PARALLEL_STYLES="1"

.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The demo supports either:
- **URL input** (paste a video URL)

### Deploy on Streamlit Community Cloud

1. Push the repo to GitHub
2. Create a new Streamlit app
3. **Main file path**: `streamlit_app.py`
4. Set secrets (App → Settings → Secrets):

```toml
FIREWORKS_API_KEY = "..."
GOOGLE_API_KEY = "..."
VISION_MODEL = "google-ai/gemma-4-26b-a4b-it"
VISION_FALLBACK_MODEL = "accounts/fireworks/models/minimax-m3"
CAPTION_MODEL = "accounts/fireworks/models/deepseek-v4-flash"
PARALLEL_STYLES = "1"
OVERLAP_PIPELINE = "1"
PREFETCH_DEPTH = "2"
```

The published **Docker image** uses the same Gemma 4 + M3 fallback + DeepSeek stack. Both `FIREWORKS_API_KEY` and `GOOGLE_API_KEY` are baked at build time (GitHub Actions secrets).

Streamlit will give you a public **demo URL** (use it for the submission “Demo Application URL” field).

## Environment variables (most important)

- **`FIREWORKS_API_KEY`**: required
- **`VISION_MODEL`**: required (vision describe step)
- **`CAPTION_MODEL`**: required (style caption step)
- **`PARALLEL_STYLES`**: `1` recommended (parallelize the 4 style calls)
- **Frame sampling**: `FRAME_INTERVAL_S`, `FRAME_COUNT_MIN`, `FRAME_COUNT_MAX`, optional `FRAME_COUNT` override

