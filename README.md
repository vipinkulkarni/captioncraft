# CaptionCraft

Video captioning agent for AMD Developer Hackathon (Track 2).

Given a video URL and a list of styles, CaptionCraft produces one caption per style (`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`) and writes `results.json` in the Track-2 contract format.

## Architecture (high level)

This repo uses a **describe → rewrite** pipeline:

1. Download video and sample frames (`FRAME_MODE=scene` midpoints by default)
2. Run a single **vision “describe”** call (Kimi K2.6) to produce a factual description
3. Rewrite that description into 4 styles using DeepSeek (optionally in parallel)

This keeps the expensive vision step to **one call per clip**, then fans out into cheap caption rewrites.

## Local quickstart (CLI)

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt

$env:FIREWORKS_API_KEY="..."
$env:VISION_MODEL="accounts/fireworks/models/kimi-k2p6"
$env:CAPTION_MODEL="accounts/fireworks/models/deepseek-v4-flash"
$env:DESCRIBE_DUAL="0"
$env:FRAME_MODE="scene"
$env:FRAME_WIDTH="512"
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
$env:VISION_MODEL="accounts/fireworks/models/kimi-k2p6"
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
VISION_MODEL = "accounts/fireworks/models/kimi-k2p6"
VISION_FALLBACK_MODEL = "accounts/fireworks/models/minimax-m3"
CAPTION_MODEL = "accounts/fireworks/models/deepseek-v4-flash"
DESCRIBE_DUAL = "0"
FRAME_MODE = "scene"
FRAME_WIDTH = "384"
FRAME_COUNT_MAX = "12"
API_TIMEOUT_S = "60"
PARALLEL_STYLES = "1"
OVERLAP_PIPELINE = "1"
PREFETCH_DEPTH = "2"
```

The published **Docker image** uses **Kimi K2.6** describe (scene frames) + DeepSeek captions, with MiniMax as describe fallback. `FIREWORKS_API_KEY` is baked at build time (GitHub Actions secret). `GOOGLE_API_KEY` is optional (only if you switch vision back to Gemma).

Streamlit will give you a public **demo URL** (use it for the submission “Demo Application URL” field).

## Environment variables (most important)

- **`FIREWORKS_API_KEY`**: required (Kimi, DeepSeek, MiniMax fallback)
- **`VISION_MODEL`**: required (default: `accounts/fireworks/models/kimi-k2p6`)
- **`CAPTION_MODEL`**: required (style caption step)
- **`DESCRIBE_DUAL`**: `0` for single-VLM Kimi (submission default)
- **`FRAME_MODE`**: `scene` recommended; `uniform` for even time samples
- **`PARALLEL_STYLES`**: `1` recommended (parallelize the 4 style calls)
- **Frame sampling**: `FRAME_WIDTH`, `FRAME_INTERVAL_S`, `FRAME_COUNT_MIN`, `FRAME_COUNT_MAX`, optional `FRAME_COUNT` override

