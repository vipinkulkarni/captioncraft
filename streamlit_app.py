import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path when Streamlit changes the working directory.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dotenv import load_dotenv

# Shell / Streamlit secrets win over repo .env (matches src.main).
load_dotenv(_ROOT / ".env", override=False)

from src.caption import STYLES
from src.llm_clients import is_google_ai_model, is_openrouter_model, resolve_llm_client
from src.pipeline import run_full_tasks


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


_DEFAULT_VISION_MODEL = "accounts/fireworks/models/kimi-k2p6"
_DEFAULT_CAPTION_MODEL = "accounts/fireworks/models/deepseek-v4-flash"
_DEFAULT_VISION_FALLBACK = "accounts/fireworks/models/minimax-m3"


def _apply_demo_env() -> None:
    """Align with Docker agent caption stack; skip batch-only judge for single-clip demo."""
    os.environ.setdefault("JUDGE_RETRY", "0")
    os.environ.setdefault("STYLE_JSON_MODE", "1")
    os.environ.setdefault("CAPTION_MODEL_POOL", "deepseek-v4-flash,deepseek-v4-flash")
    os.environ.setdefault("PARALLEL_STYLES", "1")
    os.environ.setdefault("STYLE_META_LEAK_SALVAGE", "1")
    os.environ.setdefault("GOOGLE_API_TIMEOUT_S", "30")
    os.environ.setdefault("DESCRIBE_MAX_ATTEMPTS_WITH_FALLBACK", "1")
    os.environ.setdefault("DESCRIBE_DUAL", "0")
    os.environ.setdefault("FRAME_MODE", "scene")
    os.environ.setdefault("FRAME_WIDTH", "512")
    os.environ.setdefault("API_TIMEOUT_S", "120")
    os.environ.setdefault("VISION_FALLBACK_MODEL", _DEFAULT_VISION_FALLBACK)


def _default_vision_model() -> str:
    return _env("VISION_MODEL") or _DEFAULT_VISION_MODEL


def _default_caption_model() -> str:
    return _env("CAPTION_MODEL") or _DEFAULT_CAPTION_MODEL


def _default_vision_fallback() -> str:
    return _env("VISION_FALLBACK_MODEL") or _DEFAULT_VISION_FALLBACK


def _pretty_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return ""
    lower = m.lower()
    if "minimax-m3" in lower:
        return "MiniMax M3"
    if "qwen3p7" in lower or "qwen3.7" in lower:
        return "Qwen3.7 Plus"
    if "gemma-4" in lower or "gemma_4" in lower:
        return "Gemma 4 (vision)"
    if "gemma" in lower:
        return "Gemma"
    if "deepseek-v4-flash" in lower:
        return "DeepSeek V4 Flash"
    return m.rsplit("/", 1)[-1]


_DEMO_VIDEO_URL = (
    "https://storage.googleapis.com/amd-hackathon-clips/"
    "1860079-uhd_2560_1440_25fps.mp4"
)


st.set_page_config(page_title="CaptionCraft Demo", page_icon="🎬", layout="centered")

st.markdown(
    """
<style>
  .cc-card {
    padding: 1rem 1rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    margin-bottom: 0.75rem;
  }
  .cc-label {
    font-size: 0.85rem;
    opacity: 0.8;
    margin-bottom: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
</style>
""",
    unsafe_allow_html=True,
)

st.title("CaptionCraft demo")
st.caption(
    "Gemma 4 describe → four DeepSeek rewrites (best-of-2 · JSON mode). "
    "Same caption stack as the Docker agent; batch runs add pipelined judge+retry."
)

with st.sidebar:
    st.subheader("Config")
    vision_model = _default_vision_model()
    caption_model = _default_caption_model()
    fallback_model = _default_vision_fallback()
    st.text_input("Vision model", value=_pretty_model(vision_model), disabled=True)
    st.text_input(
        "Vision fallback",
        value=_pretty_model(fallback_model),
        disabled=True,
    )
    st.text_input("Caption model", value=_pretty_model(caption_model), disabled=True)

    key_len = len(_env("FIREWORKS_API_KEY"))
    or_key_len = len(_env("OPENROUTER_API_KEY"))
    google_key_len = len(_env("GOOGLE_API_KEY")) or len(_env("GEMINI_API_KEY"))
    st.write(f"Fireworks API key: {'set' if key_len else 'missing'}")
    st.write(f"Google AI API key: {'set' if google_key_len else 'missing'}")
    st.write(f"OpenRouter API key: {'set' if or_key_len else 'missing'}")
    if is_google_ai_model(vision_model) and not google_key_len:
        st.error("Gemma vision needs `GOOGLE_API_KEY` in Streamlit secrets.")
    if not key_len:
        st.error(
            "Missing `FIREWORKS_API_KEY`. On Streamlit Cloud: App → Settings → Secrets."
        )

    with st.expander("Pipeline (Docker batch)"):
        st.markdown(
            """
- **Kimi K2.6** describe · scene frames @ 512px
- Overlap describe + caption across clips
- Prefetch next 2 downloads
- Pipelined **gpt-oss-120b** judge + retry
- 540s budget · deadline guard → MiniMax fallback
"""
        )
    st.caption("Demo disables judge+retry for speed (~30–90s per clip).")

st.subheader("Input")

with st.form("cc_form", clear_on_submit=False):
    video_url = st.text_input(
        "Video URL",
        value=_DEMO_VIDEO_URL,
        help="Public MP4 URL. Default is a 30s bird clip for fast demos.",
    )

    task_id = st.text_input("Task ID (optional)", value="demo_001")
    styles = st.multiselect("Styles", options=list(STYLES), default=list(STYLES))
    submitted = st.form_submit_button("Generate captions", type="primary")

if submitted:
    _apply_demo_env()

    if not _default_vision_model():
        st.error("Missing `VISION_MODEL` (or `FIREWORKS_MODEL`).")
        st.stop()
    if not _default_caption_model():
        st.error("Missing `CAPTION_MODEL` (or `FIREWORKS_MODEL`).")
        st.stop()
    vision_model = _default_vision_model()
    caption_model = _default_caption_model()
    needs_fireworks = not is_openrouter_model(caption_model) and not is_google_ai_model(
        caption_model
    )
    needs_openrouter = is_openrouter_model(vision_model) or is_openrouter_model(
        caption_model
    )
    needs_google = is_google_ai_model(vision_model)
    if needs_fireworks and not _env("FIREWORKS_API_KEY"):
        st.error("Missing `FIREWORKS_API_KEY`.")
        st.stop()
    if needs_openrouter and not _env("OPENROUTER_API_KEY"):
        st.error("Missing `OPENROUTER_API_KEY` for OpenRouter vision model.")
        st.stop()
    if needs_google and not (_env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY")):
        st.error("Missing `GOOGLE_API_KEY` for Google AI Studio vision model.")
        st.stop()

    if not styles:
        st.error("Pick at least one style.")
        st.stop()

    with st.spinner("Downloading, describing, and captioning (typically 30–90s)..."):
        vision_client = resolve_llm_client(vision_model)
        caption_client = resolve_llm_client(caption_model)
        if caption_client is None:
            st.error("Caption model must use Fireworks or OpenRouter.")
            st.stop()
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            tasks_path = td_path / "tasks.json"
            results_path = td_path / "results.json"

            resolved_video = (video_url or "").strip()
            if not resolved_video:
                st.error("Provide a video URL.")
                st.stop()

            tasks_payload = [
                {
                    "task_id": (task_id or "demo_001").strip(),
                    "video_url": resolved_video,
                    "styles": styles,
                }
            ]
            tasks_path.write_text(json.dumps(tasks_payload, indent=2), encoding="utf-8")

            run_full_tasks(
                tasks_path=tasks_path,
                results_path=results_path,
                client=vision_client or caption_client,
                caption_client=caption_client,
                vision_model=vision_model,
                caption_model=caption_model,
            )

            raw = results_path.read_text(encoding="utf-8")
            results = json.loads(raw)

    st.success("Done.")

    one = results[0]
    captions: dict[str, str] = one.get("captions", {})

    st.subheader("Output")
    for style in styles:
        st.markdown(
            f"""
<div class="cc-card">
  <div class="cc-label">{style}</div>
  <div>{(captions.get(style, "") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.download_button(
        "Download results.json",
        data=raw.encode("utf-8"),
        file_name="results.json",
        mime="application/json",
        use_container_width=True,
    )
