import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from src.caption import STYLES, get_fireworks_client
from src.pipeline import run_full_tasks


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _default_vision_model() -> str:
    return _env("VISION_MODEL") or _env("FIREWORKS_MODEL")


def _default_caption_model() -> str:
    return _env("CAPTION_MODEL") or _env("FIREWORKS_MODEL") or _default_vision_model()

def _pretty_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return ""
    lower = m.lower()
    if "minimax-m3" in lower:
        return "MiniMax M3 (vision)"
    if "deepseek-v4-flash" in lower:
        return "DeepSeek V4 Flash (text)"
    # Fall back to last path segment for readability.
    return m.rsplit("/", 1)[-1]


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
st.caption("Generate 4 styled captions from a video URL using the same pipeline as the Docker agent.")

with st.sidebar:
    st.subheader("Config")
    st.text_input("Vision model", value=_pretty_model(_default_vision_model()), disabled=True)
    st.text_input("Caption model", value=_pretty_model(_default_caption_model()), disabled=True)

    key_len = len(_env("FIREWORKS_API_KEY"))
    st.write(f"Fireworks API key: {'set' if key_len else 'missing'}")
    if 0 < key_len < 16:
        st.warning("Your API key looks like a placeholder (very short).")
    if not key_len:
        st.error(
            "Missing `FIREWORKS_API_KEY`. On Streamlit Cloud, set it in App → Settings → Secrets."
        )

st.subheader("Input")

with st.form("cc_form", clear_on_submit=False):
    video_url = st.text_input("Video URL", placeholder="https://.../video.mp4")

    task_id = st.text_input("Task ID (optional)", value="demo_001")
    styles = st.multiselect("Styles", options=list(STYLES), default=list(STYLES))
    submitted = st.form_submit_button("Generate captions", type="primary")

if submitted:
    if not _default_vision_model():
        st.error("Missing `VISION_MODEL` (or `FIREWORKS_MODEL`).")
        st.stop()
    if not _default_caption_model():
        st.error("Missing `CAPTION_MODEL` (or `FIREWORKS_MODEL`).")
        st.stop()
    if not _env("FIREWORKS_API_KEY"):
        st.error("Missing `FIREWORKS_API_KEY`.")
        st.stop()

    if not styles:
        st.error("Pick at least one style.")
        st.stop()

    with st.spinner("Running describe + style captions..."):
        client = get_fireworks_client()
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
                client=client,
                vision_model=_default_vision_model(),
                caption_model=_default_caption_model(),
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
