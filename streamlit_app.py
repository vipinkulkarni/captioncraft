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


st.set_page_config(page_title="CaptionCraft Demo", page_icon="🎬", layout="centered")

st.title("CaptionCraft demo")
st.caption("Generate 4 styled captions from a video URL using the same pipeline as the Docker agent.")

with st.sidebar:
    st.subheader("Config")
    st.text_input("Vision model", value=_default_vision_model(), disabled=True)
    st.text_input("Caption model", value=_default_caption_model(), disabled=True)
    st.text_input("Parallel styles", value=_env("PARALLEL_STYLES", "1"), disabled=True)

    has_key = bool(_env("FIREWORKS_API_KEY"))
    st.write(f"Fireworks API key: {'set' if has_key else 'missing'}")
    if not has_key:
        st.error(
            "Missing `FIREWORKS_API_KEY`. On Streamlit Cloud, set it in App → Settings → Secrets."
        )

st.subheader("Input")
video_url = st.text_input(
    "Video URL",
    placeholder="https://.../video.mp4",
)
task_id = st.text_input("Task ID (optional)", value="demo_001")
styles = st.multiselect("Styles", options=list(STYLES), default=list(STYLES))

run_btn = st.button("Generate captions", type="primary", disabled=not (video_url and styles))

if run_btn:
    if not _default_vision_model():
        st.error("Missing `VISION_MODEL` (or `FIREWORKS_MODEL`).")
        st.stop()
    if not _default_caption_model():
        st.error("Missing `CAPTION_MODEL` (or `FIREWORKS_MODEL`).")
        st.stop()
    if not _env("FIREWORKS_API_KEY"):
        st.error("Missing `FIREWORKS_API_KEY`.")
        st.stop()

    with st.spinner("Running describe + style captions..."):
        client = get_fireworks_client()
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            tasks_path = td_path / "tasks.json"
            results_path = td_path / "results.json"

            tasks_payload = [
                {
                    "task_id": (task_id or "demo_001").strip(),
                    "video_url": video_url.strip(),
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
        st.markdown(f"**{style}**")
        st.write(captions.get(style, ""))

    st.download_button(
        "Download results.json",
        data=raw.encode("utf-8"),
        file_name="results.json",
        mime="application/json",
        use_container_width=True,
    )
