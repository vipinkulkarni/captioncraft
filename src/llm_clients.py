"""LLM client routing (Fireworks, OpenRouter, Google AI Studio).

Kept separate from caption.py so lightweight callers (e.g. Streamlit) avoid
importing the full caption stack and hitting circular-import edge cases.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.env import get_float_env

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_REPO_ROOT / ".env")


def is_google_ai_model(model: str) -> bool:
    """Google AI Studio / Gemini API model ids (direct, not OpenRouter)."""
    slug = model.strip().lower()
    if not slug or slug.startswith("accounts/"):
        return False
    if slug.startswith("google-ai/"):
        return True
    return slug.startswith("gemma-")


def is_openrouter_model(model: str) -> bool:
    """OpenRouter slugs look like provider/model-name (not Fireworks accounts/ paths)."""
    slug = model.strip()
    if not slug or slug.startswith("accounts/") or is_google_ai_model(slug):
        return False
    return "/" in slug


def resolve_google_model_id(model: str) -> str:
    slug = model.strip()
    if slug.lower().startswith("google-ai/"):
        return slug.split("/", 1)[1]
    return slug


def google_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "").strip() or os.environ.get(
        "GEMINI_API_KEY", ""
    ).strip()
    if not key:
        raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY)")
    return key


def get_openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    timeout_s = get_float_env("API_TIMEOUT_S", 45.0)
    headers: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_REFERER", "").strip()
    title = os.environ.get("OPENROUTER_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=timeout_s,
        max_retries=0,
        default_headers=headers or None,
    )


def get_fireworks_client() -> OpenAI:
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing FIREWORKS_API_KEY")

    timeout_s = get_float_env("API_TIMEOUT_S", 45.0)
    return OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
        timeout=timeout_s,
        max_retries=0,
    )


def resolve_llm_client(model: str, *, fallback: OpenAI | None = None) -> OpenAI | None:
    if is_google_ai_model(model):
        google_api_key()
        return None
    if is_openrouter_model(model):
        return get_openrouter_client()
    return fallback or get_fireworks_client()
