"""Tests for LLM client routing."""

from src.caption import (
    is_google_ai_model,
    is_openrouter_model,
    resolve_caption_model_pool,
    _resolve_google_model_id,
)


class TestOpenRouterRouting:
    def test_fireworks_models_are_not_openrouter(self):
        assert not is_openrouter_model("accounts/fireworks/models/deepseek-v4-flash")
        assert not is_openrouter_model("accounts/fireworks/models/minimax-m3")

    def test_openrouter_slugs_detected(self):
        assert is_openrouter_model("google/gemma-4-26b-a4b-it:free")
        assert is_openrouter_model("google/gemma-4-31b-it:free")

    def test_google_ai_models_not_openrouter(self):
        assert not is_openrouter_model("google-ai/gemma-4-26b-a4b-it")
        assert is_google_ai_model("google-ai/gemma-4-26b-a4b-it")
        assert is_google_ai_model("gemma-4-31b-it")

    def test_resolve_google_model_id(self):
        assert _resolve_google_model_id("google-ai/gemma-4-26b-a4b-it") == "gemma-4-26b-a4b-it"
        assert _resolve_google_model_id("gemma-4-31b-it") == "gemma-4-31b-it"

    def test_default_caption_pool_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv("CAPTION_MODEL_POOL", raising=False)
        assert resolve_caption_model_pool() == []
