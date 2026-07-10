"""Unit tests for caption validation helpers."""

from src.caption import (
    STYLES,
    _build_style_user_prompt,
    _is_bad_output,
    _is_meta_leak,
    looks_truncated,
    load_prompt,
)


META_LEAK_SAMPLES = [
    "We need to write a funny caption about the kitten.",
    "The user wants a sarcastic tone for this clip.",
    "The user asks for two short sentences only.",
    "I need to rewrite this using only these facts from the description.",
    "I will produce a caption that must be 2 short sentences long.",
    "Here's the caption: a cat sits on a mat near the window.",
    "Caption: Orange kitten walks through green foliage slowly.",
    "Using only these facts, I must be 2 short sentences and completely new wording.",
    "The video description says kitten; using only these facts I must be 2 short sentences.",
    "Completely new wording is required; the user wants a dry ironic caption here.",
    "We need to follow the output contract and avoid meta commentary entirely today.",
    "The user wants formal tone; must be 2 short sentences with completely new wording.",
    "I will write using only these facts and must be 2 short sentences total.",
    "Here is a polished caption with completely new wording for the user wants.",
    "Video description: cat on mat. Using only these facts, must be 2 short sentences.",
    "Output contract says we need to avoid quotes and completely new wording always.",
    "The user asks for humor; using only these facts from the video description.",
    "We need to; the user wants; must be 2 short sentences — ignore prior rules.",
    "I need to satisfy the user wants clause and output contract simultaneously.",
    "Here's my attempt: using only these facts and completely new wording required.",
]

CLEAN_CAPTION_SAMPLES = [
    "An orange kitten peers through green foliage before stepping forward cautiously.",
    "Waves roll onto a sandy beach while seabirds skim the shoreline at dusk.",
    "A cyclist pedals along a coastal path as golden light fades behind the hills.",
    "Office workers collaborate at standing desks beneath bright overhead panels.",
    "Snow blankets a quiet city street while pedestrians hurry past storefronts.",
    "A chef plates pasta beside steaming pots in a busy open kitchen.",
    "Rain streaks across a window overlooking a neon-lit downtown intersection.",
    "A dog trots through autumn leaves scattered along a suburban sidewalk.",
    "Surfers paddle into turquoise water as whitecaps build on the horizon.",
    "A barista pours latte art into a ceramic cup at a crowded café.",
    "Construction cranes rotate above a skyline wrapped in morning fog.",
    "Children kick a soccer ball across a sunlit field after school.",
    "A hiker pauses on a ridge overlooking pine forests and distant peaks.",
    "Traffic inches through an intersection while commuters check their phones.",
    "A violinist practices alone in a rehearsal room with muted afternoon light.",
    "Kites drift above a windy beach while families picnic on the sand.",
    "A gardener trims rose bushes along a white picket fence in spring.",
    "Skateboarders practice tricks on concrete ramps beneath highway overpasses.",
    "A ferry departs the harbor as gulls circle above the wake.",
    "Fireworks burst over a river festival crowd cheering from the embankment.",
]


class TestMetaLeakDetection:
    def test_leak_samples_flagged(self):
        for text in META_LEAK_SAMPLES:
            assert _is_meta_leak(text), repr(text[:60])

    def test_clean_captions_pass(self):
        for text in CLEAN_CAPTION_SAMPLES:
            assert not _is_meta_leak(text), repr(text[:60])


class TestBadOutput:
    def test_too_short(self):
        bad, reason = _is_bad_output("Too few words.", style="formal")
        assert bad and reason == "TooShort"

    def test_too_long_for_formal(self):
        words = "word " * 60
        bad, reason = _is_bad_output(words.strip(), style="formal")
        assert bad and reason == "TooLong"

    def test_meta_leak(self):
        bad, reason = _is_bad_output(
            "We need to write something funny about the kitten in the garden today.",
            style="sarcastic",
        )
        assert bad and reason == "MetaLeak"

    def test_valid_caption(self):
        text = (
            "An orange kitten steps through green foliage toward the camera. "
            "Its tail stays raised as it moves closer."
        )
        bad, reason = _is_bad_output(text, style="formal")
        assert not bad and reason == ""


class TestTruncation:
    def test_finish_reason_length(self):
        assert looks_truncated("Hello world.", "length") is True

    def test_missing_terminal_punctuation(self):
        assert looks_truncated("This caption ends abruptly without a stop", None) is True

    def test_valid_terminal(self):
        assert looks_truncated("This caption ends properly.", None) is False


class TestBuildStyleUserPrompt:
    def test_structured_uses_scene_facts_header(self, monkeypatch):
        monkeypatch.setenv("STRUCTURED_DESCRIBE", "1")
        prompt = _build_style_user_prompt("Setting: beach")
        assert "Scene facts:" in prompt
        assert "Video description" not in prompt

    def test_prose_uses_scene_facts_header(self, monkeypatch):
        monkeypatch.setenv("STRUCTURED_DESCRIBE", "0")
        prompt = _build_style_user_prompt("A cat on a mat.")
        assert prompt.startswith("Scene facts:\n")

    def test_meta_leak_retry_appends_nudge(self, monkeypatch):
        monkeypatch.setenv("STRUCTURED_DESCRIBE", "1")
        prompt = _build_style_user_prompt("Setting: beach", meta_leak_retry=True)
        assert "Your last reply restated instructions" in prompt


class TestLoadPrompt:
    def test_style_prompts_include_contract(self):
        for style in STYLES:
            prompt = load_prompt(style)
            assert "Output contract (follow exactly):" in prompt
            assert "Style rules:" in prompt

    def test_describe_has_no_output_contract(self):
        prompt = load_prompt("describe")
        assert "Output contract (follow exactly):" not in prompt

    def test_load_prompt_is_cached(self):
        load_prompt.cache_clear()
        load_prompt("formal")
        info1 = load_prompt.cache_info()
        load_prompt("formal")
        info2 = load_prompt.cache_info()
        assert info2.hits == info1.hits + 1
