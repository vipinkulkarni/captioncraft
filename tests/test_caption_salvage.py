"""Tests for caption salvage heuristics."""

from src.caption import _is_bad_output, _is_meta_leak, _normalize_style_output
from src.caption_salvage import (
    is_drafting_junk,
    iter_salvage_candidates,
    pick_two_sentence_fit,
    pick_valid_candidate,
)


class TestDraftingJunk:
    """Rejections for planning/checklist output that previously passed salvage."""

    JUNK_SAMPLES = [
        'Better: "The park slowly pans across like a lazy Sunday afternoon—green grass, brown brick, and dapp.',
        'But need to match "rolling toward shore" and "crashing". "Deploy" works. "Merge conflict" is dev reference.',
        'Count words: 27. Good. Includes colors: white, purple, gray, orange, red. Actions: emerges, sniffs, lands, startling, watch.',
        'Check: uses color (white neck patch), action (pecking, lifting head), setting (outdoor, twigs). No emojis, under.',
        'That uses pink-breasted (color), pecks (action), dry twigs (surface), foliage (background). Under 50 words.',
        'Better: "The screen types a pink if statement and a blue display assignment, like.',
        'Actually: Orange(1) kitten(2) sits(3) on(4.',
        'Punchline: "Each line resolves a conflict in the DOM." But need to tie to the scene: code editor, dark theme.',
        'Alternatively, use increment variable and element.',
        'Use "camera pulls back" from late. Also mention waves crashing.',
        'Brown and tan boulders sit partially submerged as waves crash against them. The.',
        'Two dark-clad swimmers float in the.',
        'But "types" might not be accurate; it\'s the screen showing typing. Better: "The dark code editor scrolls pink and green syntax like a bored stenographer. The big moment is an if statement and a display style assignment." That.',
    ]

    def test_junk_samples_flagged(self):
        for text in self.JUNK_SAMPLES:
            assert is_drafting_junk(text), repr(text[:70])

    def test_bad_output_rejects_junk(self):
        for text in self.JUNK_SAMPLES:
            bad, reason = _is_bad_output(text, style="humorous_tech")
            assert bad, (text[:70], reason)


class TestSalvageCandidates:
    def test_meta_preamble_then_caption(self):
        raw = (
            "We need to write two short sentences using only these facts. "
            "The orange kitten sits on the dirt path. "
            "It walks forward with its tail raised."
        )
        assert _is_meta_leak(raw)
        salvaged, _ = pick_valid_candidate(raw, style="formal", is_valid=_is_bad_output)
        assert salvaged is not None
        assert "walks forward" in salvaged.lower() or "kitten" in salvaged.lower()
        assert not _is_meta_leak(salvaged)

    def test_label_prefix(self):
        raw = "Caption: Waves roll onto a rocky shore as white foam spreads."
        candidates = iter_salvage_candidates(raw)
        assert any(c.startswith("Waves roll") for c in candidates)

    def test_two_sentence_tail_for_too_long(self):
        raw = (
            "We need to follow the output contract and avoid meta commentary entirely today. "
            "The user wants formal tone with completely new wording required here. "
            "Teal waves crash against dark rocks. White foam spreads across the water."
        )
        fit, _ = pick_two_sentence_fit(raw, style="formal", hard_limit=58, is_valid=_is_bad_output)
        assert fit is not None
        assert "Teal waves" in fit

    def test_normalize_applies_salvage(self):
        raw = (
            "The user wants sarcasm. "
            "A grey pigeon pecks at twigs like a fussy inspector. "
            "Nothing worth reporting."
        )
        normalized = _normalize_style_output(raw, style="sarcastic")
        bad, reason = _is_bad_output(normalized, style="sarcastic")
        assert not bad, reason
