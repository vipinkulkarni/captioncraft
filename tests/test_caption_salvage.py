"""Tests for caption salvage heuristics."""

from src.caption import _is_bad_output, _is_meta_leak, _normalize_style_output
from src.caption_salvage import (
    caption_hard_fail_reason,
    is_describe_field_dump,
    is_drafting_junk,
    is_incomplete_caption,
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
        # Leaks observed in the 2026-07-11 live run and gemma_v2 bake-off:
        'But must include color: blue cap, white face. Use "blue-capped bird".',
        "Then punchline could reference the distant cars as a contrast to the peace.",
        'But does it match "exits frame leaving only marshy water"? Yes. Also uses "marshy" from background.',
        "After a vigorous splash session, it flies off, leaving the water to.",
        "Metaphor: background process, frozen mid-request. Good.",
        "We need at least one color from the subjects list here.",
        "Not necessary but could add flavor.",
        'Let\'s think: "Pecks at the twigs like debugging a nested if-else."',
        '"Unimpressed by the mountain view" is deadpan. That works.',
        "But we also have the large white rabbit? Not required to.",
        # Leaks that passed validation in the gemma_v2_fixed_salvage bake-off:
        "So we can reference the sunlight filtering and the stillness.",
        '"Teal and blue waves crash against dark rocks like a debug session hitting a wall." Hmm.',
        '"Waves crash against dark brown rocks like a server handling too many requests." But need to tie to later.',
        "A large white rabbit closes its.",
        "Then undercut: The white rabbit, meanwhile, is too busy closing its eyes to notice. "
        "That uses late action of white rabbit closing eyes. Also.",
        'Maybe: "The pigeon pecks at the twigs with its yellowish beak, like a critic scanning a draft. '
        'Then it lifts its grey head, white patch flashing, and stares — waiting for applause." That\'s.',
        '"Pigeon pecks at twigs like a developer debugging a messy PR. Then lifts its head, realizing '
        'the fix was in the log all along." But need to match actions: pecks at twigs and log, then.',
        "Kicker: maybe the bright overhead lighting makes it mundane.",
        'So her clothes. Use "orange blouse" or "beige blouse". Let\'s adjust:.',
        "Like relatable absurd comparison: like she's trying to unlock a secret code. "
        "Then punchline: She keeps glancing down at.",
        'Possibly: "The cursor hovers over a line of pink-highlighted code like a developer waiting '
        "for an API response. The autocomplete menu pops up with options, each a potential merge "
        'request for the next line." But need to match actions: early action is code displayed, late action is.',
        "Relatable: programmers know autocomplete can be weird. Punchline: absurd suggestions. Good.",
        'Let\'s refine: "Ginger kitten sits in the shade like a server waiting for an API call. '
        'Then it walks forward through the leaves, deploying a new patch." That uses.',
        # Leaks from salvage_v3 bake-off (2026-07-11):
        "Notable moments: grey rabbit crawls out of grassy mound; large white rabbit stands amidst giant flowers.",
        'Keep simple: "like it has somewhere important to be" - ironic because it\'s just a rabbit.',
        "Setting outdoor with green foliage implied. Metaphor: API endpoint, deploy.",
        "Need to use subject's color. Could say.",
        'Need to follow shape: "[Visible action] like [dev/engineering metaphor]. [Punchline].".',
        "Absurd comparison: like she's trying to decode a secret message.",
        "But ensure using at least one color: pink and blue. Actions: early (code displayed) and late (autocomplete appears). Setting: indoor computer screen. Works.",
        # Prompt-shape echo (test6 e09 humorous_non_tech closedworld run):
        'Metaphor/joke style: absurd comparison. Shape: "[Subject] [does something ordinary] '
        'like [relatable absurd comparison]. [Punchline that exaggerates or flips it '
        'using a late-scene beat].".',
        "Another: This waterfall streams down like a never-ending shower that forgot to turn off.",
        # train10_quality_humor_20260712 leak leftovers:
        'However, "solved the case" is a bit cliché. Maybe another angle.',
        "But we can include high bun or cross necklace if it fits the joke.",
        'The previous caption likely used a metaphor like "waves are like..." We need a different angle. Perhaps.',
        "The bill is not a physical object in scene, it's a comparison. Should be okay.",
        "But careful: don't invent new objects. Use the given: red track, white numbers.",
        "Revised: The light pink sleeve works the knife in rapid up-and-down motion.",
        '"Lazy river of metal" is a metaphor but doesn\'t introduce new scene.',
        '"grey asphalt" maybe not needed. Use "blue glass buildings". Let\'s try.',
        "Absurd comparisons are tone only; they must not add new physical objects "
        "as if they are in the scene (no inventing brunch tables, VIP ropes, props not in the facts).",
    ]

    def test_junk_samples_flagged(self):
        for text in self.JUNK_SAMPLES:
            assert is_drafting_junk(text), repr(text[:70])

    def test_bad_output_rejects_junk(self):
        for text in self.JUNK_SAMPLES:
            bad, reason = _is_bad_output(text, style="humorous_tech")
            assert bad, (text[:70], reason)

    # Real captions from past runs that superficially resemble drafting
    # (phrasal-verb tails, dev metaphors, "like" mid-sentence) but are valid.
    LEGIT_SAMPLES = [
        "Teal water and pale sand try to relax, but those white-foamed waves "
        "keep crashing the party. The dark rocks just sit there, soaking it all in.",
        "A small white bird commits to being the most dynamic element in this "
        "static coastal tableau. Even the waves seem to be phoning it in.",
        "A pigeon in a pastel waistcoat pecks at twigs like it's filing paperwork, "
        "then stares up as if expecting applause. Nature's middle manager, clocking in.",
        "Earth at night looks so peaceful from up here, but those orange lights "
        "mean someone's still stuck in traffic. The sun peeking over the edge is "
        "just rubbing it in.",
        "All that white water rushing down the mossy rocks, and the pool just "
        "sits there taking it. That's the kind of chill I want in my life.",
        "A pigeon with a white neck patch pecks at a pile of twigs like a dev "
        "debugging legacy code. It lifts its head to check for merge conflicts, "
        "then dives back in.",
        "Pecking at the twig pile like a developer debugging a stubborn commit. "
        "The white neck patch bobs up to check for side effects before diving back in.",
        "Turquoise water squeezes through a rock hallway just to throw itself "
        "into the ocean. The white houses cling to a green cliff like they're "
        "afraid of falling in.",
        "Guess even planets can't get a quiet night in.",
        # Tails that superficially resemble drafting-note endings but are legit.
        "The rocks have seen a thousand waves and remain a force to be reckoned with.",
        "Ten minutes of pecking and the twig pile is exactly where it was. That's just how it is.",
        "The kitten pauses mid-step, plotting something only known to her.",
    ]

    def test_legit_captions_not_flagged(self):
        for text in self.LEGIT_SAMPLES:
            assert not is_drafting_junk(text), repr(text[:70])


class TestHardFailGates:
    def test_truncated_one_liner(self):
        text = "The black editor screen watches its mult."
        assert is_incomplete_caption(text)
        assert caption_hard_fail_reason(text) == "incomplete"

    def test_describe_field_dump(self):
        text = (
            "Background: coral reef with turquoise water. "
            "Notable moments: fish swimming past rocks."
        )
        assert is_describe_field_dump(text)
        assert caption_hard_fail_reason(text) == "describe-dump"

    def test_finished_two_sentence_ok(self):
        text = (
            "A dark code editor fills the frame with pink and green syntax. "
            "An if statement appears as the cursor blinks at the end of the line."
        )
        assert not is_incomplete_caption(text)
        assert not is_describe_field_dump(text)
        assert caption_hard_fail_reason(text) == ""


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

    def test_quoted_span_recovered_from_drafting_notes(self):
        raw = (
            'Let\'s refine: "Ginger kitten sits in the shade like a server waiting '
            "for an API call. Then it walks forward through the leaves, deploying "
            'a new patch." That uses.'
        )
        salvaged, _ = pick_valid_candidate(
            raw, style="humorous_tech", is_valid=_is_bad_output
        )
        assert salvaged is not None
        assert salvaged.startswith("Ginger kitten sits")
        assert "That uses" not in salvaged

    def test_quoted_span_recovered_when_notes_follow(self):
        raw = (
            '"Pigeon pecks at twigs like a developer debugging a messy PR. '
            'Then lifts its head, realizing the fix was in the log all along." '
            "But need to match actions: pecks at twigs and log, then."
        )
        salvaged, _ = pick_valid_candidate(
            raw, style="humorous_tech", is_valid=_is_bad_output
        )
        assert salvaged is not None
        assert salvaged.startswith("Pigeon pecks")
        assert "But need" not in salvaged

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


class TestSceneMismatchAndExampleEcho:
    def test_prompt_example_echo_flagged(self):
        from src.caption_salvage import (
            is_prompt_example_echo,
            load_prompt_example_captions,
        )

        load_prompt_example_captions.cache_clear()
        leaked = (
            "The orange kitten marches through the foliage like it owns the lease. "
            "Tail raised, it approaches the camera as if we should be honored."
        )
        assert is_prompt_example_echo(leaked)

    def test_coding_clip_rejects_kitten_caption(self):
        desc = (
            "Primary subject: code editor screen (colors: black background)\n"
            "Setting: indoor close-up of computer\n"
            "Actions (early): code is typed\n"
            "Actions (late): autocomplete appears"
        )
        caption = (
            "The orange kitten marches through the foliage like it owns the lease. "
            "Tail raised, it approaches the camera as if we should be honored."
        )
        bad, reason = _is_bad_output(caption, style="sarcastic", description=desc)
        assert bad
        assert reason in ("MetaLeak", "SceneMismatch")

    def test_foreign_animal_without_example_echo(self):
        from src.caption_salvage import scene_subject_mismatch

        desc = (
            "Primary subject: code editor screen\n"
            "Setting: indoor\n"
            "Actions (early): typing\n"
            "Actions (late): scrolling"
        )
        caption = (
            "A fluffy kitten stares at the glowing monitor like it understands CSS. "
            "Then it walks away, tail high, unfinished."
        )
        assert scene_subject_mismatch(desc, caption)

    def test_matching_primary_subject_ok(self):
        desc = (
            "Primary subject: orange kitten (colors: orange, white)\n"
            "Setting: outdoor garden\n"
            "Actions (early): sits still\n"
            "Actions (late): walks forward"
        )
        caption = (
            "An orange kitten sits under green bushes in dappled light. "
            "It then walks toward the camera with its tail raised."
        )
        bad, reason = _is_bad_output(caption, style="formal", description=desc)
        assert not bad, reason
