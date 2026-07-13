"""Tests for LLM-as-judge parsing and pass logic."""

import json

from src.llm_judge import (
    CaptionJudgeScore,
    ClipJudgeResult,
    JudgeFileResult,
    aggregate_clip_judges,
    collect_calibration_samples,
    format_calibration_report,
    format_judge_summary,
    judge_result_to_dict,
    parse_distinctness_response,
    parse_judge_response,
    parse_style_judge_response,
    resolve_descriptions_path,
    resolve_judge_models,
    resolve_judge_min_score,
    _clamp_unit_score,
    _parse_unit_score,
)


SAMPLE_JUDGE_JSON = json.dumps(
    {
        "captions": {
            "formal": {
                "accuracy": 0.9,
                "style_match": 1.0,
                "issue": "",
            },
            "sarcastic": {
                "accuracy": 0.8,
                "style_match": 0.8,
                "issue": "",
            },
            "humorous_tech": {
                "accuracy": 0.8,
                "style_match": 0.4,
                "issue": "weak tech joke",
            },
            "humorous_non_tech": {
                "accuracy": 0.8,
                "style_match": 0.8,
                "issue": "",
            },
        },
        "cross_style_distinctness": 0.8,
        "distinctness_note": "",
    }
)


class TestParseJudgeResponse:
    def test_parses_valid_payload(self):
        scores, distinctness, note, err = parse_judge_response(SAMPLE_JUDGE_JSON)
        assert err == ""
        assert distinctness == 0.8
        assert note == ""
        assert scores["formal"].style_match == 1.0
        assert scores["formal"].accuracy == 0.9
        assert scores["humorous_tech"].issue == "weak tech joke"

    def test_rejects_legacy_1_to_5_scores(self):
        raw = json.dumps(
            {
                "captions": {
                    "formal": {"style_fit": 5, "accuracy": 5, "issue": ""},
                    "sarcastic": {"style_fit": 3, "accuracy": 3, "issue": ""},
                    "humorous_tech": {"style_fit": 3, "accuracy": 3, "issue": ""},
                    "humorous_non_tech": {
                        "style_fit": 3,
                        "accuracy": 3,
                        "issue": "",
                    },
                },
                "cross_style_distinctness": 0.0,
            }
        )
        scores, distinctness, _note, err = parse_judge_response(raw)
        assert err == ""
        assert scores["formal"].skipped
        assert scores["formal"].skip_reason == "judge-invalid-scores"
        assert scores["sarcastic"].skipped
        assert distinctness == 0.0

    def test_accepts_legacy_style_fit_key_in_unit_range(self):
        raw = json.dumps(
            {
                "captions": {
                    "formal": {"style_fit": 0.9, "accuracy": 0.8, "issue": ""},
                    "sarcastic": {"style_match": 0.7, "accuracy": 0.7, "issue": ""},
                    "humorous_tech": {"style_match": 0.5, "accuracy": 0.5, "issue": ""},
                    "humorous_non_tech": {"style_match": 0.6, "accuracy": 0.6, "issue": ""},
                },
                "cross_style_distinctness": 0.5,
            }
        )
        scores, distinctness, _n, err = parse_judge_response(raw)
        assert err == ""
        assert scores["formal"].style_match == 0.9
        assert distinctness == 0.5

    def test_unit_scores_in_range_ok(self):
        raw = json.dumps(
            {
                "captions": {
                    "formal": {"accuracy": 0.4, "style_match": 0.9, "issue": ""},
                    "sarcastic": {"accuracy": 0.0, "style_match": 1.0, "issue": ""},
                    "humorous_tech": {"accuracy": 1.0, "style_match": 0.5, "issue": ""},
                    "humorous_non_tech": {"accuracy": 0.8, "style_match": 0.8, "issue": ""},
                },
                "cross_style_distinctness": 0.6,
            }
        )
        scores, _d, _n, err = parse_judge_response(raw)
        assert err == ""
        assert not scores["formal"].skipped
        assert scores["formal"].accuracy == 0.4
        assert scores["formal"].style_match == 0.9

    def test_invalid_json(self):
        _scores, _d, _n, err = parse_judge_response("{bad")
        assert err.startswith("InvalidJSON")


class TestParseStyleJudgeResponse:
    def test_parses_single_style(self):
        raw = json.dumps({"style_match": 0.8, "accuracy": 1.0, "issue": ""})
        score, err = parse_style_judge_response(raw, style="formal")
        assert err == ""
        assert score is not None
        assert score.style_match == 0.8
        assert score.accuracy == 1.0

    def test_parses_truncated_json(self):
        raw = '{"style_match":0.8,"accuracy":1.0,"issue":"weak'
        score, err = parse_style_judge_response(raw, style="formal")
        assert err == ""
        assert score is not None
        assert score.style_match == 0.8
        assert score.accuracy == 1.0

    def test_rejects_scores_outside_unit_interval(self):
        raw = json.dumps({"accuracy": 4, "style_match": 5, "issue": ""})
        score, err = parse_style_judge_response(raw, style="formal")
        assert score is None
        assert "0, 1" in err

    def test_meta_leak_clamps_and_flags(self):
        raw = json.dumps(
            {
                "accuracy": 0.95,
                "style_match": 0.9,
                "meta_leak": True,
                "issue": "planning prose",
            }
        )
        score, err = parse_style_judge_response(raw, style="humorous_non_tech")
        assert err == ""
        assert score is not None
        assert score.meta_leak
        assert score.accuracy <= 0.2
        assert score.style_match <= 0.2
        assert "meta-leak" in score.issue.lower()
        assert not score.passes(min_score=0.8)

    def test_meta_leak_from_truncated_json(self):
        raw = '{"accuracy":0.9,"style_match":0.9,"meta_leak":true,"issue":"revised'
        score, err = parse_style_judge_response(raw, style="formal")
        assert err == ""
        assert score is not None
        assert score.meta_leak
        assert not score.passes(min_score=0.8)


class TestDistinctnessUnitScale:
    def test_parses_unit_distinctness(self):
        score, note, err = parse_distinctness_response(
            '{"cross_style_distinctness":0.7,"distinctness_note":"ok"}'
        )
        assert err == ""
        assert score == 0.7
        assert note == "ok"

    def test_rejects_legacy_1_to_5_distinctness(self):
        score, _note, err = parse_distinctness_response(
            '{"cross_style_distinctness":4,"distinctness_note":""}'
        )
        assert score == 0.0
        assert "0, 1" in err


class TestJudgePassLogic:
    def test_passes_at_threshold(self):
        score = CaptionJudgeScore(style="formal", style_match=0.8, accuracy=0.8)
        assert score.passes(min_score=0.8)
        assert not CaptionJudgeScore(
            style="formal", style_match=0.8, accuracy=0.7
        ).passes(min_score=0.8)

    def test_passes_on_mean_not_min_axis(self):
        # Punchy official-style: slightly lower accuracy, strong style → mean ≥ 0.9
        score = CaptionJudgeScore(style="sarcastic", accuracy=0.86, style_match=0.96)
        assert abs(score.average - 0.91) < 1e-9
        assert score.passes(min_score=0.9)

    def test_min_score_clamps_without_legacy_remap(self):
        # Values >1 are clamped to 1.0, not divided by 5.
        assert resolve_judge_min_score(4) == 1.0
        assert resolve_judge_min_score(0.8) == 0.8
        score = CaptionJudgeScore(style="formal", style_match=0.8, accuracy=0.8)
        assert not score.passes(min_score=4)

    def test_skipped_fails(self):
        score = CaptionJudgeScore(
            style="formal",
            style_match=1.0,
            accuracy=1.0,
            skipped=True,
            skip_reason="error",
        )
        assert not score.passes(min_score=0.8)

    def test_meta_leak_fails_even_with_high_scores(self):
        score = CaptionJudgeScore(
            style="formal",
            style_match=1.0,
            accuracy=1.0,
            meta_leak=True,
            issue="drafting",
        )
        assert not score.passes(min_score=0.8)

    def test_scene_mismatch_clamps_llm_rubber_stamp(self):
        from src.llm_judge import _clamp_score_for_scene

        soft = CaptionJudgeScore(
            style="sarcastic",
            accuracy=0.95,
            style_match=0.95,
            issue="",
        )
        desc = (
            "Primary subject: code editor screen (colors: black)\n"
            "Setting: indoor\n"
            "Actions (early): typing\n"
            "Actions (late): autocomplete"
        )
        caption = (
            "The orange kitten marches through the foliage like it owns the lease. "
            "Tail raised, it approaches the camera as if we should be honored."
        )
        clamped = _clamp_score_for_scene(soft, caption=caption, description=desc)
        assert clamped.meta_leak
        assert clamped.accuracy <= 0.1
        assert not clamped.passes(min_score=0.8)

    def test_incomplete_clamps_llm_rubber_stamp(self):
        from src.llm_judge import _clamp_score_for_scene

        soft = CaptionJudgeScore(
            style="sarcastic",
            accuracy=0.95,
            style_match=0.9,
            issue="",
        )
        caption = "The black editor screen watches its mult."
        clamped = _clamp_score_for_scene(
            soft, caption=caption, description="", style="sarcastic"
        )
        assert clamped.meta_leak
        assert clamped.accuracy <= 0.1
        assert "incomplete" in clamped.issue

    def test_describe_dump_auto_skips(self):
        from src.llm_judge import _auto_skip_caption

        text = (
            "Background: coral reef with turquoise water. "
            "Notable moments: a fish glides past."
        )
        skipped = _auto_skip_caption(text, "humorous_non_tech")
        assert skipped is not None
        assert skipped.skip_reason == "describe-dump"
        assert skipped.accuracy == 0.0

    def test_summary_and_failures(self):
        clip = ClipJudgeResult(
            task_id="e01",
            captions={
                "formal": CaptionJudgeScore(style="formal", style_match=0.9, accuracy=0.9),
                "humorous_tech": CaptionJudgeScore(
                    style="humorous_tech",
                    style_match=0.4,
                    accuracy=0.8,
                    issue="no tech humor",
                ),
            },
            cross_style_distinctness=0.4,
            distinctness_note="too similar",
        )
        result = JudgeFileResult(
            clips=[clip],
            model="test-model",
            min_score=0.8,
            descriptions_provided=False,
        )
        assert result.passes == 1
        assert result.total == 2
        assert result.mean_score is not None
        assert 0.6 < result.mean_score < 0.85
        assert any("humorous_tech" in f for f in result.failures())
        assert result.low_distinctness()
        text = format_judge_summary(result)
        assert "leaderboard_proxy=" in text
        assert "per-caption scores" in text
        assert "e01/humorous_tech:" in text
        assert "mean=" in text
        assert "pass@" not in text
        exported = judge_result_to_dict(result)
        assert exported["passes"] == 1
        assert exported["mean_score"] == result.mean_score


class TestPanelAggregation:
    def test_median_aggregate_clip(self):
        judge_a = ClipJudgeResult(
            task_id="e01",
            captions={
                "formal": CaptionJudgeScore(style="formal", style_match=0.8, accuracy=0.8),
                "humorous_tech": CaptionJudgeScore(
                    style="humorous_tech", style_match=0.4, accuracy=0.8
                ),
            },
            cross_style_distinctness=0.8,
        )
        judge_b = ClipJudgeResult(
            task_id="e01",
            captions={
                "formal": CaptionJudgeScore(style="formal", style_match=0.4, accuracy=0.6),
                "humorous_tech": CaptionJudgeScore(
                    style="humorous_tech",
                    style_match=0.6,
                    accuracy=0.6,
                    issue="weak joke",
                ),
            },
            cross_style_distinctness=0.4,
        )
        aggregated = aggregate_clip_judges({"a": judge_a, "b": judge_b})
        assert aggregated.captions["formal"].style_match == 0.6
        assert aggregated.captions["formal"].accuracy == 0.7
        assert aggregated.captions["humorous_tech"].style_match == 0.5
        assert aggregated.cross_style_distinctness == 0.6

    def test_panel_summary_lists_per_judge(self):
        sub = JudgeFileResult(
            clips=[],
            model="accounts/fireworks/models/gpt-oss-120b",
            min_score=0.8,
            descriptions_provided=True,
        )
        result = JudgeFileResult(
            clips=[],
            model="panel(median): gpt-oss-120b, kimi-k2-6",
            min_score=0.8,
            descriptions_provided=True,
            panel_models=[
                "accounts/fireworks/models/gpt-oss-120b",
                "accounts/fireworks/models/kimi-k2-6",
            ],
            per_judge={"accounts/fireworks/models/gpt-oss-120b": sub},
        )
        text = format_judge_summary(result)
        assert "per-judge:" in text
        assert "gpt-oss-120b" in text

    def test_resolve_panel_models_default(self, monkeypatch):
        monkeypatch.delenv("JUDGE_MODELS", raising=False)
        monkeypatch.delenv("JUDGE_MODEL", raising=False)
        models = resolve_judge_models(panel=True)
        assert len(models) == 3
        assert any("deepseek" in m for m in models)

    def test_calibration_collects_near_threshold(self):
        clip = ClipJudgeResult(
            task_id="e01",
            captions={
                "formal": CaptionJudgeScore(style="formal", style_match=0.8, accuracy=0.8),
                "humorous_tech": CaptionJudgeScore(
                    style="humorous_tech", style_match=1.0, accuracy=1.0
                ),
            },
        )
        result = JudgeFileResult(
            clips=[clip],
            model="test",
            min_score=0.8,
            descriptions_provided=False,
        )
        data = [
            {
                "task_id": "e01",
                "captions": {"formal": "A formal caption.", "humorous_tech": "Great joke."},
            }
        ]
        samples = collect_calibration_samples(result, data, limit=5)
        assert len(samples) == 1
        assert samples[0]["style"] == "formal"
        assert "style_match" in samples[0]
        report = format_calibration_report(samples)
        assert "e01/formal" in report


class TestUnitScoreParsing:
    def test_parse_rejects_out_of_range(self):
        assert _parse_unit_score(4) is None
        assert _parse_unit_score(5) is None
        assert _parse_unit_score(-0.1) is None
        assert _parse_unit_score(1.01) is None

    def test_parse_accepts_unit_interval(self):
        assert _parse_unit_score(0.4) == 0.4
        assert _parse_unit_score(0.9) == 0.9
        assert _parse_unit_score(0) == 0.0
        assert _parse_unit_score(1) == 1.0

    def test_clamp_no_longer_promotes_1_to_5(self):
        assert _clamp_unit_score(4) == 0.0
        assert _clamp_unit_score(0.8) == 0.8


class TestResolveDescriptionsPath:
    def test_prefers_live_sibling(self, tmp_path):
        results = tmp_path / "results.json"
        results.write_text("[]", encoding="utf-8")
        live = tmp_path / "descriptions_live.json"
        live.write_text('{"descriptions": {"e01": "live"}}', encoding="utf-8")
        assert resolve_descriptions_path(None, results_path=results) == live

    def test_explicit_path_wins(self, tmp_path):
        results = tmp_path / "results.json"
        results.write_text("[]", encoding="utf-8")
        live = tmp_path / "descriptions_live.json"
        live.write_text('{"descriptions": {"e01": "live"}}', encoding="utf-8")
        explicit = tmp_path / "other.json"
        explicit.write_text('{"descriptions": {}}', encoding="utf-8")
        assert resolve_descriptions_path(explicit, results_path=results) == explicit
