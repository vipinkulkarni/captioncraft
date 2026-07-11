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
    parse_judge_response,
    parse_style_judge_response,
    resolve_judge_models,
)


SAMPLE_JUDGE_JSON = json.dumps(
    {
        "captions": {
            "formal": {
                "style_fit": 5,
                "accuracy": 4,
                "specificity": 4,
                "issue": "",
            },
            "sarcastic": {
                "style_fit": 4,
                "accuracy": 4,
                "specificity": 3,
                "issue": "",
            },
            "humorous_tech": {
                "style_fit": 2,
                "accuracy": 4,
                "specificity": 4,
                "issue": "weak tech joke",
            },
            "humorous_non_tech": {
                "style_fit": 4,
                "accuracy": 4,
                "specificity": 4,
                "issue": "",
            },
        },
        "cross_style_distinctness": 4,
        "distinctness_note": "",
    }
)


class TestParseJudgeResponse:
    def test_parses_valid_payload(self):
        scores, distinctness, note, err = parse_judge_response(SAMPLE_JUDGE_JSON)
        assert err == ""
        assert distinctness == 4
        assert note == ""
        assert scores["formal"].style_fit == 5
        assert scores["humorous_tech"].issue == "weak tech joke"

    def test_clamps_scores(self):
        raw = json.dumps(
            {
                "captions": {
                    "formal": {"style_fit": 9, "accuracy": 0, "specificity": 3, "issue": ""},
                    "sarcastic": {"style_fit": 3, "accuracy": 3, "specificity": 3, "issue": ""},
                    "humorous_tech": {"style_fit": 3, "accuracy": 3, "specificity": 3, "issue": ""},
                    "humorous_non_tech": {
                        "style_fit": 3,
                        "accuracy": 3,
                        "specificity": 3,
                        "issue": "",
                    },
                },
                "cross_style_distinctness": 0,
            }
        )
        scores, distinctness, _note, err = parse_judge_response(raw)
        assert err == ""
        assert scores["formal"].style_fit == 5
        assert scores["formal"].accuracy == 1
        assert distinctness == 1

    def test_invalid_json(self):
        _scores, _d, _n, err = parse_judge_response("{bad")
        assert err.startswith("InvalidJSON")


class TestParseStyleJudgeResponse:
    def test_parses_single_style(self):
        raw = json.dumps({"style_fit": 4, "accuracy": 5, "specificity": 3, "issue": ""})
        score, err = parse_style_judge_response(raw, style="formal")
        assert err == ""
        assert score is not None
        assert score.style_fit == 4


    def test_parses_truncated_json(self):
        raw = '{"style_fit":4,"accuracy":5,"specificity":3,"issue":"weak'
        score, err = parse_style_judge_response(raw, style="formal")
        assert err == ""
        assert score is not None
        assert score.style_fit == 4
        assert score.specificity == 3


class TestJudgePassLogic:
    def test_passes_at_threshold(self):
        score = CaptionJudgeScore(style="formal", style_fit=3, accuracy=3, specificity=3)
        assert score.passes(min_score=3)
        assert not CaptionJudgeScore(
            style="formal", style_fit=3, accuracy=2, specificity=4
        ).passes(min_score=3)

    def test_skipped_fails(self):
        score = CaptionJudgeScore(
            style="formal",
            style_fit=5,
            accuracy=5,
            specificity=5,
            skipped=True,
            skip_reason="error",
        )
        assert not score.passes(min_score=3)

    def test_summary_and_failures(self):
        clip = ClipJudgeResult(
            task_id="e01",
            captions={
                "formal": CaptionJudgeScore(style="formal", style_fit=4, accuracy=4, specificity=4),
                "humorous_tech": CaptionJudgeScore(
                    style="humorous_tech",
                    style_fit=2,
                    accuracy=4,
                    specificity=4,
                    issue="no tech humor",
                ),
            },
            cross_style_distinctness=2,
            distinctness_note="too similar",
        )
        result = JudgeFileResult(
            clips=[clip],
            model="test-model",
            min_score=3,
            descriptions_provided=False,
        )
        assert result.passes == 1
        assert result.total == 2
        assert any("humorous_tech" in f for f in result.failures())
        assert result.low_distinctness()
        text = format_judge_summary(result)
        assert "judge: 1/2" in text or "1/2" in text
        exported = judge_result_to_dict(result)
        assert exported["passes"] == 1


class TestPanelAggregation:
    def test_median_aggregate_clip(self):
        judge_a = ClipJudgeResult(
            task_id="e01",
            captions={
                "formal": CaptionJudgeScore(style="formal", style_fit=4, accuracy=4, specificity=4),
                "humorous_tech": CaptionJudgeScore(
                    style="humorous_tech", style_fit=2, accuracy=4, specificity=4
                ),
            },
            cross_style_distinctness=4,
        )
        judge_b = ClipJudgeResult(
            task_id="e01",
            captions={
                "formal": CaptionJudgeScore(style="formal", style_fit=2, accuracy=3, specificity=3),
                "humorous_tech": CaptionJudgeScore(
                    style="humorous_tech", style_fit=3, accuracy=3, specificity=3, issue="weak joke"
                ),
            },
            cross_style_distinctness=2,
        )
        aggregated = aggregate_clip_judges({"a": judge_a, "b": judge_b})
        assert aggregated.captions["formal"].style_fit == 3
        assert aggregated.captions["formal"].accuracy == 4
        assert aggregated.captions["humorous_tech"].style_fit == 2
        assert aggregated.captions["humorous_tech"].accuracy == 4
        assert aggregated.cross_style_distinctness == 3

    def test_panel_summary_lists_per_judge(self):
        sub = JudgeFileResult(
            clips=[],
            model="accounts/fireworks/models/gpt-oss-120b",
            min_score=3,
            descriptions_provided=True,
        )
        result = JudgeFileResult(
            clips=[],
            model="panel(median): gpt-oss-120b, kimi-k2-6",
            min_score=3,
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
                "formal": CaptionJudgeScore(style="formal", style_fit=3, accuracy=3, specificity=3),
                "humorous_tech": CaptionJudgeScore(
                    style="humorous_tech", style_fit=5, accuracy=5, specificity=5
                ),
            },
        )
        result = JudgeFileResult(
            clips=[clip],
            model="test",
            min_score=3,
            descriptions_provided=False,
        )
        data = [{"task_id": "e01", "captions": {"formal": "A formal caption.", "humorous_tech": "Great joke."}}]
        samples = collect_calibration_samples(result, data, limit=5)
        assert len(samples) == 1
        assert samples[0]["style"] == "formal"
        report = format_calibration_report(samples)
        assert "e01/formal" in report
