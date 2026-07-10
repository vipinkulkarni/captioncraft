"""Tests for LLM-as-judge parsing and pass logic."""

import json

from src.llm_judge import (
    CaptionJudgeScore,
    ClipJudgeResult,
    JudgeFileResult,
    format_judge_summary,
    judge_result_to_dict,
    parse_judge_response,
    parse_style_judge_response,
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
        assert "judge: 1/2" in text
        exported = judge_result_to_dict(result)
        assert exported["passes"] == 1
