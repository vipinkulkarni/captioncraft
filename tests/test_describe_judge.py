"""Unit tests for describe-vs-video judge parse and pick-best."""

from src.describe_judge import (
    DescribeJudgeScore,
    parse_describe_judge_response,
    pick_best_describe,
)
from src.results import DescribeError, DescribeResult


def test_parse_describe_judge_json():
    raw = '{"faithfulness":0.9,"coverage":0.8,"issue":""}'
    score = parse_describe_judge_response(raw, judge_model="m3")
    assert score.ok
    assert score.faithfulness == 0.9
    assert score.coverage == 0.8
    assert score.proxy == 0.85


def test_parse_describe_judge_rejects_out_of_range():
    raw = '{"faithfulness":1.5,"coverage":0.8,"issue":""}'
    score = parse_describe_judge_response(raw)
    assert not score.ok
    assert score.parse_error == "InvalidJSON"


def test_parse_describe_judge_regex_fallback():
    raw = 'noise "faithfulness": 0.7, "coverage": 0.6, "issue": "thin" more'
    score = parse_describe_judge_response(raw)
    assert score.ok
    assert score.faithfulness == 0.7
    assert score.coverage == 0.6
    assert score.issue == "thin"


def test_pick_prefers_higher_proxy():
    primary = DescribeResult(text="a", error=None)
    alternate = DescribeResult(text="b", error=None)
    p_score = DescribeJudgeScore(faithfulness=0.6, coverage=0.6)
    a_score = DescribeJudgeScore(faithfulness=0.95, coverage=0.9)
    winner, model, _, _ = pick_best_describe(
        primary=primary,
        primary_model="m3",
        alternate=alternate,
        alternate_model="qwen",
        primary_score=p_score,
        alternate_score=a_score,
    )
    assert winner is alternate
    assert model == "qwen"


def test_pick_falls_back_to_sole_ok():
    primary = DescribeResult(text=None, error=DescribeError.API, error_detail="fail")
    alternate = DescribeResult(text="ok", error=None)
    winner, model, _, _ = pick_best_describe(
        primary=primary,
        primary_model="m3",
        alternate=alternate,
        alternate_model="qwen",
        primary_score=None,
        alternate_score=None,
    )
    assert winner is alternate
    assert model == "qwen"


def test_pick_tie_prefers_primary():
    primary = DescribeResult(text="a", error=None)
    alternate = DescribeResult(text="b", error=None)
    score = DescribeJudgeScore(faithfulness=0.8, coverage=0.8)
    winner, model, _, _ = pick_best_describe(
        primary=primary,
        primary_model="m3",
        alternate=alternate,
        alternate_model="qwen",
        primary_score=score,
        alternate_score=DescribeJudgeScore(faithfulness=0.8, coverage=0.8),
    )
    assert winner is primary
    assert model == "m3"
