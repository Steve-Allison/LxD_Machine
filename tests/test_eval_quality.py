"""Unit tests for quality eval — math, aggregation, golden set loading.

LLM-driven metric functions are not exercised here (they would need a real
OpenAI client or heavy mocking). Tests focus on the deterministic logic
that's most likely to regress silently: rank-weighted precision math,
harmonic-mean aggregation, golden set validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lxd.eval import metrics as _metrics_module
from lxd.eval import report as _report_module
from lxd.eval.models import (
    AnswerRelevanceScore,
    ClaimVerdict,
    ContextJudgement,
    ContextPrecisionScore,
    EvalReport,
    EvalRunSummary,
    FaithfulnessScore,
    GoldenQuestion,
    PerQuestionResult,
)
from lxd.eval.report import append_run_to_history, summarise_report
from lxd.eval.runner import load_golden_set

_rank_weighted_precision = _metrics_module._rank_weighted_precision  # pyright: ignore[reportPrivateUsage]
_harmonic_mean = _report_module._harmonic_mean  # pyright: ignore[reportPrivateUsage]

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Rank-weighted precision math
# ---------------------------------------------------------------------------


def test_rank_weighted_precision_zero_when_nothing_relevant() -> None:
    judgements = [
        ContextJudgement(citation_label="a", rank=1, relevant=False),
        ContextJudgement(citation_label="b", rank=2, relevant=False),
    ]
    assert _rank_weighted_precision(judgements) == 0.0


def test_rank_weighted_precision_perfect_when_top_n_all_relevant() -> None:
    judgements = [
        ContextJudgement(citation_label="a", rank=1, relevant=True),
        ContextJudgement(citation_label="b", rank=2, relevant=True),
    ]
    # P@1 = 1/1, P@2 = 2/2, mean = 1.0
    assert _rank_weighted_precision(judgements) == pytest.approx(1.0)


def test_rank_weighted_precision_penalises_late_relevance() -> None:
    # Two relevant items but the first ones are noise.
    judgements = [
        ContextJudgement(citation_label="a", rank=1, relevant=False),
        ContextJudgement(citation_label="b", rank=2, relevant=False),
        ContextJudgement(citation_label="c", rank=3, relevant=True),
        ContextJudgement(citation_label="d", rank=4, relevant=True),
    ]
    # P@3 = 1/3 ≈ 0.333, P@4 = 2/4 = 0.5; mean over the two relevant items = (0.333 + 0.5)/2 = 0.4167
    expected = ((1 / 3) + (2 / 4)) / 2
    assert _rank_weighted_precision(judgements) == pytest.approx(expected)


def test_rank_weighted_precision_empty_returns_zero() -> None:
    assert _rank_weighted_precision([]) == 0.0


# ---------------------------------------------------------------------------
# Harmonic mean aggregation
# ---------------------------------------------------------------------------


def test_harmonic_mean_is_dominated_by_lowest_input() -> None:
    # Arithmetic mean of [0.1, 0.9, 0.9] is 0.633; harmonic should be much lower.
    arithmetic = (0.1 + 0.9 + 0.9) / 3
    harmonic = _harmonic_mean([0.1, 0.9, 0.9])
    assert harmonic is not None
    assert harmonic < arithmetic
    assert harmonic == pytest.approx(3 / (1 / 0.1 + 1 / 0.9 + 1 / 0.9))


def test_harmonic_mean_returns_none_when_any_value_missing() -> None:
    assert _harmonic_mean([0.9, None, 0.8]) is None


def test_harmonic_mean_returns_zero_when_any_value_zero() -> None:
    # Zero kills the harmonic mean — that's the desired shape.
    assert _harmonic_mean([0.9, 0.0, 0.8]) == 0.0


# ---------------------------------------------------------------------------
# summarise_report aggregation
# ---------------------------------------------------------------------------


def _result(
    *,
    question: str,
    faithfulness: float | None,
    relevance: float | None,
    precision: float | None,
    status: str = "answered",
) -> PerQuestionResult:
    return PerQuestionResult(
        question=question,
        answer_text="…",
        answer_status=status,
        citations=[],
        faithfulness=FaithfulnessScore(score=faithfulness),
        answer_relevance=AnswerRelevanceScore(score=relevance),
        context_precision=ContextPrecisionScore(score=precision),
    )


def test_summarise_report_drops_none_scores_from_mean() -> None:
    results = [
        _result(question="q1", faithfulness=1.0, relevance=0.9, precision=0.8),
        _result(question="q2", faithfulness=None, relevance=None, precision=None),
        _result(question="q3", faithfulness=0.5, relevance=0.7, precision=0.6),
    ]
    summary = summarise_report(results)
    # Means computed over the two non-None values only.
    assert summary.mean_faithfulness == pytest.approx((1.0 + 0.5) / 2)
    assert summary.mean_answer_relevance == pytest.approx((0.9 + 0.7) / 2)
    assert summary.mean_context_precision == pytest.approx((0.8 + 0.6) / 2)
    # Attrition is visible.
    assert summary.question_count == 3
    assert summary.answered_count == 3


def test_summarise_report_answered_count_reflects_status() -> None:
    results = [
        _result(question="q1", faithfulness=1.0, relevance=1.0, precision=1.0),
        _result(
            question="q2",
            faithfulness=None,
            relevance=None,
            precision=None,
            status="no_results",
        ),
    ]
    summary = summarise_report(results)
    assert summary.answered_count == 1


def test_summarise_report_composite_none_when_any_dimension_lacks_data() -> None:
    results = [
        _result(question="q1", faithfulness=None, relevance=0.9, precision=0.9),
    ]
    summary = summarise_report(results)
    assert summary.composite_score is None


# ---------------------------------------------------------------------------
# Golden set loading + validation
# ---------------------------------------------------------------------------


def test_load_golden_set_rejects_non_list_root(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"question": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON array"):
        load_golden_set(path)


def test_load_golden_set_rejects_empty_question(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps([{"question": "", "expected_answer_topics": [], "expected_source_files": []}]),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_golden_set(path)


def test_load_golden_set_round_trip(tmp_path: Path) -> None:
    payload = [
        {
            "question": "Q1",
            "expected_answer_topics": ["a", "b"],
            "expected_source_files": ["foo.md"],
            "domain": "theories",
        }
    ]
    path = tmp_path / "good.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    parsed = load_golden_set(path)
    assert len(parsed) == 1
    assert parsed[0].question == "Q1"
    assert parsed[0].domain == "theories"


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------


def test_append_run_to_history_creates_jsonl(tmp_path: Path) -> None:
    history = tmp_path / "history" / "runs.jsonl"
    report = EvalReport(
        run_started_at="2026-05-31T00:00:00+00:00",
        run_finished_at="2026-05-31T00:01:00+00:00",
        judge_model="gpt-4o-mini",
        embed_model="text-embedding-3-small",
        summary=EvalRunSummary(
            question_count=1,
            answered_count=1,
            mean_faithfulness=0.8,
            mean_answer_relevance=0.9,
            mean_context_precision=0.7,
            composite_score=0.79,
        ),
        per_question=[
            _result(question="Q1", faithfulness=0.8, relevance=0.9, precision=0.7),
        ],
    )
    append_run_to_history(report, history)
    append_run_to_history(report, history)
    lines = history.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    parsed_first = json.loads(lines[0])
    assert parsed_first["judge_model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Pydantic model invariants
# ---------------------------------------------------------------------------


def test_golden_question_is_frozen() -> None:
    q = GoldenQuestion(question="x")
    with pytest.raises(ValidationError):
        q.question = "y"  # type: ignore[misc]


def test_claim_verdict_rejects_extra_keys() -> None:
    with pytest.raises(ValidationError):
        ClaimVerdict.model_validate(
            {"claim": "x", "supported": True, "rationale": "ok", "extra_field": 1}
        )
