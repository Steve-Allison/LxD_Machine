"""Aggregate per-question results into a summary, format for humans, persist history."""

from __future__ import annotations

import json
from pathlib import Path

from lxd.eval.models import (
    EvalReport,
    EvalRunSummary,
    PerQuestionResult,
)


def summarise_report(per_question: list[PerQuestionResult]) -> EvalRunSummary:
    """Aggregate per-question scores into a single summary row.

    ``None`` scores (errors, no claims, embedding failures) are dropped from
    the mean — they don't pollute the headline number, but the count of
    answered-vs-questioned makes attrition visible.
    """
    question_count = len(per_question)
    faithfulness_values = [
        p.faithfulness.score for p in per_question if p.faithfulness.score is not None
    ]
    relevance_values = [
        p.answer_relevance.score for p in per_question if p.answer_relevance.score is not None
    ]
    precision_values = [
        p.context_precision.score for p in per_question if p.context_precision.score is not None
    ]

    mean_faithfulness = _safe_mean(faithfulness_values)
    mean_relevance = _safe_mean(relevance_values)
    mean_precision = _safe_mean(precision_values)
    composite = _harmonic_mean([mean_faithfulness, mean_relevance, mean_precision])
    answered_count = sum(1 for p in per_question if p.answer_status == "answered")

    return EvalRunSummary(
        question_count=question_count,
        answered_count=answered_count,
        mean_faithfulness=mean_faithfulness,
        mean_answer_relevance=mean_relevance,
        mean_context_precision=mean_precision,
        composite_score=composite,
    )


def format_console_report(report: EvalReport) -> str:
    """Render a tight human-readable summary of an EvalReport.

    Intent is one screenful: headline composite score, the three component
    scores, attrition counters, and a per-question table showing where
    failures happened.
    """
    lines: list[str] = []
    lines.append("== LxD quality eval ==")
    lines.append(f"  run:        {report.run_started_at} → {report.run_finished_at}")
    lines.append(f"  judge:      {report.judge_model}")
    lines.append(f"  embedder:   {report.embed_model}")
    lines.append("")

    s = report.summary
    lines.append(f"  questions:  {s.question_count} total, {s.answered_count} answered")
    lines.append(f"  composite:  {_fmt(s.composite_score)}  (harmonic mean)")
    lines.append(f"  faithfulness:      {_fmt(s.mean_faithfulness)}")
    lines.append(f"  answer relevance:  {_fmt(s.mean_answer_relevance)}")
    lines.append(f"  context precision: {_fmt(s.mean_context_precision)}")
    lines.append("")

    if report.per_question:
        lines.append("  per-question scores:")
        lines.append(f"    {'F':>5}  {'R':>5}  {'P':>5}  status              question")
        for p in report.per_question:
            f = _fmt_compact(p.faithfulness.score)
            r = _fmt_compact(p.answer_relevance.score)
            pc = _fmt_compact(p.context_precision.score)
            status = p.answer_status[:18].ljust(18)
            question = p.question if len(p.question) <= 80 else p.question[:77] + "..."
            lines.append(f"    {f:>5}  {r:>5}  {pc:>5}  {status}  {question}")
    return "\n".join(lines)


def append_run_to_history(report: EvalReport, history_path: Path) -> None:
    """Append one JSONL row per run, latest at EOF.

    The history file lives next to the data store (``data/openai/`` by
    default) and is gitignored. Each line is a complete
    :class:`EvalReport` so downstream tooling can reconstruct full detail.
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump_json()
    with history_path.open("a", encoding="utf-8") as f:
        f.write(payload)
        f.write("\n")


def write_report_json(report: EvalReport, path: Path) -> None:
    """Write a single eval report to a pretty-printed JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _harmonic_mean(values: list[float | None]) -> float | None:
    """Harmonic mean over non-None values; returns None if any is None or zero.

    The harmonic mean is dominated by the lowest input — if faithfulness is
    great but context precision is awful, the composite reflects the awful
    score. That's the right shape for a quality metric: you can't paper
    over a bad dimension by being great on the others.
    """
    if not values or any(v is None for v in values):
        return None
    floats = [v for v in values if v is not None]
    if any(v <= 0.0 for v in floats):
        return 0.0
    inv_sum = sum(1.0 / v for v in floats)
    return len(floats) / inv_sum


def _fmt(value: float | None) -> str:
    if value is None:
        return "  --"
    return f"{value:.3f}"


def _fmt_compact(value: float | None) -> str:
    if value is None:
        return "  --"
    return f"{value:.2f}"
