"""Continuous quality eval for the LxD answer pipeline.

Exposes RAGAS-style metrics — faithfulness, answer relevance, context
precision — that judge end-to-end answer quality rather than the
retrieval-only recall/MRR scores in :mod:`lxd.retrieval.eval`.

Public entry point is :func:`run_quality_eval` plus the dataclasses in
:mod:`lxd.eval.models`. The CLI lives at ``lxd.cli.eval_quality``.
"""

from __future__ import annotations

from lxd.eval.metrics import (
    compute_answer_relevance,
    compute_context_precision,
    compute_faithfulness,
)
from lxd.eval.models import (
    AnswerRelevanceScore,
    ContextJudgement,
    ContextPrecisionScore,
    EvalReport,
    EvalRunSummary,
    FaithfulnessScore,
    GoldenQuestion,
    PerQuestionResult,
)
from lxd.eval.report import (
    append_run_to_history,
    format_console_report,
    summarise_report,
)
from lxd.eval.runner import load_golden_set, run_quality_eval

__all__ = [
    "AnswerRelevanceScore",
    "ContextJudgement",
    "ContextPrecisionScore",
    "EvalReport",
    "EvalRunSummary",
    "FaithfulnessScore",
    "GoldenQuestion",
    "PerQuestionResult",
    "append_run_to_history",
    "compute_answer_relevance",
    "compute_context_precision",
    "compute_faithfulness",
    "format_console_report",
    "load_golden_set",
    "run_quality_eval",
    "summarise_report",
]
