"""Orchestrate the quality eval against the live answer pipeline.

For each golden question:

  1. Run the answer pipeline (``answer_question``) to get the synthesised
     answer + citations.
  2. Re-run ``search_chunks`` to get the typed retrieved chunks with text.
     The pipeline doesn't expose chunk text in the envelope, so this is the
     cheapest seam — search is cached and incremental, so the cost is small.
  3. Compute the three metrics.
  4. Build a typed ``PerQuestionResult``.

Aggregates into an :class:`EvalReport` and returns it. Persistence and
console rendering live in :mod:`lxd.eval.report`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from lxd.eval.metrics import (
    compute_answer_relevance,
    compute_context_precision,
    compute_faithfulness,
)
from lxd.eval.models import (
    EvalReport,
    GoldenQuestion,
    PerQuestionResult,
)
from lxd.eval.report import summarise_report
from lxd.ingest.embedder import embed_texts
from lxd.retrieval.query_pipeline import answer_question, search_chunks
from lxd.settings.models import RuntimeConfig

_log = structlog.get_logger(__name__)


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    """Load and validate a golden quality set from a JSON file.

    The file format is a JSON array of objects matching :class:`GoldenQuestion`.
    Pydantic validation catches malformed entries with a precise error.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(
            f"Golden set at {path} must be a JSON array; got {type(payload).__name__}."
        )
    return [GoldenQuestion.model_validate(item) for item in payload]


async def run_quality_eval(
    *,
    golden_set: list[GoldenQuestion],
    config: RuntimeConfig,
    judge_model: str = "gpt-4o-mini",
    judge_timeout_secs: float = 60.0,
    max_context_chunks: int = 8,
    api_key_env: str = "OPENAI_API_KEY",
) -> EvalReport:
    """Run RAGAS-style quality eval over the golden set.

    Args:
        golden_set: Questions + (optional) topic / source ground truth.
        config: Runtime configuration that drives retrieval + synthesis.
        judge_model: OpenAI chat model used for the LLM-judged metrics
            (faithfulness, answer relevance, context precision).
        judge_timeout_secs: Hard timeout per LLM call inside the metrics.
        max_context_chunks: Cap on how many top-ranked chunks feed the
            metrics — matches the synthesis ``max_chunks`` so the eval sees
            what the synthesiser saw.
        api_key_env: Environment variable holding the OpenAI API key.

    Returns:
        Typed :class:`EvalReport` ready to render or persist.
    """
    run_started_at = datetime.now(UTC).isoformat()

    per_question: list[PerQuestionResult] = []
    for question in golden_set:
        result = await _eval_one_question(
            question=question,
            config=config,
            judge_model=judge_model,
            judge_timeout_secs=judge_timeout_secs,
            max_context_chunks=max_context_chunks,
            api_key_env=api_key_env,
        )
        per_question.append(result)
        _log.info(
            "eval_question_done",
            question=question.question,
            faithfulness=result.faithfulness.score,
            answer_relevance=result.answer_relevance.score,
            context_precision=result.context_precision.score,
        )

    summary = summarise_report(per_question)
    run_finished_at = datetime.now(UTC).isoformat()

    return EvalReport(
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        judge_model=judge_model,
        embed_model=config.models.embed,
        summary=summary,
        per_question=per_question,
    )


async def _eval_one_question(
    *,
    question: GoldenQuestion,
    config: RuntimeConfig,
    judge_model: str,
    judge_timeout_secs: float,
    max_context_chunks: int,
    api_key_env: str,
) -> PerQuestionResult:
    """Run the pipeline + three metrics for a single question.

    A failure in any one metric only nulls that metric's score — the other
    two still run. A failure in the answer pipeline (rare; usually means the
    store is empty) short-circuits with empty metric scores.
    """
    envelope = answer_question(
        question=question.question,
        config=config,
        domain=question.domain,
    )

    outcome = search_chunks(
        question=question.question,
        config=config,
        domain=question.domain,
        limit=max_context_chunks,
    )
    top_chunks = outcome.ranked[:max_context_chunks]
    context_texts = [chunk.text for chunk in top_chunks]
    context_labels_and_texts = [(chunk.citation_label, chunk.text) for chunk in top_chunks]

    matched_entity_ids_raw = envelope.metadata.get("matched_entity_ids", [])
    if isinstance(matched_entity_ids_raw, list):
        matched_entity_ids = [str(e) for e in matched_entity_ids_raw if isinstance(e, str)]
    else:
        matched_entity_ids = []

    faithfulness = await compute_faithfulness(
        answer=envelope.answer_text,
        contexts=context_texts,
        judge_model=judge_model,
        timeout_secs=judge_timeout_secs,
        api_key_env=api_key_env,
    )

    answer_relevance = await compute_answer_relevance(
        question=question.question,
        answer=envelope.answer_text,
        judge_model=judge_model,
        embed_fn=lambda texts: embed_texts(config, texts),
        timeout_secs=judge_timeout_secs,
        api_key_env=api_key_env,
    )

    context_precision = await compute_context_precision(
        question=question.question,
        contexts=context_labels_and_texts,
        judge_model=judge_model,
        timeout_secs=judge_timeout_secs,
        api_key_env=api_key_env,
    )

    return PerQuestionResult(
        question=question.question,
        answer_text=envelope.answer_text,
        answer_status=envelope.answer_status.value,
        citations=envelope.citations,
        matched_entity_ids=matched_entity_ids,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        warnings=envelope.warnings,
    )
