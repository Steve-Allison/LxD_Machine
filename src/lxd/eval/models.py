"""Typed data model for quality eval inputs, per-metric outputs, and reports."""

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Frozen, strict, extra-forbid base for every eval model."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)


class GoldenQuestion(_Frozen):
    """One entry in the quality eval golden set.

    ``expected_answer_topics`` are short phrases the answer should mention
    to be considered comprehensive. They are NOT used to compute scores
    directly — RAGAS-style metrics are evidence-driven — but they are
    written to the report so a human reviewer can spot-check coverage.
    """

    question: str = Field(
        description="Natural-language question to put to the answer pipeline.",
        min_length=1,
    )
    expected_answer_topics: list[str] = Field(
        default_factory=list,
        description=(
            "Short phrases the answer should plausibly mention. Human-review aid; "
            "does not enter the numeric score."
        ),
    )
    expected_source_files: list[str] = Field(
        default_factory=list,
        description=(
            "Optional ground-truth source files. Surfaced in the report for human "
            "review of context precision."
        ),
    )
    domain: str | None = Field(
        default=None,
        description="Optional domain filter passed to the answer pipeline.",
    )


class ClaimVerdict(_Frozen):
    """LLM verdict for one atomic claim extracted from the synthesised answer."""

    claim: str
    supported: bool
    rationale: str = Field(
        default="",
        description="Short reason the LLM gave for supported / refuted.",
    )


class FaithfulnessScore(_Frozen):
    """Faithfulness = supported_claims / total_claims.

    Measures whether the answer's assertions are grounded in retrieved
    context. 1.0 = every claim supported. 0.0 = none supported. ``None``
    indicates the metric could not be computed (e.g. no claims extracted,
    LLM failure).
    """

    score: float | None
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    error: str | None = None


class AnswerRelevanceScore(_Frozen):
    """Answer relevance via reverse-question generation + cosine similarity.

    Generate N questions the answer could plausibly be answering, then
    measure how close (cosine similarity in embedding space) each generated
    question is to the original. High = the answer addresses the question.
    """

    score: float | None
    generated_questions: list[str] = Field(default_factory=list)
    similarities: list[float] = Field(default_factory=list)
    error: str | None = None


class ContextJudgement(_Frozen):
    """Per-context judgement: was this chunk relevant to the question?"""

    citation_label: str
    rank: int = Field(description="1-indexed position in the retrieved set.")
    relevant: bool
    rationale: str = ""


class ContextPrecisionScore(_Frozen):
    """Context precision = rank-weighted precision over retrieved chunks.

    Heavier weight on early ranks: if a relevant chunk is at rank 1, that
    counts more than the same chunk at rank 10. Score is mean of
    ``relevant@k`` for k where the chunk at rank k is judged relevant.
    """

    score: float | None
    judgements: list[ContextJudgement] = Field(default_factory=list)
    error: str | None = None


class PerQuestionResult(_Frozen):
    """All metrics + raw answer for one golden question."""

    question: str
    answer_text: str
    answer_status: str
    citations: list[str]
    matched_entity_ids: list[str] = Field(default_factory=list)
    faithfulness: FaithfulnessScore
    answer_relevance: AnswerRelevanceScore
    context_precision: ContextPrecisionScore
    warnings: list[str] = Field(default_factory=list)


class EvalRunSummary(_Frozen):
    """Aggregate scores across the eval set, ignoring questions with errors."""

    question_count: int
    answered_count: int
    mean_faithfulness: float | None
    mean_answer_relevance: float | None
    mean_context_precision: float | None
    composite_score: float | None = Field(
        default=None,
        description="Harmonic mean of the three metrics — single headline number.",
    )


class EvalReport(_Frozen):
    """Full report: summary + per-question detail.

    Written as JSON for tooling and appended as one JSONL line per run
    to ``data/openai/eval_quality_runs.jsonl`` so historical scores can
    be compared.
    """

    run_started_at: str = Field(description="ISO-8601 UTC timestamp the run started.")
    run_finished_at: str
    judge_model: str = Field(description="LLM used as judge for faithfulness / context precision.")
    embed_model: str = Field(description="Embedding model used for answer-relevance similarity.")
    summary: EvalRunSummary
    per_question: list[PerQuestionResult]
