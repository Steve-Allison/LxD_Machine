"""RAGAS-style metric implementations.

Three metrics, all LLM-judged with minimal homegrown logic so the project
doesn't need to pull in the heavyweight ``ragas`` library (which would drag
in langchain + a host of conflicting Pydantic constraints).

Pattern across all three:

    1. Decompose the input via an LLM prompt (claims / questions /
       judgements).
    2. Score deterministically from the decomposition.
    3. Return a typed score; never raise — partial failures populate the
       ``error`` field so the run still produces a usable report.

Every LLM call is bounded by ``timeout_secs`` from the runtime config
plus the OpenAI client's own retry policy. Errors are swallowed into the
score's ``error`` field so a single bad question doesn't poison the run.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import structlog

from lxd.eval.models import (
    AnswerRelevanceScore,
    ClaimVerdict,
    ContextJudgement,
    ContextPrecisionScore,
    FaithfulnessScore,
)
from lxd.ingest.llm_client import call_openai_async

_log = structlog.get_logger(__name__)


_JSON_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

_CLAIM_EXTRACTION_PROMPT = """\
You decompose answers into atomic verifiable claims.

Given an ANSWER, return a JSON object with a single key ``claims`` whose value
is a list of strings. Each claim must be:
  - Atomic: one assertion per claim, no compound sentences.
  - Self-contained: understandable without reading the rest of the answer.
  - Factual: an assertion about the world, not a meta-comment ("the document
    discusses..." is NOT a claim about the world).
  - Verbatim-faithful: preserve the answer's wording where possible; do not
    paraphrase, summarise, or invent new content.

If the answer makes zero verifiable claims (e.g. "I don't know"), return
``{"claims": []}``.
"""

_CLAIM_VERIFY_PROMPT = """\
You judge whether a CLAIM is supported by a set of CONTEXT passages.

Rules:
  - "Supported" means at least one context passage states or strongly implies
    the claim. Paraphrases count if the meaning matches.
  - "Refuted" means at least one passage contradicts the claim.
  - If the claim is neither supported nor refuted, mark it NOT supported
    (default to skepticism — unsupported is the safer call).
  - Ignore prior knowledge. Judge ONLY against the supplied context.

Return JSON: ``{"supported": bool, "rationale": "one short sentence"}``.
"""


async def compute_faithfulness(
    *,
    answer: str,
    contexts: list[str],
    judge_model: str,
    timeout_secs: float = 60.0,
    api_key_env: str = "OPENAI_API_KEY",
) -> FaithfulnessScore:
    """Compute faithfulness = supported_claims / total_claims.

    Args:
        answer: The synthesised answer text to audit.
        contexts: Retrieved chunk texts the answer was supposed to be based on.
        judge_model: OpenAI chat model used for both decomposition and
            verification (typically ``gpt-4o-mini``).
        timeout_secs: Hard timeout per LLM call.
        api_key_env: Environment variable holding the OpenAI API key.

    Returns:
        ``FaithfulnessScore`` with ``score`` in [0, 1] or ``None`` on error.
        The ``verdicts`` list always carries the per-claim breakdown.
    """
    if not answer.strip() or not contexts:
        return FaithfulnessScore(score=None, error="empty answer or context")

    try:
        raw_claims = await call_openai_async(
            system_prompt=_CLAIM_EXTRACTION_PROMPT,
            user_prompt=f"ANSWER:\n{answer}",
            model=judge_model,
            temperature=0.0,
            timeout=timeout_secs,
            max_tokens=1500,
            response_format=_JSON_RESPONSE_FORMAT,
            api_key_env=api_key_env,
        )
    except Exception as exc:
        _log.warning("eval_faithfulness_claim_extraction_failed", error=str(exc))
        return FaithfulnessScore(score=None, error=f"claim extraction failed: {exc}")

    claims = _parse_claims(raw_claims)
    if not claims:
        return FaithfulnessScore(score=None, error="no claims extracted")

    joined_context = _join_contexts(contexts)
    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        verdict = await _verify_one_claim(
            claim=claim,
            joined_context=joined_context,
            judge_model=judge_model,
            timeout_secs=timeout_secs,
            api_key_env=api_key_env,
        )
        verdicts.append(verdict)

    supported = sum(1 for v in verdicts if v.supported)
    return FaithfulnessScore(
        score=supported / len(verdicts),
        verdicts=verdicts,
    )


async def _verify_one_claim(
    *,
    claim: str,
    joined_context: str,
    judge_model: str,
    timeout_secs: float,
    api_key_env: str,
) -> ClaimVerdict:
    """Return a single ClaimVerdict; never raises."""
    try:
        raw = await call_openai_async(
            system_prompt=_CLAIM_VERIFY_PROMPT,
            user_prompt=f"CLAIM: {claim}\n\nCONTEXT:\n{joined_context}",
            model=judge_model,
            temperature=0.0,
            timeout=timeout_secs,
            max_tokens=200,
            response_format=_JSON_RESPONSE_FORMAT,
            api_key_env=api_key_env,
        )
    except Exception as exc:
        _log.warning("eval_faithfulness_claim_verify_failed", error=str(exc), claim=claim)
        return ClaimVerdict(claim=claim, supported=False, rationale=f"verify failed: {exc}")

    payload = _safe_json(raw)
    supported = bool(payload.get("supported", False))
    rationale = str(payload.get("rationale", ""))[:300]
    return ClaimVerdict(claim=claim, supported=supported, rationale=rationale)


def _parse_claims(raw: str) -> list[str]:
    payload = _safe_json(raw)
    claims_raw = payload.get("claims", [])
    if not isinstance(claims_raw, list):
        return []
    out: list[str] = []
    for item in claims_raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _join_contexts(contexts: list[str]) -> str:
    pieces = [f"[CTX {i + 1}]\n{ctx.strip()}" for i, ctx in enumerate(contexts) if ctx.strip()]
    return "\n\n".join(pieces)


# ---------------------------------------------------------------------------
# Answer relevance
# ---------------------------------------------------------------------------

_QUESTION_GENERATION_PROMPT = """\
You generate questions that an ANSWER could plausibly be answering.

Given an ANSWER, return a JSON object with one key ``questions`` whose value
is a list of {N_QUESTIONS} short natural-language questions. Each question must be:
  - One sentence, ending with a question mark.
  - Self-contained: standalone, no pronouns referring to the answer.
  - Faithful: the answer must plausibly address it.
  - Distinct: questions should cover different angles where the answer is rich.

Return exactly {N_QUESTIONS} questions.
"""


async def compute_answer_relevance(
    *,
    question: str,
    answer: str,
    judge_model: str,
    embed_fn: EmbedFn,
    timeout_secs: float = 60.0,
    api_key_env: str = "OPENAI_API_KEY",
    n_generated: int = 3,
) -> AnswerRelevanceScore:
    """Compute answer relevance via reverse question generation.

    The intuition: a high-quality answer is one where, if you asked the model
    "what question would this answer?", it would generate questions close to
    the original.

    Args:
        question: The original user question.
        answer: The synthesised answer to score.
        judge_model: OpenAI chat model used to generate candidate questions.
        embed_fn: Sync function that turns a list of strings into vectors.
            Typically a thin wrapper over the project's embedder.
        timeout_secs: Hard timeout for the LLM call.
        api_key_env: Environment variable holding the OpenAI API key.
        n_generated: How many candidate questions to generate.

    Returns:
        ``AnswerRelevanceScore`` with ``score`` = mean cosine similarity
        between the original question and the generated ones, in [0, 1]
        (cosines below zero clipped to zero).
    """
    if not answer.strip():
        return AnswerRelevanceScore(score=None, error="empty answer")

    try:
        raw = await call_openai_async(
            system_prompt=_QUESTION_GENERATION_PROMPT.replace("{N_QUESTIONS}", str(n_generated)),
            user_prompt=f"ANSWER:\n{answer}",
            model=judge_model,
            temperature=0.0,
            timeout=timeout_secs,
            max_tokens=500,
            response_format=_JSON_RESPONSE_FORMAT,
            api_key_env=api_key_env,
        )
    except Exception as exc:
        _log.warning("eval_answer_relevance_generation_failed", error=str(exc))
        return AnswerRelevanceScore(score=None, error=f"question generation failed: {exc}")

    generated = _parse_questions(raw)
    if not generated:
        return AnswerRelevanceScore(score=None, error="no questions generated")

    try:
        vectors = embed_fn([question, *generated])
    except Exception as exc:
        _log.warning("eval_answer_relevance_embed_failed", error=str(exc))
        return AnswerRelevanceScore(
            score=None,
            generated_questions=generated,
            error=f"embedding failed: {exc}",
        )

    if len(vectors) != 1 + len(generated):
        return AnswerRelevanceScore(
            score=None,
            generated_questions=generated,
            error="embedding count mismatch",
        )

    q_vec = vectors[0]
    similarities = [max(0.0, _cosine(q_vec, gv)) for gv in vectors[1:]]
    score = sum(similarities) / len(similarities)
    return AnswerRelevanceScore(
        score=score,
        generated_questions=generated,
        similarities=similarities,
    )


def _parse_questions(raw: str) -> list[str]:
    payload = _safe_json(raw)
    items = payload.get("questions", [])
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return num / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Context precision
# ---------------------------------------------------------------------------

_CONTEXT_RELEVANCE_PROMPT = """\
You judge whether a CONTEXT passage is relevant to a QUESTION.

A passage is "relevant" if it contains information that would help answer the
question — even partially. Background, definitions, related concepts, and
direct answers all count as relevant. Unrelated text does NOT count, even if
it shares some surface vocabulary.

Return JSON: ``{"relevant": bool, "rationale": "one short sentence"}``.
"""


async def compute_context_precision(
    *,
    question: str,
    contexts: list[tuple[str, str]],
    judge_model: str,
    timeout_secs: float = 60.0,
    api_key_env: str = "OPENAI_API_KEY",
) -> ContextPrecisionScore:
    """Compute rank-weighted context precision.

    Args:
        question: The original user question.
        contexts: List of ``(citation_label, text)`` tuples in retrieval rank
            order. The first tuple is rank 1, the second is rank 2, etc.
        judge_model: OpenAI chat model used to judge each context.
        timeout_secs: Hard timeout per LLM call.
        api_key_env: Environment variable holding the OpenAI API key.

    Returns:
        ``ContextPrecisionScore`` where ``score`` is the rank-weighted mean
        of relevance (RAGAS-style): for each k, if rank-k is relevant,
        contribute precision@k to the running total; divide by the total
        relevant. Score is in [0, 1].
    """
    if not contexts:
        return ContextPrecisionScore(score=None, error="no contexts")

    judgements: list[ContextJudgement] = []
    for rank, (label, text) in enumerate(contexts, start=1):
        judgement = await _judge_one_context(
            question=question,
            label=label,
            text=text,
            rank=rank,
            judge_model=judge_model,
            timeout_secs=timeout_secs,
            api_key_env=api_key_env,
        )
        judgements.append(judgement)

    score = _rank_weighted_precision(judgements)
    return ContextPrecisionScore(score=score, judgements=judgements)


async def _judge_one_context(
    *,
    question: str,
    label: str,
    text: str,
    rank: int,
    judge_model: str,
    timeout_secs: float,
    api_key_env: str,
) -> ContextJudgement:
    """Return a single ContextJudgement; never raises."""
    try:
        raw = await call_openai_async(
            system_prompt=_CONTEXT_RELEVANCE_PROMPT,
            user_prompt=f"QUESTION: {question}\n\nCONTEXT:\n{text}",
            model=judge_model,
            temperature=0.0,
            timeout=timeout_secs,
            max_tokens=200,
            response_format=_JSON_RESPONSE_FORMAT,
            api_key_env=api_key_env,
        )
    except Exception as exc:
        _log.warning("eval_context_precision_judge_failed", error=str(exc), label=label)
        return ContextJudgement(
            citation_label=label,
            rank=rank,
            relevant=False,
            rationale=f"judge failed: {exc}",
        )

    payload = _safe_json(raw)
    relevant = bool(payload.get("relevant", False))
    rationale = str(payload.get("rationale", ""))[:300]
    return ContextJudgement(
        citation_label=label,
        rank=rank,
        relevant=relevant,
        rationale=rationale,
    )


def _rank_weighted_precision(judgements: list[ContextJudgement]) -> float:
    """RAGAS-style rank-weighted precision.

    For each rank k where judgement[k] is relevant, contribute precision@k
    to the running total. Divide by the total number of relevant items.
    """
    if not judgements:
        return 0.0
    total_relevant = sum(1 for j in judgements if j.relevant)
    if total_relevant == 0:
        return 0.0
    cumulative_relevant = 0
    running = 0.0
    for k, judgement in enumerate(judgements, start=1):
        if judgement.relevant:
            cumulative_relevant += 1
            precision_at_k = cumulative_relevant / k
            running += precision_at_k
    return running / total_relevant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_json(raw: str) -> dict[str, Any]:
    """Parse JSON; return empty dict on failure rather than raising."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError, TypeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


# Type alias for the embedding callable used by answer-relevance scoring.
type EmbedFn = Callable[[list[str]], list[list[float]]]
