"""Critique an existing design artefact bundle against fresh corpus evidence.

Runs one :func:`lxd.retrieval.query_pipeline.search_chunks` call against a
question derived from the caller's focus question (or the artefact's own
topic) and asks the design-agent LLM to score the artefact against that
evidence. Failure-safe throughout: a malformed artefact, an empty
retrieval, or an LLM failure all degrade to a :class:`CritiqueResult` with
``warnings`` set rather than raising, so a critique call never blocks a
caller from seeing partial feedback.
"""

import asyncio
import json
from typing import Any, Final

import structlog
from pydantic import ValidationError

from lxd.agents._retrieval import citation_labels, format_evidence_block
from lxd.agents.artefacts import CritiqueResult, DesignArtefactBundle
from lxd.ingest.llm_client import call_with_fallback_async
from lxd.retrieval.query_pipeline import search_chunks
from lxd.settings.models import RuntimeConfig

_log = structlog.get_logger(__name__)

_CRITIQUE_SYSTEM_PROMPT: Final = """\
You are a senior instructional designer critiquing a colleague's draft \
learning-design artefact bundle (objectives, modality plan, outline, \
assessment) against the ARTEFACT text and grounding EVIDENCE below.

Score these dimensions from 0.0 (poor) to 1.0 (excellent):
- objective_alignment: do the outline and assessment actually serve the
  stated objectives?
- evidence_grounding: is the artefact consistent with the EVIDENCE, or
  does it contradict / go well beyond what the evidence supports?
- assessment_validity: do the assessment items plausibly measure the
  stated objectives at the right Bloom's level?
- clarity: is the artefact specific and actionable rather than generic?

Give an overall_score (0.0-1.0, your holistic judgement — not necessarily
the mean of the dimensions) and 2-5 concise, actionable feedback bullets.

Return ONLY a JSON object shaped exactly like:
{"overall_score": 0.0, "dimension_scores": {"objective_alignment": 0.0, \
"evidence_grounding": 0.0, "assessment_validity": 0.0, "clarity": 0.0}, \
"feedback": ["..."]}
No preamble, no markdown fences."""

_CRITIQUE_USER_PROMPT: Final = """\
ARTEFACT:
{artefact}

EVIDENCE:
{evidence}

FOCUS QUESTION (optional, may be blank):
{focus_question}
"""


def critique_design(
    artefact_json_or_bundle: str | dict[str, Any] | DesignArtefactBundle,
    question_context: str,
    config: RuntimeConfig,
) -> CritiqueResult:
    """Score a design artefact bundle against fresh corpus evidence.

    Args:
        artefact_json_or_bundle: A :class:`DesignArtefactBundle`, a dict
            matching its shape, or a JSON string of either. Falls back to
            treating unparseable input as opaque free text — the critique
            is then based on that text alone, still grounded against
            fresh evidence.
        question_context: Optional focus question steering both the
            retrieval query and the critique's attention. Falls back to
            the artefact's own topic (or, failing that, the artefact text
            itself) when blank.
        config: Runtime configuration; reads ``config.design_agent``.

    Returns:
        A :class:`CritiqueResult`. Never raises — every failure mode
        degrades to a result with ``warnings`` set.
    """
    return asyncio.run(critique_design_async(artefact_json_or_bundle, question_context, config))


async def critique_design_async(
    artefact_json_or_bundle: str | dict[str, Any] | DesignArtefactBundle,
    question_context: str,
    config: RuntimeConfig,
) -> CritiqueResult:
    """Async body of :func:`critique_design`.

    Exposed separately so :mod:`lxd.agents.design` can await this
    directly instead of nesting a second ``asyncio.run`` call inside its
    own event loop (which ``asyncio.run`` forbids).
    """
    artefact_text, topic_hint = _render_artefact(artefact_json_or_bundle)
    if not artefact_text:
        return CritiqueResult(warnings=["Empty or unparseable artefact; nothing to critique."])

    query = question_context.strip() or topic_hint or artefact_text[:200]
    outcome = search_chunks(query, config, limit=config.design_agent.critique_retrieval_top_k)
    ranked = outcome.ranked
    evidence_block = format_evidence_block(ranked)
    citations = citation_labels(ranked)
    warnings = list(outcome.warnings)
    if not ranked:
        warnings.append("No corpus evidence retrieved for the critique query.")

    user_prompt = _CRITIQUE_USER_PROMPT.format(
        artefact=artefact_text,
        evidence=evidence_block,
        focus_question=question_context.strip() or "(none)",
    )
    cfg = config.design_agent
    try:
        raw = await call_with_fallback_async(
            system_prompt=_CRITIQUE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            primary_backend=cfg.backend,
            openai_model=cfg.openai_model,
            ollama_model=cfg.ollama_model,
            fallback_backend=cfg.fallback_backend,
            temperature=cfg.temperature,
            openai_timeout=cfg.timeout_secs,
            ollama_timeout=cfg.timeout_secs,
            max_tokens=cfg.max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        _log.warning("critique_llm_call_failed", error=str(exc))
        warnings.append(f"Critique LLM call failed: {exc}")
        return CritiqueResult(citations=citations, warnings=warnings)

    parsed = _safe_json(raw)
    if not parsed:
        warnings.append("Critique model returned unparseable output.")
        return CritiqueResult(citations=citations, warnings=warnings)

    overall_score = _coerce_score(parsed.get("overall_score")) or 0.0
    dimension_scores = _coerce_dimension_scores(parsed.get("dimension_scores"))
    feedback = _coerce_feedback(parsed.get("feedback"))
    if not feedback:
        warnings.append("Critique model returned no feedback bullets.")

    return CritiqueResult(
        overall_score=overall_score,
        dimension_scores=dimension_scores,
        feedback=feedback,
        citations=citations,
        warnings=warnings,
    )


def _render_artefact(
    artefact: str | dict[str, Any] | DesignArtefactBundle,
) -> tuple[str, str]:
    """Return ``(text_representation, topic_hint)`` for any accepted artefact shape."""
    if isinstance(artefact, DesignArtefactBundle):
        return artefact.model_dump_json(indent=2), artefact.topic
    if isinstance(artefact, dict):
        return _render_mapping(artefact)
    text = artefact.strip()
    if not text:
        return "", ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError, TypeError:
        return text, ""
    if isinstance(parsed, dict):
        return _render_mapping(parsed)
    return text, ""


def _render_mapping(payload: dict[str, Any]) -> tuple[str, str]:
    try:
        bundle = DesignArtefactBundle.model_validate(payload)
    except ValidationError:
        topic = payload.get("topic")
        return json.dumps(payload, indent=2), str(topic) if isinstance(topic, str) else ""
    return bundle.model_dump_json(indent=2), bundle.topic


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError, TypeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def _coerce_dimension_scores(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    scores: dict[str, float] = {}
    for key, raw_score in value.items():
        score = _coerce_score(raw_score)
        if score is not None:
            scores[str(key)] = score
    return scores


def _coerce_feedback(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
