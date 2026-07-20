"""Multi-step design-artefact agent (Phase 3b/c SOTA roadmap).

``design_learning`` runs a bounded step machine — clarify the brief,
retrieve pedagogy evidence, draft objectives / modality plan / outline /
assessment, then one critique+revise pass — hard-capped at
``config.design_agent.max_steps`` (default 6, one per named step after
clarify: retrieve, objectives, modality, outline, assessment, critique).

Two circuit breakers keep a bad run cheap and predictable:

- **Retrieval**: two consecutive empty ``search_chunks`` calls (original
  query, then one broadened query) stop the whole run and return an
  ungrounded skeleton bundle with a warning, rather than drafting
  artefacts against nothing.
- **Max steps**: each drafting step checks the remaining step budget
  before running; once exhausted, ``design_learning`` returns whatever
  artefacts it already drafted, flagged in ``warnings``.

Every LLM call is individually try/excepted — a single step's failure
degrades that step to an empty artefact plus a warning, it never aborts
the whole bundle.
"""

import asyncio
import json
from typing import Any, Final

import structlog

from lxd.agents._retrieval import citation_labels, format_evidence_block
from lxd.agents.artefacts import (
    AssessmentBlueprint,
    DesignArtefactBundle,
    LearningObjectives,
    ModalityPlan,
    Outline,
)
from lxd.agents.critique import critique_design_async
from lxd.domain.brief import LearnerBrief
from lxd.ingest.llm_client import call_with_fallback_async
from lxd.retrieval.query_pipeline import RankedChunk, search_chunks
from lxd.settings.models import RuntimeConfig

_log = structlog.get_logger(__name__)

# Steps counted against ``max_steps``: retrieve, objectives, modality,
# outline, assessment, critique+revise. "Clarify" (merging the brief into
# a working query) is free bookkeeping, not a counted step.
_COUNTED_STEPS: Final = ("retrieve", "objectives", "modality", "outline", "assessment", "critique")

_OBJECTIVES_SYSTEM_PROMPT: Final = """\
You are an instructional designer drafting Bloom's-taxonomy-aligned \
learning objectives for the TOPIC below, grounded in the EVIDENCE \
provided and tailored to the LEARNER BRIEF. Each objective must be a \
single, measurable, observable statement (one verb, one outcome).

Return ONLY a JSON object: {"objectives": ["...", "..."]} with 3-6 \
objectives. No preamble, no markdown fences."""

_MODALITY_SYSTEM_PROMPT: Final = """\
You are an instructional designer recommending a delivery modality for \
the TOPIC below, grounded in the EVIDENCE provided and the LEARNER \
BRIEF's stated modality preference and constraints (if any).

Return ONLY a JSON object: {"modality_plan": "2-4 sentence recommendation \
with rationale"}. No preamble, no markdown fences."""

_OUTLINE_SYSTEM_PROMPT: Final = """\
You are an instructional designer sequencing a learning experience for \
the TOPIC below, grounded in the EVIDENCE provided, in service of the \
OBJECTIVES already drafted and the LEARNER BRIEF.

Return ONLY a JSON object: {"outline": ["Module 1: ...", "Module 2: ..."]} \
with 3-8 ordered items. No preamble, no markdown fences."""

_ASSESSMENT_SYSTEM_PROMPT: Final = """\
You are an instructional designer drafting an assessment blueprint for \
the TOPIC below, grounded in the EVIDENCE provided, mapped to the \
OBJECTIVES already drafted, and appropriate for the LEARNER BRIEF.

Return ONLY a JSON object: {"assessment": ["Item 1: ...", "Item 2: ..."]} \
with 2-6 items, each naming the objective it measures. No preamble, no \
markdown fences."""


def design_learning(
    topic: str,
    brief: LearnerBrief,
    config: RuntimeConfig,
    *,
    max_steps: int | None = None,
) -> DesignArtefactBundle:
    """Draft a grounded learning-design artefact bundle for ``topic``.

    Args:
        topic: The learning goal / topic to design for.
        brief: Optional audience/modality/Bloom-target/constraints
            context (see :class:`lxd.domain.brief.LearnerBrief`).
        config: Runtime configuration; reads ``config.design_agent``.
        max_steps: Overrides ``config.design_agent.max_steps`` when set.

    Returns:
        A :class:`DesignArtefactBundle`. Always returns — every failure
        mode (empty retrieval, LLM failure, step-budget exhaustion)
        degrades to a partial bundle with ``warnings`` set rather than
        raising.
    """
    return asyncio.run(_design_learning_async(topic, brief, config, max_steps=max_steps))


async def _design_learning_async(
    topic: str,
    brief: LearnerBrief,
    config: RuntimeConfig,
    *,
    max_steps: int | None,
) -> DesignArtefactBundle:
    cap = max_steps if max_steps is not None else config.design_agent.max_steps
    log = _log.bind(topic=topic, session_id=brief.session_id, max_steps=cap)
    warnings: list[str] = []
    steps_completed = 0

    # "Clarify" — fold the brief into the retrieval query. No step budget
    # spent; this is bookkeeping, not an agent action.
    query = _clarify_query(topic, brief)
    brief_summary = _summarize_brief(brief)

    if steps_completed >= cap:
        return _empty_bundle(topic, steps_completed, [*warnings, "max_steps=0; no steps ran."])

    ranked, retrieval_warnings, circuit_tripped = _retrieve_pedagogy_evidence(query, config)
    steps_completed += 1
    warnings.extend(retrieval_warnings)
    if circuit_tripped:
        log.warning("design_agent_retrieval_circuit_tripped")
        return _empty_bundle(topic, steps_completed, warnings)

    evidence_block = format_evidence_block(ranked)
    citations = citation_labels(ranked)

    objectives = LearningObjectives(citations=citations)
    if steps_completed < cap:
        items, step_warnings = await _draft_json_list(
            system_prompt=_OBJECTIVES_SYSTEM_PROMPT,
            user_prompt=(
                f"TOPIC:\n{topic}\n\nLEARNER BRIEF:\n{brief_summary}\n\nEVIDENCE:\n{evidence_block}"
            ),
            config=config,
            json_key="objectives",
        )
        objectives = LearningObjectives(items=items, citations=citations if items else [])
        warnings.extend(step_warnings)
        steps_completed += 1
    else:
        warnings.append("max_steps reached before drafting objectives.")

    modality_plan = ModalityPlan(citations=citations)
    if steps_completed < cap:
        text, step_warnings = await _draft_json_text(
            system_prompt=_MODALITY_SYSTEM_PROMPT,
            user_prompt=(
                f"TOPIC:\n{topic}\n\nLEARNER BRIEF:\n{brief_summary}\n\nEVIDENCE:\n{evidence_block}"
            ),
            config=config,
            json_key="modality_plan",
        )
        modality_plan = ModalityPlan(text=text, citations=citations if text else [])
        warnings.extend(step_warnings)
        steps_completed += 1
    else:
        warnings.append("max_steps reached before drafting the modality plan.")

    outline = Outline(citations=citations)
    if steps_completed < cap:
        items, step_warnings = await _draft_json_list(
            system_prompt=_OUTLINE_SYSTEM_PROMPT,
            user_prompt=(
                f"TOPIC:\n{topic}\n\nOBJECTIVES:\n{_bullet(objectives.items)}\n\n"
                f"LEARNER BRIEF:\n{brief_summary}\n\nEVIDENCE:\n{evidence_block}"
            ),
            config=config,
            json_key="outline",
        )
        outline = Outline(items=items, citations=citations if items else [])
        warnings.extend(step_warnings)
        steps_completed += 1
    else:
        warnings.append("max_steps reached before drafting the outline.")

    assessment = AssessmentBlueprint(citations=citations)
    if steps_completed < cap:
        items, step_warnings = await _draft_json_list(
            system_prompt=_ASSESSMENT_SYSTEM_PROMPT,
            user_prompt=(
                f"TOPIC:\n{topic}\n\nOBJECTIVES:\n{_bullet(objectives.items)}\n\n"
                f"LEARNER BRIEF:\n{brief_summary}\n\nEVIDENCE:\n{evidence_block}"
            ),
            config=config,
            json_key="assessment",
        )
        assessment = AssessmentBlueprint(items=items, citations=citations if items else [])
        warnings.extend(step_warnings)
        steps_completed += 1
    else:
        warnings.append("max_steps reached before drafting the assessment blueprint.")

    bundle = DesignArtefactBundle(
        topic=topic,
        objectives=objectives,
        modality_plan=modality_plan,
        outline=outline,
        assessment=assessment,
        steps_completed=steps_completed,
        warnings=warnings,
    )

    if steps_completed < cap:
        bundle = await _critique_and_revise(bundle, query, config)
        steps_completed += 1
        bundle = bundle.model_copy(update={"steps_completed": steps_completed})
    else:
        bundle = bundle.model_copy(
            update={"warnings": [*bundle.warnings, "max_steps reached before critique+revise."]}
        )

    log.info(
        "design_learning_complete",
        steps_completed=steps_completed,
        warning_count=len(bundle.warnings),
    )
    return bundle


def _clarify_query(topic: str, brief: LearnerBrief) -> str:
    """Fold brief fields into one retrieval query for the pedagogy-evidence step."""
    parts = [f"instructional design pedagogy for teaching {topic.strip()}"]
    if brief.modality:
        parts.append(f"in a {brief.modality} format")
    if brief.bloom_target:
        parts.append(f"targeting {brief.bloom_target}-level understanding")
    return " ".join(parts)


def _summarize_brief(brief: LearnerBrief) -> str:
    lines = []
    if brief.audience:
        lines.append(f"Audience: {brief.audience}")
    if brief.modality:
        lines.append(f"Modality preference: {brief.modality}")
    if brief.bloom_target:
        lines.append(f"Bloom's target: {brief.bloom_target}")
    if brief.constraints:
        lines.append(f"Constraints: {brief.constraints}")
    return "\n".join(lines) if lines else "(no learner brief provided)"


def _bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "(none drafted yet)"


def _retrieve_pedagogy_evidence(
    query: str, config: RuntimeConfig
) -> tuple[list[RankedChunk], list[str], bool]:
    """Retrieve pedagogy evidence with one broadened retry on an empty first hit.

    Returns ``(ranked, warnings, circuit_tripped)``. ``circuit_tripped`` is
    ``True`` only when every attempt (up to
    ``config.design_agent.max_empty_retrievals``) returned zero chunks —
    the caller should stop the whole run in that case.
    """
    attempts = max(config.design_agent.max_empty_retrievals, 1)
    variants = _query_variants(query, attempts)
    warnings: list[str] = []
    for variant in variants:
        outcome = search_chunks(variant, config, limit=config.design_agent.retrieval_top_k)
        warnings.extend(outcome.warnings)
        if outcome.ranked:
            return outcome.ranked, warnings, False
    warnings.append(
        f"No pedagogy evidence retrieved after {len(variants)} attempt(s); "
        "returning an ungrounded skeleton bundle."
    )
    return [], warnings, True


def _query_variants(query: str, count: int) -> list[str]:
    candidates = [query, f"{query} best practices overview"]
    return candidates[: max(count, 1)] or candidates[:1]


async def _draft_json_list(
    *, system_prompt: str, user_prompt: str, config: RuntimeConfig, json_key: str
) -> tuple[list[str], list[str]]:
    raw, call_warnings = await _call_design_llm(system_prompt, user_prompt, config, json_key)
    if raw is None:
        return [], call_warnings
    parsed = _safe_json(raw)
    items = parsed.get(json_key)
    if not isinstance(items, list):
        return [], [*call_warnings, f"{json_key} model returned unparseable output."]
    cleaned = [str(item).strip() for item in items if isinstance(item, str) and str(item).strip()]
    if not cleaned:
        return [], [*call_warnings, f"{json_key} model returned no usable items."]
    return cleaned, call_warnings


async def _draft_json_text(
    *, system_prompt: str, user_prompt: str, config: RuntimeConfig, json_key: str
) -> tuple[str, list[str]]:
    raw, call_warnings = await _call_design_llm(system_prompt, user_prompt, config, json_key)
    if raw is None:
        return "", call_warnings
    parsed = _safe_json(raw)
    value = parsed.get(json_key)
    if not isinstance(value, str) or not value.strip():
        return "", [*call_warnings, f"{json_key} model returned unparseable output."]
    return value.strip(), call_warnings


async def _call_design_llm(
    system_prompt: str, user_prompt: str, config: RuntimeConfig, step_label: str
) -> tuple[str | None, list[str]]:
    cfg = config.design_agent
    try:
        raw = await call_with_fallback_async(
            system_prompt=system_prompt,
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
        _log.warning("design_step_llm_call_failed", step=step_label, error=str(exc))
        return None, [f"{step_label} drafting failed: {exc}"]
    if not raw:
        return None, [f"{step_label} model returned an empty response."]
    return raw, []


async def _critique_and_revise(
    bundle: DesignArtefactBundle, query: str, config: RuntimeConfig
) -> DesignArtefactBundle:
    """Run one critique pass and fold low-scoring feedback into ``bundle.warnings``.

    This is a critique+*surface* pass rather than a full re-drafting loop:
    the critique's score and feedback are attached to the bundle so
    callers see exactly what a reviewer would flag, without spending a
    second full drafting round-trip inside the same bounded run.
    """
    critique = await critique_design_async(bundle, query, config)
    revision_notes = [f"critique_overall_score={critique.overall_score:.2f}", *critique.feedback]
    if critique.warnings:
        revision_notes.extend(f"critique_warning: {w}" for w in critique.warnings)
    return bundle.model_copy(update={"warnings": [*bundle.warnings, *revision_notes]})


def _empty_bundle(topic: str, steps_completed: int, warnings: list[str]) -> DesignArtefactBundle:
    return DesignArtefactBundle(topic=topic, steps_completed=steps_completed, warnings=warnings)


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError, TypeError:
        return {}
    return payload if isinstance(payload, dict) else {}
