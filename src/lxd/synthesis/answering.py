"""Generate final answer envelopes from ranked evidence chunks via local MLX.

Two modes:

- :func:`synthesize_answer` — one-shot: blocks until the local MLX
  model finishes, returns a single :class:`AnswerEnvelope`.
- :func:`stream_synthesize_answer` — iterator: yields incremental
  :class:`StreamingTextDelta` events as the model emits text, then
  exactly one terminal :class:`AnswerEnvelope` once the stream
  completes (or sooner on error). Both stay on the local MLX
  backend; remote synthesis was rejected as incompatible with the
  local-first design (see ``feedback_local_only_no_remote_rerank.md``
  and the SOTA-plan strike of item ``[#4]``).
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

from lxd.domain.brief import LearnerBrief
from lxd.domain.json_utils import strip_thinking
from lxd.domain.status import QueryAnswerStatus
from lxd.mlx.chat import ChatError, complete_chat, stream_chat
from lxd.retrieval.graph_routing import GraphContext
from lxd.settings.models import RuntimeConfig
from lxd.synthesis.citation_alignment import SentenceCitation, align_citations
from lxd.synthesis.sampler import Sampler, SamplerFailure, SamplerRequest

# Public, single source of truth for the synthesis preamble. Extracted so
# the MCP Prompt resource (`mcp/prompts.py`) can surface the exact text
# that gets prepended to every synthesis prompt without duplication —
# clients auditing the system see the same instructions the model sees.
SYNTHESIS_PREAMBLE_BASE: Final = (
    "Answer the question using only the evidence below.\n"
    "If the evidence is insufficient, say so plainly.\n"
    "Do not invent facts.\n"
    "\n"
    "Cite every assertion inline using ``[citation_label]`` markers placed\n"
    "immediately after the sentence they support. Use the exact citation\n"
    "label string from the evidence header — do not paraphrase, abbreviate,\n"
    "or invent labels. A sentence may carry multiple markers (e.g.\n"
    '"X is Y [chunk_a] [chunk_b]."). A sentence with no citable evidence\n'
    "MUST end with no marker — never fabricate one to satisfy the format.\n"
)
SYNTHESIS_PREAMBLE_TRANSITIVE_SOURCES: Final = (
    "\nEvidence chunks may include a ``Sources:`` line listing the underlying\n"
    "research files the chunk was synthesised from. When such sources are\n"
    "present, prefer citing assertions that trace back to authoritative\n"
    "sources, and you may mention the source file name in prose when helpful.\n"
)
SYNTHESIS_PREAMBLE_GRAPH_CONTEXT: Final = (
    "\nA structured knowledge-graph context is provided above the evidence\n"
    "chunks. Use the entity definitions, community reports, and extracted\n"
    "claims to frame your answer, understand domain relationships, and ensure\n"
    "accurate terminology. If graph context conflicts with evidence chunks,\n"
    "ground your answer in the evidence chunks.\n"
)


def synthesis_preamble(
    *,
    has_transitive_sources: bool = False,
    has_graph_context: bool = False,
) -> str:
    """Return the synthesis preamble, conditionally including contextual guidance."""
    parts = [SYNTHESIS_PREAMBLE_BASE]
    if has_transitive_sources:
        parts.append(SYNTHESIS_PREAMBLE_TRANSITIVE_SOURCES)
    if has_graph_context:
        parts.append(SYNTHESIS_PREAMBLE_GRAPH_CONTEXT)
    return "".join(parts)


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    """Ranked evidence chunk passed to the synthesis stage."""

    citation_label: str
    text: str
    score: float
    chunk_id: str | None = None
    document_id: str | None = None
    cited_sources: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AnswerEnvelope:
    """Envelope returned by answer synthesis with metadata and citations."""

    answer_status: QueryAnswerStatus
    answer_text: str
    citations: list[str]
    warnings: list[str]
    metadata: dict[str, object]
    sentence_citations: list[SentenceCitation] = field(default_factory=list)
    graph_context: GraphContext | None = None


@dataclass(frozen=True, slots=True)
class StreamingTextDelta:
    """One text chunk emitted by :func:`stream_synthesize_answer`."""

    text: str


def format_graph_context_for_prompt(context: GraphContext) -> str:
    """Format structured GraphContext into a prompt section for the LLM."""
    if context.is_empty():
        return ""

    sections: list[str] = []

    if context.conflicts:
        sections.append("## Identified Domain Nuances & Disagreements")
        for c in context.conflicts:
            ent_a = c.subject_entity_id or "Concept A"
            ent_b = c.object_entity_id or "Concept B"
            sections.append(
                f"- Between [{ent_a}] and [{ent_b}]:\n  {c.claim_a_text}\n  VS.\n  {c.claim_b_text}"
            )
        sections.append("")

    if context.entity_profiles:
        sections.append("## Relevant Entity Context")
        for ep in context.entity_profiles:
            summary = ep.llm_summary or ep.deterministic_summary
            sections.append(f"### {ep.label} ({ep.entity_type}, {ep.domain})\n{summary}")
        sections.append("")

    if context.community_reports:
        sections.append("## Broader Context (Thematic Communities)")
        for cr in context.community_reports:
            summary = cr.llm_summary or cr.deterministic_summary
            sections.append(
                f"### Community {cr.community_id} (Level {cr.community_level})\n{summary}"
            )
        sections.append("")

    if context.claims:
        sections.append("## Key Domain Claims")
        for claim in context.claims:
            sections.append(f"- [{claim.claim_type.upper()}] {claim.claim_text}")
        sections.append("")

    return "\n".join(sections)


def synthesize_answer(
    question: str,
    evidence: list[EvidenceChunk],
    config: RuntimeConfig,
    *,
    graph_context_prompt: str = "",
    brief: LearnerBrief | None = None,
    sampler: Sampler | None = None,
) -> AnswerEnvelope:
    """Synthesize an answer using retrieved evidence and local MLX or client sampling.

    Args:
        question: User query.
        evidence: Ranked evidence chunks.
        config: Runtime configuration object.
        graph_context_prompt: Pre-formatted graph context string.
        brief: Optional learner brief constraining audience, modality,
            and Bloom target. Framing instructions are appended to the
            prompt without loosening the evidence-grounding
            requirement.
        sampler: Optional client-side sampler. When provided (the MCP
            server passes one when ``mcp.synthesis_backend`` is set to
            ``client_sampling``), the synthesis prompt is dispatched via
            :func:`fastmcp.Context.sample` so the connected client's LLM
            answers. On :class:`SamplerFailure` (client did not advertise
            sampling capability, upstream error, empty response) the call
            transparently falls back to the server-side MLX path and
            surfaces the reason as a warning on the resulting envelope.

    Returns:
        Answer envelope from synthesis or fallback.
    """
    citations = [chunk.citation_label for chunk in evidence]
    prompt = _build_prompt(
        question, evidence, graph_context_prompt=graph_context_prompt, brief=brief
    )
    if sampler is not None:
        request = SamplerRequest(
            prompt=prompt,
            temperature=config.synthesis.temperature,
            max_tokens=config.synthesis.max_tokens,
        )
        try:
            answer_text = _strip_thinking(sampler(request)).strip()
        except SamplerFailure as exc:
            fallback = _synthesize_answer_local(prompt=prompt, config=config, citations=citations)
            warning = f"Client sampling unavailable ({exc}); answered via server LLM instead."
            return replace(fallback, warnings=[*fallback.warnings, warning])
        if not answer_text:
            fallback = _synthesize_answer_local(prompt=prompt, config=config, citations=citations)
            warning = "Client sampling returned an empty response; answered via server LLM instead."
            return replace(fallback, warnings=[*fallback.warnings, warning])
        return AnswerEnvelope(
            answer_status=QueryAnswerStatus.ANSWERED,
            answer_text=answer_text,
            citations=citations,
            warnings=[],
            metadata={},
            sentence_citations=align_citations(answer_text=answer_text, valid_labels=citations),
        )
    return _synthesize_answer_local(prompt=prompt, config=config, citations=citations)


def _synthesize_answer_local(
    *,
    prompt: str,
    config: RuntimeConfig,
    citations: list[str],
) -> AnswerEnvelope:
    """Run synthesis on the configured local MLX model."""
    try:
        raw = complete_chat(
            config,
            prompt,
            temperature=config.synthesis.temperature,
            max_tokens=config.synthesis.max_tokens,
            timeout_secs=float(config.synthesis.timeout_secs),
        )
    except ChatError as exc:
        return synthesis_unavailable_answer(citations, f"Synthesis model unavailable: {exc}")
    answer_text = _strip_thinking(raw).strip()
    if not answer_text:
        return synthesis_unavailable_answer(
            citations,
            "Synthesis model returned an empty response.",
        )
    return AnswerEnvelope(
        answer_status=QueryAnswerStatus.ANSWERED,
        answer_text=answer_text,
        citations=citations,
        warnings=[],
        metadata={},
        sentence_citations=align_citations(answer_text=answer_text, valid_labels=citations),
    )


def stream_synthesize_answer(
    question: str,
    evidence: list[EvidenceChunk],
    config: RuntimeConfig,
    *,
    graph_context_prompt: str = "",
    brief: LearnerBrief | None = None,
) -> Iterator[StreamingTextDelta | AnswerEnvelope]:
    """Stream a synthesis answer from the local MLX model.

    Yields zero or more :class:`StreamingTextDelta` events as the model
    emits text, then exactly one terminal :class:`AnswerEnvelope` (the
    same object :func:`synthesize_answer` would return). Consumers that
    don't drain the iterator to completion miss the envelope; consumers
    that abandon mid-stream simply lose the answer status without
    leaking server resources beyond the request lifetime.

    Errors during the stream short-circuit: the iterator yields a
    single :class:`AnswerEnvelope` with
    ``QueryAnswerStatus.SYNTHESIS_UNAVAILABLE`` and stops.
    """
    citations = [chunk.citation_label for chunk in evidence]
    prompt = _build_prompt(
        question, evidence, graph_context_prompt=graph_context_prompt, brief=brief
    )
    try:
        stream = stream_chat(
            config,
            prompt,
            temperature=config.synthesis.temperature,
            max_tokens=config.synthesis.max_tokens,
            timeout_secs=float(config.synthesis.timeout_secs),
        )
    except ChatError as exc:
        yield synthesis_unavailable_answer(citations, f"Synthesis model unavailable: {exc}")
        return
    accumulated: list[str] = []
    try:
        for text in stream:
            accumulated.append(text)
            yield StreamingTextDelta(text=text)
    except ChatError as exc:
        yield synthesis_unavailable_answer(citations, f"Synthesis stream interrupted: {exc}")
        return
    full_text = _strip_thinking("".join(accumulated)).strip()
    if not full_text:
        yield synthesis_unavailable_answer(
            citations,
            "Synthesis model returned an empty response.",
        )
        return
    yield AnswerEnvelope(
        answer_status=QueryAnswerStatus.ANSWERED,
        answer_text=full_text,
        citations=citations,
        warnings=[],
        metadata={"streamed": True},
        sentence_citations=align_citations(answer_text=full_text, valid_labels=citations),
    )


def synthesis_unavailable_answer(citations: list[str], warning: str) -> AnswerEnvelope:
    """Build an answer envelope for synthesis failures.

    Args:
        citations: Citation labels to include in the envelope.
        warning: Warning message to return with fallback answers.

    Returns:
        Answer envelope for synthesis-unavailable status.
    """
    return AnswerEnvelope(
        answer_status=QueryAnswerStatus.SYNTHESIS_UNAVAILABLE,
        answer_text="Evidence was retrieved, but the configured synthesis model is unavailable.",
        citations=citations,
        warnings=[warning],
        metadata={},
    )


def probe_synthesis_model(config: RuntimeConfig) -> tuple[bool, str | None]:
    """Probe MLX backend availability and return probe metadata.

    Args:
        config: Runtime configuration object.

    Returns:
        Tuple of `(supported, warning)` for synthesis backend.
    """
    try:
        raw = complete_chat(
            config,
            "Reply with exactly OK.",
            temperature=0.0,
            max_tokens=16,
            timeout_secs=10.0,
        )
    except ChatError as exc:
        return False, str(exc)
    if not _strip_thinking(raw).strip():
        return False, "Synthesis probe returned an empty response."
    return True, None


def _build_prompt(
    question: str,
    evidence: list[EvidenceChunk],
    *,
    graph_context_prompt: str = "",
    brief: LearnerBrief | None = None,
) -> str:
    evidence_block = "\n\n".join(_format_evidence_chunk(item) for item in evidence)
    preamble = synthesis_preamble(
        has_transitive_sources=any(item.cited_sources for item in evidence),
        has_graph_context=bool(graph_context_prompt),
    )
    sections = [preamble]
    if graph_context_prompt:
        sections.append(graph_context_prompt)
    brief_block = _format_learner_brief(brief)
    if brief_block:
        sections.append(brief_block)
    sections.append(f"Question:\n{question}\n")
    sections.append(f"Evidence:\n{evidence_block}\n")
    return "\n".join(sections)


def _format_learner_brief(brief: LearnerBrief | None) -> str:
    """Render the ``## Learner Brief`` prompt section, or ``""`` when empty.

    Only the fields the caller actually set are listed — an absent field
    means "no constraint", not "none". The trailing instruction keeps the
    brief additive framing rather than a licence to ignore the evidence.
    """
    if brief is None or brief.is_empty():
        return ""
    lines = ["## Learner Brief"]
    if brief.audience:
        lines.append(f"- Audience: {brief.audience}")
    if brief.modality:
        lines.append(f"- Modality: {brief.modality}")
    if brief.bloom_target:
        lines.append(f"- Bloom target: {brief.bloom_target}")
    if brief.constraints:
        lines.append(f"- Constraints: {brief.constraints}")
    lines.append(
        "Tailor the answer's framing, depth, and examples to this brief "
        "without ignoring the evidence-grounding requirement above.\n"
    )
    return "\n".join(lines)


def _format_evidence_chunk(item: EvidenceChunk) -> str:
    """Render an evidence chunk for the synthesis prompt.

    Includes a ``Sources:`` line when the chunk carries transitive citations
    so the LLM can attribute claims back to the originating research.
    """
    header = f"[{item.citation_label}]"
    if item.cited_sources:
        header += f"\nSources: {', '.join(item.cited_sources)}"
    return f"{header}\n{item.text}"


def _strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` scratchpad blocks emitted by reasoning models."""
    return strip_thinking(text)


def insufficient_evidence_answer() -> AnswerEnvelope:
    """Build an insufficient-evidence answer envelope.

    Returns:
        Answer envelope with `insufficient_evidence` status.
    """
    return AnswerEnvelope(
        answer_status=QueryAnswerStatus.INSUFFICIENT_EVIDENCE,
        answer_text="Evidence was retrieved, but it is not sufficient to ground a reliable answer.",
        citations=[],
        warnings=[],
        metadata={},
    )


def no_results_answer() -> AnswerEnvelope:
    """Build a no-results answer envelope.

    Returns:
        Answer envelope with `no_results` status.
    """
    return AnswerEnvelope(
        answer_status=QueryAnswerStatus.NO_RESULTS,
        answer_text="No matching evidence was found in the current store.",
        citations=[],
        warnings=[],
        metadata={},
    )


def no_retrieval_needed_answer(rationale: str) -> AnswerEnvelope:
    """Build the canned response for queries the router decided to skip.

    Args:
        rationale: One-sentence reason the router gave (e.g. "greeting,
            no corpus signal needed").

    Returns:
        Answer envelope with ``no_retrieval_needed`` status. Empty
        citations and empty sentence_citations — there is nothing to
        cite.
    """
    body = (
        "This question does not appear to require information from the "
        "instructional-design corpus. If you intended a corpus-related "
        "question, please rephrase to point at a specific concept, "
        "framework, or model."
    )
    return AnswerEnvelope(
        answer_status=QueryAnswerStatus.NO_RETRIEVAL_NEEDED,
        answer_text=body,
        citations=[],
        warnings=[],
        metadata={"router_rationale": rationale},
    )
