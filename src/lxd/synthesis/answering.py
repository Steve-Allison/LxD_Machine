"""Generate final answer envelopes from ranked evidence chunks.

Two modes:

- :func:`synthesize_answer` — one-shot: blocks until the local Ollama
  model finishes, returns a single :class:`AnswerEnvelope`.
- :func:`stream_synthesize_answer` — iterator: yields incremental
  :class:`StreamingTextDelta` events as the model emits text, then
  exactly one terminal :class:`AnswerEnvelope` once the stream
  completes (or sooner on error). Both stay on the local Ollama
  backend; remote synthesis was rejected as incompatible with the
  local-first design (see ``feedback_local_only_no_remote_rerank.md``
  and the SOTA-plan strike of item ``[#4]``).
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Final

import ollama

from lxd.domain.status import QueryAnswerStatus
from lxd.ingest.llm_client import get_ollama_client
from lxd.retrieval.graph_routing import GraphContext
from lxd.settings.models import RuntimeConfig
from lxd.synthesis.citation_alignment import SentenceCitation, align_citations
from lxd.synthesis.sampler import Sampler, SamplerFailure, SamplerRequest

_THINK_BLOCK_PATTERN: Final = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


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
    "present, your citations should reference both the chunk citation label\n"
    "AND the underlying sources transitively, e.g.\n"
    '"[citation_label] (citing source_a.md, source_b.pdf)".\n'
)
SYNTHESIS_PREAMBLE_GRAPH_CONTEXT: Final = (
    "\nThe graph context below provides structured knowledge about entities,\n"
    "communities, and claims relevant to the question. Use it to frame your\n"
    "answer but ground all facts in the source evidence.\n"
)


def synthesis_preamble(
    *,
    has_transitive_sources: bool = True,
    has_graph_context: bool = True,
) -> str:
    """Return the synthesis-prompt preamble with optional sub-sections enabled.

    Used by both the runtime synthesis path (where ``has_*`` flags reflect
    the actual evidence on the call) and the MCP ``lxd_synthesis_preamble``
    prompt (where both flags default to ``True`` so clients see every
    sub-section the system might emit).
    """
    text = SYNTHESIS_PREAMBLE_BASE
    if has_transitive_sources:
        text += SYNTHESIS_PREAMBLE_TRANSITIVE_SOURCES
    if has_graph_context:
        text += SYNTHESIS_PREAMBLE_GRAPH_CONTEXT
    return text


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    """Evidence snippet and score used for synthesis.

    ``cited_sources`` carries the underlying-source filenames parsed from
    the chunk's wiki frontmatter (``**Sources**:`` line). When present, the
    synthesis prompt instructs the model to cite both the wiki page (via
    ``citation_label``) and the underlying sources, giving the user a
    transitive provenance trail back to the original research.
    """

    citation_label: str
    text: str
    score: float
    cited_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnswerEnvelope:
    """Final answer payload including citations and warnings.

    ``sentence_citations`` carries per-sentence attribution: each entry is
    one sentence of ``answer_text`` plus the citation labels the model
    attributed to it inline. Sentences with empty ``citation_labels``
    are unattributed claims — surface them in the UI as a hallucination
    risk signal.
    """

    answer_status: QueryAnswerStatus
    answer_text: str
    citations: list[str]
    warnings: list[str]
    metadata: dict[str, object]
    sentence_citations: list[SentenceCitation] = field(default_factory=list)
    graph_context: GraphContext | None = None


@dataclass(frozen=True, slots=True)
class StreamingTextDelta:
    """Incremental text emitted by :func:`stream_synthesize_answer`.

    Streamed deltas are model-raw — they may include partial tokens or
    fragments of ``<think>...</think>`` reasoning blocks the model
    sometimes emits. The terminal :class:`AnswerEnvelope` carries the
    full text with think blocks stripped, so consumers that only need
    the final answer can ignore the deltas; consumers that want
    progress UI render them as they arrive.
    """

    text: str


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


def synthesize_answer(
    question: str,
    evidence: list[EvidenceChunk],
    config: RuntimeConfig,
    *,
    graph_context_prompt: str = "",
    sampler: Sampler | None = None,
) -> AnswerEnvelope:
    """Synthesize an answer from retrieved evidence chunks.

    Args:
        question: User question text.
        evidence: Evidence chunks used for synthesis.
        config: Runtime configuration object.
        graph_context_prompt: Optional graph context to prepend to the prompt.
        sampler: Optional client-side sampler. When provided (the MCP
            server passes one when ``mcp.synthesis_backend`` is set to
            ``client_sampling``), the synthesis prompt is dispatched via
            :func:`fastmcp.Context.sample` so the connected client's LLM
            answers. On :class:`SamplerFailure` (client did not advertise
            sampling capability, upstream error, empty response) the call
            transparently falls back to the server-side Ollama path and
            surfaces the reason as a warning on the resulting envelope.

    Returns:
        Answer envelope from synthesis or fallback.
    """
    citations = [chunk.citation_label for chunk in evidence]
    prompt = _build_prompt(question, evidence, graph_context_prompt=graph_context_prompt)
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
    """Run synthesis on the server-side Ollama model (the default path)."""
    try:
        response = _client(config).generate(
            model=config.models.llm,
            prompt=prompt,
            think=False if config.models.llm_no_think else None,
            options={
                "temperature": config.synthesis.temperature,
                "num_predict": config.synthesis.max_tokens,
            },
        )
    except (ollama.RequestError, ollama.ResponseError) as exc:
        return synthesis_unavailable_answer(citations, f"Synthesis model unavailable: {exc}")
    answer_text = _strip_thinking(str(response["response"])).strip()
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
) -> Iterator[StreamingTextDelta | AnswerEnvelope]:
    """Stream a synthesis answer from the local Ollama model.

    Yields zero or more :class:`StreamingTextDelta` events as the model
    emits text, then exactly one terminal :class:`AnswerEnvelope` (the
    same object :func:`synthesize_answer` would return). Consumers that
    don't drain the iterator to completion miss the envelope; consumers
    that abandon mid-stream simply lose the answer status without
    leaking server resources beyond Ollama's own request lifetime.

    Errors during the stream short-circuit: the iterator yields a
    single :class:`AnswerEnvelope` with
    ``QueryAnswerStatus.SYNTHESIS_UNAVAILABLE`` and stops.
    """
    citations = [chunk.citation_label for chunk in evidence]
    prompt = _build_prompt(question, evidence, graph_context_prompt=graph_context_prompt)
    try:
        stream = _client(config).generate(
            model=config.models.llm,
            prompt=prompt,
            think=False if config.models.llm_no_think else None,
            stream=True,
            options={
                "temperature": config.synthesis.temperature,
                "num_predict": config.synthesis.max_tokens,
            },
        )
    except (ollama.RequestError, ollama.ResponseError) as exc:
        yield synthesis_unavailable_answer(citations, f"Synthesis model unavailable: {exc}")
        return

    accumulated: list[str] = []
    try:
        for response_chunk in stream:
            text = str(response_chunk.response or "")
            if not text:
                continue
            accumulated.append(text)
            yield StreamingTextDelta(text=text)
    except (ollama.RequestError, ollama.ResponseError) as exc:
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
    """Probe backend availability and return probe metadata.

    Args:
        config: Runtime configuration object.

    Returns:
        Tuple of `(supported, warning)` for synthesis backend.
    """
    try:
        response = _client(config).generate(
            model=config.models.llm,
            prompt="Reply with exactly OK.",
            think=False if config.models.llm_no_think else None,
            options={"temperature": 0, "num_predict": 16},
        )
    except (ollama.RequestError, ollama.ResponseError) as exc:
        return False, str(exc)
    if not _strip_thinking(str(response["response"])).strip():
        return False, "Synthesis probe returned an empty response."
    return True, None


def _build_prompt(
    question: str, evidence: list[EvidenceChunk], *, graph_context_prompt: str = ""
) -> str:
    evidence_block = "\n\n".join(_format_evidence_chunk(item) for item in evidence)
    preamble = synthesis_preamble(
        has_transitive_sources=any(item.cited_sources for item in evidence),
        has_graph_context=bool(graph_context_prompt),
    )
    sections = [preamble]
    if graph_context_prompt:
        sections.append(graph_context_prompt)
    sections.append(f"Question:\n{question}\n")
    sections.append(f"Evidence:\n{evidence_block}\n")
    return "\n".join(sections)


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
    """Remove ``<think>...</think>`` scratchpad blocks emitted by reasoning models.

    Uses a single non-greedy DOTALL regex pass rather than the previous
    iterative ``index``/slice loop, which was O(n^2) on the number of blocks.
    """
    return _THINK_BLOCK_PATTERN.sub("", text).strip()


def _client(config: RuntimeConfig) -> ollama.Client:
    return get_ollama_client(str(config.ollama.url), float(config.synthesis.timeout_secs))


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
