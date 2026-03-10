from __future__ import annotations

from dataclasses import dataclass

import ollama

from lxd.domain.status import QueryAnswerStatus
from lxd.settings.models import RuntimeConfig


@dataclass(frozen=True)
class EvidenceChunk:
    citation_label: str
    text: str
    score: float


@dataclass(frozen=True)
class AnswerEnvelope:
    answer_status: QueryAnswerStatus
    answer_text: str
    citations: list[str]
    warnings: list[str]
    metadata: dict[str, object]


def no_results_answer() -> AnswerEnvelope:
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
) -> AnswerEnvelope:
    citations = [chunk.citation_label for chunk in evidence]
    prompt = _build_prompt(question, evidence)
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
    )


def synthesis_unavailable_answer(citations: list[str], warning: str) -> AnswerEnvelope:
    return AnswerEnvelope(
        answer_status=QueryAnswerStatus.SYNTHESIS_UNAVAILABLE,
        answer_text="Evidence was retrieved, but the configured synthesis model is unavailable.",
        citations=citations,
        warnings=[warning],
        metadata={},
    )


def probe_synthesis_model(config: RuntimeConfig) -> tuple[bool, str | None]:
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


def _build_prompt(question: str, evidence: list[EvidenceChunk]) -> str:
    evidence_block = "\n\n".join(f"[{item.citation_label}]\n{item.text}" for item in evidence)
    return (
        "Answer the question using only the evidence below.\n"
        "If the evidence is insufficient, say so plainly.\n"
        "Do not invent facts.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence_block}\n"
    )


def _strip_thinking(text: str) -> str:
    start_tag = "<think>"
    end_tag = "</think>"
    cleaned = text
    while start_tag in cleaned and end_tag in cleaned:
        start_index = cleaned.index(start_tag)
        end_index = cleaned.index(end_tag, start_index) + len(end_tag)
        cleaned = (cleaned[:start_index] + cleaned[end_index:]).strip()
    return cleaned


def _client(config: RuntimeConfig) -> ollama.Client:
    return ollama.Client(host=str(config.ollama.url), timeout=float(config.synthesis.timeout_secs))


def insufficient_evidence_answer() -> AnswerEnvelope:
    return AnswerEnvelope(
        answer_status=QueryAnswerStatus.INSUFFICIENT_EVIDENCE,
        answer_text="Evidence was retrieved, but it is not sufficient to ground a reliable answer.",
        citations=[],
        warnings=[],
        metadata={},
    )
