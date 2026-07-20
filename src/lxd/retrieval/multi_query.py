"""Multi-query expansion — LLM-generated paraphrases for retrieval fan-out.

A single literal query embeds close to chunks that happen to share its
exact vocabulary and misses chunks that answer the same question with
different phrasing. Multi-query fan-out asks the LLM for a handful of
paraphrases of the user's question, retrieves against each one, and
fuses the resulting candidate lists (see
:mod:`lxd.retrieval.query_pipeline`) so a lexical mismatch on any one
phrasing no longer sinks a relevant chunk out of the ranked results.

Failure-safe: any LLM failure or unparseable response yields an empty
list, so the caller's fan-out collapses back to single-query retrieval
rather than breaking — multi-query becomes a no-op, not a hard error.
"""

import asyncio
import json
from typing import Any, Final

import structlog

from lxd.ingest.llm_client import call_openai_async
from lxd.settings.models import RetrievalConfig

_log = structlog.get_logger(__name__)

_MULTI_QUERY_SYSTEM_PROMPT: Final = """\
You are helping search an instructional-design knowledge base.

Given the user's QUESTION, write {count} alternative phrasings that
preserve the original intent but vary vocabulary, phrasing, or angle
of approach — e.g. swapping jargon for plain language, or framing the
same concept from a different angle. Each paraphrase must be a
complete, standalone question a reader could search with.

Return ONLY a JSON object with one key, "paraphrases", holding a JSON
array of exactly {count} strings. No preamble, no markdown fences."""


def generate_query_paraphrases(question: str, config: RetrievalConfig) -> list[str]:
    """Generate up to ``config.multi_query_count`` paraphrases of ``question``.

    Args:
        question: The user's question text.
        config: ``retrieval`` section of the runtime config.

    Returns:
        A deduplicated list of non-empty paraphrases (never including
        the original question, compared case-insensitively). Returns
        an empty list on any failure — timeout, malformed JSON, or a
        blank question — so the caller falls back to single-query
        retrieval. Never raises.
    """
    cleaned = question.strip()
    if not cleaned:
        return []
    try:
        raw = asyncio.run(
            call_openai_async(
                system_prompt=_MULTI_QUERY_SYSTEM_PROMPT.format(count=config.multi_query_count),
                user_prompt=f"QUESTION:\n{cleaned}",
                model=config.multi_query_model,
                temperature=config.multi_query_temperature,
                timeout=config.multi_query_timeout_secs,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
        )
    except Exception as exc:
        _log.warning("multi_query_generation_failed", error=str(exc))
        return []
    return _parse_paraphrases(raw, original=cleaned, limit=config.multi_query_count)


def _parse_paraphrases(raw: str, *, original: str, limit: int) -> list[str]:
    """Parse the JSON the LLM returned; never raises."""
    payload = _safe_json(raw)
    candidates = payload.get("paraphrases")
    if not isinstance(candidates, list):
        return []
    seen: set[str] = {original.casefold()}
    paraphrases: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        cleaned_candidate = candidate.strip()
        if not cleaned_candidate:
            continue
        folded = cleaned_candidate.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        paraphrases.append(cleaned_candidate)
        if len(paraphrases) >= limit:
            break
    return paraphrases


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload
