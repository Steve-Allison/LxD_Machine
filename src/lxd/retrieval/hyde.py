"""HyDE — Hypothetical Document Embeddings (query rewriting).

For an under-specified or jargon-light query like *"what is backward
design"*, the dense retrieval signal is fragile: the literal question
embeds far from the chunks that actually answer it (those chunks talk
about "outcome-aligned curriculum design" or similar phrasings).
HyDE asks the local LLM for a hypothetical answer to the question,
then embeds *that* — bringing the query vector into the
neighbourhood of the actual answer chunks.

Local-first: uses the configured Ollama model
(``config.retrieval.hyde_model``). One LLM call per query; only
when ``config.retrieval.hyde_enabled`` is True (default False).
Adds ~2-5s of query latency on a local 14B model — opt-in.

Failure-safe: if Ollama is unreachable or returns nothing useful,
``generate_hypothetical_answer`` returns the empty string. The
caller then falls back to embedding the literal question, so HyDE
turns into a no-op rather than breaking retrieval.
"""

import re
from typing import Final

import ollama
import structlog

from lxd.ingest.llm_client import get_ollama_client
from lxd.settings.models import RuntimeConfig

_THINK_BLOCK_PATTERN: Final = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_log = structlog.get_logger(__name__)

_HYDE_PROMPT: Final = """\
You are answering a knowledge-base search query.

Write a concise, plausible answer to the question below. Use the
vocabulary and phrasing that would appear in an authoritative source.
Two to four sentences is plenty. Do not say "I don't know" — write the
most likely correct answer based on common knowledge of the topic.

Question:
{question}

Hypothetical answer:
"""


def generate_hypothetical_answer(question: str, config: RuntimeConfig) -> str:
    """Generate a hypothetical answer for HyDE retrieval.

    Returns the answer text, or an empty string on any failure so the
    caller can fall back to embedding the literal question (graceful
    degradation — HyDE becomes a no-op rather than a hard error).
    """
    cleaned = question.strip()
    if not cleaned:
        return ""
    cfg = config.retrieval
    prompt = _HYDE_PROMPT.format(question=cleaned)
    try:
        client = get_ollama_client(str(config.ollama.url), float(cfg.hyde_timeout_secs))
        response = client.generate(
            model=cfg.hyde_model,
            prompt=prompt,
            think=False if config.models.llm_no_think else None,
            options={
                "temperature": cfg.hyde_temperature,
                "num_predict": cfg.hyde_max_tokens,
            },
        )
    except (ollama.RequestError, ollama.ResponseError) as exc:
        _log.warning("hyde_generation_failed", error=str(exc))
        return ""
    raw = str(response.get("response") if isinstance(response, dict) else response.response or "")
    return _THINK_BLOCK_PATTERN.sub("", raw).strip()
