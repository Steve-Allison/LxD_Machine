"""Contextual retrieval chunker (Anthropic-style).

For each chunk, ask the local Ollama model for a one-sentence
"this chunk discusses X in the context of Y" summary, prepend it to
the chunk text **before embedding only** (the stored chunk text stays
clean for citation rendering), and cache the summary keyed on
``(chunk_hash, model)``. The augmented form gives the embedder extra
signal beyond the literal chunk text — useful for ambiguous queries
on long documents.

Local-first: uses the configured Ollama model
(``config.chunking.contextual_summary_model``). No remote calls.

Opt-in: ``config.chunking.contextual_summary_enabled`` defaults to
False. When disabled the pipeline runs unchanged. When enabled, the
contextual path bypasses the embedding cache (each run re-embeds the
augmented text); the *summary* cache survives across runs so the
expensive Ollama summarisation step is paid once per chunk.

Storage:
    LanceDB table ``contextual_summary_cache`` adjacent to
    ``chunk_vectors`` and ``embedding_cache`` so the cache moves with
    the data dir on copy. String column rather than vector — same
    cache pattern as :mod:`lxd.ingest.embedding_cache` minus the
    ``vector`` field.
"""

import re
from dataclasses import dataclass
from typing import Any

import ollama
import pyarrow as pa
import structlog

from lxd.ingest.llm_client import get_ollama_client
from lxd.settings.models import RuntimeConfig
from lxd.stores.lance_sql import in_clause

_TABLE_NAME = "contextual_summary_cache"
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_log = structlog.get_logger(__name__)

_CONTEXT_PROMPT = """\
You are a documentation indexer. A user will search a knowledge base.

Given a CHUNK from a larger document, write ONE concise sentence that
states what the chunk is about, situating it in its document. The
sentence will be prepended to the chunk's text before embedding so
retrieval can find this chunk for queries about its topic.

Return only the sentence — no prefix, no quotes, no extra lines.

DOCUMENT TITLE: {document_title}
DOCUMENT SUMMARY: {document_summary}

CHUNK:
{chunk_text}
"""


@dataclass(frozen=True, slots=True)
class SummaryCacheLookupResult:
    """Result of looking up chunk hashes in the contextual summary cache."""

    hits: dict[int, str]
    misses_indices: list[int]

    @property
    def hit_count(self) -> int:
        return len(self.hits)

    @property
    def miss_count(self) -> int:
        return len(self.misses_indices)


def open_summary_cache_table(database: Any) -> Any:
    """Open the contextual summary cache table, creating it when missing."""
    try:
        return database.open_table(_TABLE_NAME)
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, ValueError) and "was not found" not in str(exc):
            raise
        return database.create_table(
            _TABLE_NAME,
            schema=_cache_schema(),
            mode="create",
        )


def lookup_summaries(
    cache_table: Any,
    *,
    chunk_hashes: list[str],
    model: str,
) -> SummaryCacheLookupResult:
    """Look up cached chunk-context summaries.

    Returns summaries keyed by input index, plus the indices that
    need fresh generation. LanceDB-level failures degrade to "all
    miss" so the pipeline can still proceed.
    """
    if not chunk_hashes:
        return SummaryCacheLookupResult(hits={}, misses_indices=[])
    cache_keys = [_cache_key(h, model) for h in chunk_hashes]
    unique_keys = sorted(set(cache_keys))
    try:
        rows = (
            cache_table.search()
            .where(in_clause("cache_key", unique_keys))
            .select(["cache_key", "summary"])
            .to_list()
        )
    except (FileNotFoundError, ValueError) as exc:
        _log.warning("contextual_summary_cache_lookup_skipped", error=str(exc))
        return SummaryCacheLookupResult(hits={}, misses_indices=list(range(len(chunk_hashes))))
    by_key = {str(row["cache_key"]): str(row["summary"]) for row in rows if row.get("summary")}
    hits: dict[int, str] = {}
    misses: list[int] = []
    for idx, key in enumerate(cache_keys):
        cached = by_key.get(key)
        if cached is not None:
            hits[idx] = cached
        else:
            misses.append(idx)
    return SummaryCacheLookupResult(hits=hits, misses_indices=misses)


def store_summaries(
    cache_table: Any,
    *,
    chunk_hashes: list[str],
    summaries: list[str],
    model: str,
) -> int:
    """Persist newly generated summaries to the cache.

    Empty / whitespace-only summaries are silently skipped — no value
    in caching a failed generation. Idempotent: existing entries with
    the same cache_key are deleted first.
    """
    if not chunk_hashes:
        return 0
    if len(chunk_hashes) != len(summaries):
        raise ValueError(
            f"chunk_hashes ({len(chunk_hashes)}) and summaries ({len(summaries)}) length mismatch"
        )
    deduped: dict[str, str] = {}
    for chunk_hash, summary in zip(chunk_hashes, summaries, strict=True):
        cleaned = summary.strip()
        if not cleaned:
            continue
        deduped[_cache_key(chunk_hash, model)] = cleaned
    if not deduped:
        return 0
    try:
        cache_table.delete(in_clause("cache_key", sorted(deduped)))
    except (FileNotFoundError, ValueError) as exc:
        _log.debug("contextual_summary_pre_delete_skipped", error=str(exc))
    rows = [
        {
            "cache_key": key,
            "chunk_hash": key.split("|", 1)[0],
            "model": model,
            "summary": summary,
        }
        for key, summary in deduped.items()
    ]
    try:
        cache_table.add(rows)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _log.warning("contextual_summary_cache_store_failed", error=str(exc))
        return 0
    return len(rows)


def generate_chunk_context(
    *,
    chunk_text: str,
    document_title: str,
    document_summary: str,
    config: RuntimeConfig,
) -> str:
    """Generate a one-sentence context summary via local Ollama.

    Returns the generated sentence, or an empty string on any failure
    so the pipeline can fall back to embedding the chunk without
    context (graceful degradation).
    """
    cfg = config.chunking
    prompt = _CONTEXT_PROMPT.format(
        document_title=document_title,
        document_summary=document_summary or "(no summary available)",
        chunk_text=chunk_text,
    )
    try:
        client = get_ollama_client(
            str(config.ollama.url), float(cfg.contextual_summary_timeout_secs)
        )
        response = client.generate(
            model=cfg.contextual_summary_model,
            prompt=prompt,
            think=False if config.models.llm_no_think else None,
            options={
                "temperature": cfg.contextual_summary_temperature,
                "num_predict": cfg.contextual_summary_max_tokens,
            },
        )
    except (ollama.RequestError, ollama.ResponseError) as exc:
        _log.warning("contextual_summary_failed", error=str(exc))
        return ""
    raw = str(response.get("response") if isinstance(response, dict) else response.response or "")
    text = _THINK_BLOCK_PATTERN.sub("", raw).strip()
    # Keep it one line — many models emit a paragraph break followed by
    # commentary; truncate at the first newline.
    return text.split("\n", 1)[0].strip()


def augment_chunk_for_embedding(chunk_text: str, summary: str) -> str:
    """Build the embed-time text: ``summary + blank line + chunk_text``.

    The augmented form is used for embedding only — the stored chunk
    text stays as ``chunk_text`` so citations / retrieval display
    remain clean. An empty ``summary`` returns the chunk text
    unchanged so the augmentation is a no-op when generation failed.
    """
    if not summary:
        return chunk_text
    return f"{summary}\n\n{chunk_text}"


def _cache_key(chunk_hash: str, model: str) -> str:
    return f"{chunk_hash}|{model}"


def _cache_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("cache_key", pa.string()),
            pa.field("chunk_hash", pa.string()),
            pa.field("model", pa.string()),
            pa.field("summary", pa.string()),
        ]
    )
