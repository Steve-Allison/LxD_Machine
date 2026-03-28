"""Relation extraction from document chunks using LLM (OpenAI primary, Ollama fallback).

Uses the shared llm_client for async concurrency, prompt caching,
and OpenAI Batch API support.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from lxd.domain.ids import blake3_hex
from lxd.ingest.llm_client import (
    build_cached_system_prompt,
    call_with_fallback_async,
    collect_batch_results,
    prepare_batch_jsonl,
    run_concurrent_extraction,
    submit_batch,
)
from lxd.settings.models import RuntimeConfig
from lxd.stores.models import ExtractedRelationRecord, MentionRecord

_log = structlog.get_logger(__name__)

_RELATION_BASE_PROMPT = """You are a knowledge graph builder specialising in learning experience design (LxD), instructional design, and educational theory.

Given a text chunk and a list of entity IDs already identified in that text, extract semantic relationships between those entities.

Rules:
- subject and object must be entity IDs from the provided list
- predicate must be from the provided list of valid predicates
- confidence: 0.7–1.0 for relationships explicitly stated in the text, 0.4–0.69 for clearly implied relationships
- Only extract relationships supported by the text — do not infer beyond what is written
- Respond with JSON only — no explanation, no markdown fences

JSON output format:
{"relations": [{"subject": "entity_id", "predicate": "predicate", "object": "entity_id", "confidence": 0.85}]}

Return {"relations": []} if no clear relationships exist between the provided entities."""


@dataclass(frozen=True)
class _RawRelation:
    subject: str
    predicate: str
    object_: str
    confidence: float


# ---------------------------------------------------------------------------
# Public: per-chunk sync extraction (used by pipeline.py during ingest)
# ---------------------------------------------------------------------------


def extract_relations_for_chunk(
    chunk_id: str,
    document_id: str,
    source_rel_path: str,
    chunk_text: str,
    mention_records: list[MentionRecord],
    valid_predicates: frozenset[str],
    config: RuntimeConfig,
) -> list[ExtractedRelationRecord]:
    """Extract semantic relations from a chunk using LLM (OpenAI with Ollama fallback).

    Returns an empty list if the chunk has too few entity mentions or if all backends fail.
    """
    cfg = config.relation_extraction

    entity_ids = sorted({m.entity_id for m in mention_records})
    if len(entity_ids) < cfg.min_entity_mentions:
        return []

    if not valid_predicates:
        return []

    raw = _call_with_fallback_sync(chunk_text, entity_ids, valid_predicates, config)
    if not raw:
        _log.debug("chunk=%s entities=%d raw_relations=0 stored=0", chunk_id[:8], len(entity_ids))
        return []

    records = _build_relation_records(
        raw=raw,
        chunk_id=chunk_id,
        document_id=document_id,
        source_rel_path=source_rel_path,
        entity_ids=entity_ids,
        valid_predicates=valid_predicates,
        config=config,
    )

    _log.debug(
        "chunk=%s entities=%d raw=%d stored=%d",
        chunk_id[:8],
        len(entity_ids),
        len(raw),
        len(records),
    )
    return records


# ---------------------------------------------------------------------------
# Public: bulk async extraction (used by pipeline.py for batch processing)
# ---------------------------------------------------------------------------


def extract_relations_for_chunks_async(
    chunks_data: list[dict[str, Any]],
    valid_predicates: frozenset[str],
    config: RuntimeConfig,
) -> list[ExtractedRelationRecord]:
    """Extract relations for multiple chunks concurrently.

    Args:
        chunks_data: List of dicts with keys: chunk_id, document_id, source_rel_path,
            chunk_text, mention_records.
        valid_predicates: Set of allowed predicate strings.
        config: Runtime configuration.

    Returns:
        All extracted relation records.
    """
    if not valid_predicates or not chunks_data:
        return []
    return asyncio.run(_extract_relations_bulk_async(chunks_data, valid_predicates, config))


# ---------------------------------------------------------------------------
# Public: Batch API
# ---------------------------------------------------------------------------


def prepare_relations_batch_jsonl(
    chunks_data: list[dict[str, Any]],
    valid_predicates: frozenset[str],
    config: RuntimeConfig,
    output_dir: Path,
) -> Path:
    """Prepare a JSONL file for OpenAI Batch API relation extraction."""
    cfg = config.relation_extraction

    # Collect all entity IDs for prompt caching
    all_entity_set: set[str] = set()
    for item in chunks_data:
        mentions: list[MentionRecord] = item["mention_records"]
        for m in mentions:
            all_entity_set.add(m.entity_id)
    all_entity_ids = sorted(all_entity_set)

    cached_prompt = build_cached_system_prompt(
        _RELATION_BASE_PROMPT,
        entity_vocabulary=all_entity_ids,
        predicate_vocabulary=sorted(valid_predicates),
    )

    items: list[dict[str, Any]] = []
    for item in chunks_data:
        entity_ids = sorted({m.entity_id for m in item["mention_records"]})
        if len(entity_ids) < cfg.min_entity_mentions:
            continue
        items.append(
            {
                "custom_id": item["chunk_id"],
                "chunk_id": item["chunk_id"],
                "document_id": item["document_id"],
                "source_rel_path": item["source_rel_path"],
                "chunk_text": item["chunk_text"],
                "entity_ids": entity_ids,
            }
        )

    if not items:
        raise RuntimeError("No qualifying chunks for relation extraction batch.")

    def _build_messages(item: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cached_prompt},
            {
                "role": "user",
                "content": _build_user_prompt(
                    item["chunk_text"], item["entity_ids"], valid_predicates
                ),
            },
        ]

    output_path = output_dir / "relations_batch.jsonl"
    prepare_batch_jsonl(
        items,
        build_messages_fn=_build_messages,
        model=cfg.openai_model,
        temperature=cfg.temperature,
        max_tokens=1000,
        response_format={"type": "json_object"},
        output_path=output_path,
    )

    meta_path = output_dir / "relations_batch_chunks.json"
    meta_path.write_text(json.dumps({item["custom_id"]: item for item in items}, indent=2))

    return output_path


def submit_relations_batch(jsonl_path: Path, config: RuntimeConfig) -> str:
    """Submit the relations JSONL to OpenAI Batch API."""
    api_key_env = config.openai.api_key_env if config.openai else "OPENAI_API_KEY"
    return submit_batch(
        jsonl_path,
        description="lxd-machine relation extraction batch",
        api_key_env=api_key_env,
        metadata={"type": "relations"},
    )


def collect_relations_batch(
    batch_id: str,
    config: RuntimeConfig,
    chunks_meta_path: Path,
    valid_predicates: frozenset[str],
) -> list[ExtractedRelationRecord]:
    """Download and parse relations from a completed batch."""
    chunks_meta: dict[str, dict[str, Any]] = json.loads(chunks_meta_path.read_text())
    api_key_env = config.openai.api_key_env if config.openai else "OPENAI_API_KEY"

    def _parse(custom_id: str, content: str) -> list[ExtractedRelationRecord] | None:
        meta = chunks_meta.get(custom_id)
        if meta is None:
            return None
        raw = _parse_response(content)
        return _build_relation_records(
            raw=raw,
            chunk_id=meta["chunk_id"],
            document_id=meta["document_id"],
            source_rel_path=meta["source_rel_path"],
            entity_ids=meta["entity_ids"],
            valid_predicates=valid_predicates,
            config=config,
        )

    results = collect_batch_results(batch_id, parse_fn=_parse, api_key_env=api_key_env)

    all_records: list[ExtractedRelationRecord] = []
    for records in results:
        if records:
            all_records.extend(records)

    _log.info("batch_relations_collected", batch_id=batch_id, total=len(all_records))
    return all_records


def build_valid_predicates(relation_records: list[Any]) -> frozenset[str]:
    """Derive the set of entity-to-entity predicate types from the loaded ontology."""
    return frozenset(r.relation_type for r in relation_records if r.origin_kind == "entity")


# ---------------------------------------------------------------------------
# Async internals
# ---------------------------------------------------------------------------


async def _extract_relations_bulk_async(
    chunks_data: list[dict[str, Any]],
    valid_predicates: frozenset[str],
    config: RuntimeConfig,
) -> list[ExtractedRelationRecord]:
    """Async concurrent relation extraction for multiple chunks."""
    cfg = config.relation_extraction

    # Collect all entity IDs for prompt caching
    all_entity_set: set[str] = set()
    filtered_items: list[dict[str, Any]] = []
    for item in chunks_data:
        mentions: list[MentionRecord] = item["mention_records"]
        entity_ids = sorted({m.entity_id for m in mentions})
        if len(entity_ids) < cfg.min_entity_mentions:
            continue
        item["_entity_ids"] = entity_ids
        filtered_items.append(item)
        all_entity_set.update(entity_ids)

    if not filtered_items:
        return []

    cached_prompt = build_cached_system_prompt(
        _RELATION_BASE_PROMPT,
        entity_vocabulary=sorted(all_entity_set),
        predicate_vocabulary=sorted(valid_predicates),
    )

    api_key_env = config.openai.api_key_env if config.openai else "OPENAI_API_KEY"
    ollama_host = str(config.ollama.url)

    async def _extract_one(item: dict[str, Any]) -> list[ExtractedRelationRecord]:
        entity_ids: list[str] = item["_entity_ids"]
        raw_text = await call_with_fallback_async(
            system_prompt=cached_prompt,
            user_prompt=_build_user_prompt(item["chunk_text"], entity_ids, valid_predicates),
            primary_backend=cfg.backend,
            openai_model=cfg.openai_model,
            ollama_model=cfg.ollama_model,
            fallback_backend=cfg.fallback_backend,
            temperature=cfg.temperature,
            openai_timeout=float(cfg.timeout_secs),
            ollama_timeout=float(cfg.timeout_secs),
            max_tokens=1000,
            response_format={"type": "json_object"},
            api_key_env=api_key_env,
            ollama_host=ollama_host,
        )

        raw = _parse_response(raw_text)
        return _build_relation_records(
            raw=raw,
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            source_rel_path=item["source_rel_path"],
            entity_ids=entity_ids,
            valid_predicates=valid_predicates,
            config=config,
        )

    results = await run_concurrent_extraction(
        filtered_items,
        _extract_one,
        max_concurrent=cfg.max_concurrent,
        sub_batch_size=cfg.sub_batch_size,
        label="relation_extraction",
    )

    all_records: list[ExtractedRelationRecord] = []
    for records in results:
        all_records.extend(records)

    _log.info("async_relation_extraction_complete", total=len(all_records))
    return all_records


# ---------------------------------------------------------------------------
# Sync fallback (used by per-chunk extraction in pipeline)
# ---------------------------------------------------------------------------


def _call_with_fallback_sync(
    chunk_text: str,
    entity_ids: list[str],
    valid_predicates: frozenset[str],
    config: RuntimeConfig,
) -> list[_RawRelation]:
    """Synchronous call with OpenAI → Ollama fallback."""
    cfg = config.relation_extraction

    if cfg.backend == "openai":
        try:
            return _call_openai_sync(chunk_text, entity_ids, valid_predicates, config)
        except Exception as exc:
            _log.warning("OpenAI relation extraction failed, trying fallback: %s", exc)
            if cfg.fallback_backend == "ollama":
                try:
                    return _call_ollama_sync(chunk_text, entity_ids, valid_predicates, config)
                except Exception as fallback_exc:
                    _log.warning("Ollama relation extraction fallback failed: %s", fallback_exc)
            return []

    if cfg.backend == "ollama":
        try:
            return _call_ollama_sync(chunk_text, entity_ids, valid_predicates, config)
        except Exception as exc:
            _log.warning("Ollama relation extraction failed: %s", exc)
            return []

    return []


def _call_openai_sync(
    chunk_text: str,
    entity_ids: list[str],
    valid_predicates: frozenset[str],
    config: RuntimeConfig,
) -> list[_RawRelation]:
    import openai

    cfg = config.relation_extraction
    openai_cfg = config.openai
    api_key_env = openai_cfg.api_key_env if openai_cfg else "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env!r} is not set.")

    client = openai.OpenAI(api_key=api_key, timeout=float(cfg.timeout_secs))
    response = client.chat.completions.create(
        model=cfg.openai_model,
        messages=[
            {"role": "system", "content": _RELATION_BASE_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(chunk_text, entity_ids, valid_predicates),
            },
        ],
        temperature=cfg.temperature,
        response_format={"type": "json_object"},
        max_tokens=1000,
    )
    content = response.choices[0].message.content or ""
    return _parse_response(content)


def _call_ollama_sync(
    chunk_text: str,
    entity_ids: list[str],
    valid_predicates: frozenset[str],
    config: RuntimeConfig,
) -> list[_RawRelation]:
    import ollama

    cfg = config.relation_extraction
    client = ollama.Client(host=str(config.ollama.url), timeout=float(cfg.timeout_secs))
    response = client.chat(
        model=cfg.ollama_model,
        messages=[
            {"role": "system", "content": _RELATION_BASE_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(chunk_text, entity_ids, valid_predicates),
            },
        ],
        options={"temperature": cfg.temperature},
        format="json",
    )
    content = (
        response["message"]["content"] if isinstance(response, dict) else response.message.content
    )
    return _parse_response(content or "")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_user_prompt(
    chunk_text: str,
    entity_ids: list[str],
    valid_predicates: frozenset[str],
) -> str:
    """Build the user prompt for a single chunk."""
    entities_str = "\n".join(f"  - {e}" for e in entity_ids)
    predicates_str = ", ".join(sorted(valid_predicates))
    return (
        f"Text:\n{chunk_text}\n\n"
        f"Entity IDs found in this text:\n{entities_str}\n\n"
        f"Valid predicates: {predicates_str}"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(raw_text: str) -> list[_RawRelation]:
    """Parse LLM JSON response into raw relation objects."""
    try:
        data = json.loads(raw_text)
        items = data.get("relations", []) if isinstance(data, dict) else []
        results: list[_RawRelation] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            subject = item.get("subject")
            predicate = item.get("predicate")
            object_ = item.get("object")
            confidence = item.get("confidence", 0.5)
            if not (
                isinstance(subject, str) and isinstance(predicate, str) and isinstance(object_, str)
            ):
                continue
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.5
            results.append(
                _RawRelation(
                    subject=subject, predicate=predicate, object_=object_, confidence=confidence
                )
            )
        return results
    except (json.JSONDecodeError, AttributeError):
        return []


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------


def _active_model(config: RuntimeConfig) -> str:
    """Return the model name that will be used for extraction."""
    cfg = config.relation_extraction
    if cfg.backend == "openai":
        return cfg.openai_model
    return cfg.ollama_model


def _build_relation_records(
    *,
    raw: list[_RawRelation],
    chunk_id: str,
    document_id: str,
    source_rel_path: str,
    entity_ids: list[str],
    valid_predicates: frozenset[str],
    config: RuntimeConfig,
) -> list[ExtractedRelationRecord]:
    """Build validated ExtractedRelationRecord list from raw LLM output.

    Pure function reused by sync, async, and batch paths.
    """
    cfg = config.relation_extraction
    if not raw:
        return []

    timestamp = datetime.now(UTC).isoformat()
    entity_id_set = set(entity_ids)
    model_name = _active_model(config)
    records: list[ExtractedRelationRecord] = []

    for rel in raw[: cfg.max_relations_per_chunk]:
        if rel.subject not in entity_id_set or rel.object_ not in entity_id_set:
            continue
        if rel.predicate not in valid_predicates:
            continue
        if rel.subject == rel.object_:
            continue
        records.append(
            ExtractedRelationRecord(
                relation_id=blake3_hex(chunk_id, rel.subject, rel.predicate, rel.object_),
                chunk_id=chunk_id,
                document_id=document_id,
                source_rel_path=source_rel_path,
                subject_entity_id=rel.subject,
                predicate=rel.predicate,
                object_entity_id=rel.object_,
                confidence=max(0.0, min(1.0, rel.confidence)),
                extraction_model=model_name,
                extracted_at=timestamp,
            )
        )

    return records
