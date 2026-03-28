"""Extract fine-grained factual claims from chunks using LLM.

Uses the shared llm_client for async concurrency, prompt caching,
and OpenAI Batch API support.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
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
from lxd.stores.models import ClaimRecord
from lxd.stores.sqlite import insert_claims, load_chunk_ids_with_claims

_log = structlog.get_logger(__name__)

_VALID_CLAIM_TYPES = frozenset({"assertion", "definition", "comparison", "causal", "procedural"})

_CLAIM_BASE_PROMPT = """You are a knowledge graph builder specialising in learning experience design (LxD), instructional design, and educational theory.

Given a text chunk and a list of entity IDs found in that text, extract factual assertions (claims).

Rules:
- Each claim should be a single statement that could be true or false
- Link claims to subject/object entities where applicable
- A claim may have only a subject entity, or no entity linkage if it states a general domain fact
- claim_type must be one of: assertion, definition, comparison, causal, procedural
- confidence: 0.7–1.0 for claims explicitly stated, 0.4–0.69 for clearly implied claims
- Only extract claims grounded in the text — do not hallucinate
- Respond with JSON only — no explanation, no markdown fences

JSON output format:
{"claims": [{"claim_text": "...", "subject": "entity_id or null", "object": "entity_id or null", "claim_type": "assertion", "confidence": 0.85}]}

Return {"claims": []} if no clear factual assertions exist."""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def extract_claims_for_chunks(
    connection: sqlite3.Connection,
    config: RuntimeConfig,
    *,
    force: bool = False,
) -> int:
    """Extract claims from all qualifying chunks (async concurrent).

    Qualifying chunks have >= claim_extraction_min_mentions entity mentions.
    Incremental: skips chunks that already have claims unless force=True.

    Returns:
        Number of claims extracted.
    """
    return asyncio.run(_extract_claims_async(connection, config, force=force))


def prepare_claims_batch_jsonl(
    connection: sqlite3.Connection,
    config: RuntimeConfig,
    output_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Prepare a JSONL file for OpenAI Batch API claim extraction.

    Returns the path to the JSONL file.
    """
    chunks_to_process, entity_ids_by_chunk, all_entity_ids = _load_qualifying_chunks(
        connection, config, force=force
    )

    if not chunks_to_process:
        raise RuntimeError("No qualifying chunks to process.")

    kg_cfg = config.knowledge_graph
    cached_prompt = build_cached_system_prompt(_CLAIM_BASE_PROMPT, entity_vocabulary=all_entity_ids)

    items: list[dict[str, Any]] = []
    for row in chunks_to_process:
        chunk_id = str(row["chunk_id"])
        entity_ids = entity_ids_by_chunk.get(chunk_id, [])
        items.append(
            {
                "custom_id": chunk_id,
                "chunk_id": chunk_id,
                "document_id": str(row["document_id"]),
                "source_rel_path": str(row["source_rel_path"]),
                "chunk_text": str(row["text"]),
                "entity_ids": entity_ids,
            }
        )

    def _build_messages(item: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cached_prompt},
            {"role": "user", "content": _build_user_prompt(item["chunk_text"], item["entity_ids"])},
        ]

    output_path = output_dir / "claims_batch.jsonl"
    prepare_batch_jsonl(
        items,
        build_messages_fn=_build_messages,
        model=kg_cfg.claim_extraction_model,
        temperature=kg_cfg.claim_extraction_temperature,
        max_tokens=2000,
        response_format={"type": "json_object"},
        output_path=output_path,
    )

    # Write chunk metadata sidecar for result collection
    meta_path = output_dir / "claims_batch_chunks.json"
    meta_path.write_text(json.dumps({item["custom_id"]: item for item in items}, indent=2))

    return output_path


def submit_claims_batch(jsonl_path: Path, config: RuntimeConfig) -> str:
    """Submit the claims JSONL to OpenAI Batch API.

    Returns the batch ID.
    """
    api_key_env = config.openai.api_key_env if config.openai else "OPENAI_API_KEY"
    return submit_batch(
        jsonl_path,
        description="lxd-machine claim extraction batch",
        api_key_env=api_key_env,
        metadata={"type": "claims"},
    )


def collect_claims_batch(
    batch_id: str,
    connection: sqlite3.Connection,
    config: RuntimeConfig,
    chunks_meta_path: Path,
) -> int:
    """Download and insert claims from a completed batch.

    Returns the number of claims inserted.
    """
    chunks_meta: dict[str, dict[str, Any]] = json.loads(chunks_meta_path.read_text())
    api_key_env = config.openai.api_key_env if config.openai else "OPENAI_API_KEY"

    def _parse(custom_id: str, content: str) -> list[ClaimRecord] | None:
        meta = chunks_meta.get(custom_id)
        if meta is None:
            return None
        raw_claims = _parse_response(content)
        return _build_claim_records(
            raw_claims=raw_claims,
            chunk_id=meta["chunk_id"],
            document_id=meta["document_id"],
            source_rel_path=meta["source_rel_path"],
            entity_ids=meta["entity_ids"],
            config=config,
        )

    results = collect_batch_results(batch_id, parse_fn=_parse, api_key_env=api_key_env)

    total = 0
    all_records: list[ClaimRecord] = []
    for records in results:
        if records:
            all_records.extend(records)

    if all_records:
        insert_claims(connection, all_records)
        total = len(all_records)

    _log.info("batch_claims_collected", batch_id=batch_id, total_claims=total)
    return total


# ---------------------------------------------------------------------------
# Async internals
# ---------------------------------------------------------------------------


async def _extract_claims_async(
    connection: sqlite3.Connection,
    config: RuntimeConfig,
    *,
    force: bool = False,
) -> int:
    """Async concurrent claim extraction with sub-batch commits."""
    kg_cfg = config.knowledge_graph

    chunks_to_process, entity_ids_by_chunk, all_entity_ids = _load_qualifying_chunks(
        connection, config, force=force
    )

    if not chunks_to_process:
        _log.info("claim_extraction_no_qualifying_chunks")
        return 0

    # Build cache-friendly system prompt once for all calls
    cached_prompt = build_cached_system_prompt(_CLAIM_BASE_PROMPT, entity_vocabulary=all_entity_ids)

    api_key_env = config.openai.api_key_env if config.openai else "OPENAI_API_KEY"
    ollama_host = str(config.ollama.url)

    async def _extract_one(row: Any) -> list[ClaimRecord]:
        chunk_id = str(row["chunk_id"])
        document_id = str(row["document_id"])
        source_rel_path = str(row["source_rel_path"])
        chunk_text = str(row["text"])
        entity_ids = entity_ids_by_chunk.get(chunk_id, [])

        raw_text = await call_with_fallback_async(
            system_prompt=cached_prompt,
            user_prompt=_build_user_prompt(chunk_text, entity_ids),
            primary_backend=kg_cfg.claim_extraction_backend,
            openai_model=kg_cfg.claim_extraction_model,
            ollama_model=kg_cfg.claim_extraction_fallback_model,
            temperature=kg_cfg.claim_extraction_temperature,
            openai_timeout=float(kg_cfg.claim_extraction_timeout_secs),
            ollama_timeout=float(kg_cfg.claim_extraction_timeout_secs),
            max_tokens=2000,
            response_format={"type": "json_object"},
            api_key_env=api_key_env,
            ollama_host=ollama_host,
        )

        raw_claims = _parse_response(raw_text)
        return _build_claim_records(
            raw_claims=raw_claims,
            chunk_id=chunk_id,
            document_id=document_id,
            source_rel_path=source_rel_path,
            entity_ids=entity_ids,
            config=config,
        )

    total_claims = 0

    def _commit_batch(results: list[list[ClaimRecord]]) -> None:
        nonlocal total_claims
        all_records: list[ClaimRecord] = []
        for records in results:
            all_records.extend(records)
        if all_records:
            insert_claims(connection, all_records)
            total_claims += len(all_records)

    await run_concurrent_extraction(
        chunks_to_process,
        _extract_one,
        max_concurrent=kg_cfg.claim_extraction_max_concurrent,
        sub_batch_size=kg_cfg.claim_extraction_sub_batch_size,
        commit_fn=_commit_batch,
        label="claim_extraction",
    )

    _log.info("claim_extraction_complete", total_claims=total_claims)
    return total_claims


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_qualifying_chunks(
    connection: sqlite3.Connection,
    config: RuntimeConfig,
    *,
    force: bool = False,
) -> tuple[list[Any], dict[str, list[str]], list[str]]:
    """Load qualifying chunks and pre-fetch all entity IDs.

    Returns:
        (chunks_to_process, entity_ids_by_chunk, all_entity_ids_sorted)
    """
    kg_cfg = config.knowledge_graph
    min_mentions = kg_cfg.claim_extraction_min_mentions

    qualifying_rows = connection.execute(
        """
        SELECT
            c.chunk_id,
            c.document_id,
            c.source_rel_path,
            c.text,
            COUNT(DISTINCT m.entity_id) AS entity_count
        FROM chunk_rows c
        JOIN mention_rows m ON c.chunk_id = m.chunk_id
        GROUP BY c.chunk_id
        HAVING entity_count >= ?
        ORDER BY c.chunk_id
        """,
        (min_mentions,),
    ).fetchall()

    if not qualifying_rows:
        return [], {}, []

    # Determine which chunks to skip (already have claims)
    existing_claim_chunks = load_chunk_ids_with_claims(connection) if not force else set()
    chunks_to_process = [
        row for row in qualifying_rows if str(row["chunk_id"]) not in existing_claim_chunks
    ]

    _log.info(
        "claim_extraction_qualifying",
        qualifying_chunks=len(qualifying_rows),
        already_extracted=len(qualifying_rows) - len(chunks_to_process),
        to_process=len(chunks_to_process),
    )

    if not chunks_to_process:
        return [], {}, []

    # Pre-fetch ALL entity IDs per chunk in one query (eliminates N+1)
    chunk_ids = [str(row["chunk_id"]) for row in chunks_to_process]
    placeholders = ",".join("?" * len(chunk_ids))
    entity_rows = connection.execute(
        f"SELECT chunk_id, entity_id FROM mention_rows WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()

    entity_ids_by_chunk: dict[str, list[str]] = {}
    all_entity_set: set[str] = set()
    for er in entity_rows:
        cid = str(er["chunk_id"])
        eid = str(er["entity_id"])
        entity_ids_by_chunk.setdefault(cid, [])
        if eid not in set(entity_ids_by_chunk[cid]):
            entity_ids_by_chunk[cid].append(eid)
        all_entity_set.add(eid)

    # Sort entity lists for determinism
    for cid in entity_ids_by_chunk:
        entity_ids_by_chunk[cid].sort()

    all_entity_ids = sorted(all_entity_set)

    return chunks_to_process, entity_ids_by_chunk, all_entity_ids


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_user_prompt(chunk_text: str, entity_ids: list[str]) -> str:
    """Build the user prompt for a single chunk."""
    entities_str = "\n".join(f"  - {e}" for e in entity_ids)
    return f"Text:\n{chunk_text}\n\nEntity IDs found in this text:\n{entities_str}"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(raw_text: str) -> list[dict[str, Any]]:
    """Parse LLM JSON response into raw claim dicts."""
    try:
        data = json.loads(raw_text)
        items = data.get("claims", []) if isinstance(data, dict) else []
        return [item for item in items if isinstance(item, dict)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------


def _build_claim_records(
    *,
    raw_claims: list[dict[str, Any]],
    chunk_id: str,
    document_id: str,
    source_rel_path: str,
    entity_ids: list[str],
    config: RuntimeConfig,
) -> list[ClaimRecord]:
    """Build validated ClaimRecord list from raw LLM output.

    Pure function reused by both async and batch paths.
    """
    kg_cfg = config.knowledge_graph
    if not raw_claims:
        return []

    timestamp = datetime.now(UTC).isoformat()
    model_name = _active_model(config)
    entity_id_set = set(entity_ids)
    records: list[ClaimRecord] = []

    for claim in raw_claims[: kg_cfg.claim_max_per_chunk]:
        claim_text = claim.get("claim_text", "")
        if not claim_text or not isinstance(claim_text, str):
            continue

        subject = claim.get("subject")
        obj = claim.get("object")
        claim_type = claim.get("claim_type", "assertion")
        confidence = claim.get("confidence", 0.5)

        # Validate entity linkage
        if isinstance(subject, str) and subject not in entity_id_set:
            subject = None
        if isinstance(obj, str) and obj not in entity_id_set:
            obj = None
        if not isinstance(subject, str):
            subject = None
        if not isinstance(obj, str):
            obj = None

        # Validate claim type
        if claim_type not in _VALID_CLAIM_TYPES:
            claim_type = "assertion"

        # Clamp confidence
        if not isinstance(confidence, (int, float)):
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.5
        confidence = max(0.0, min(1.0, float(confidence)))

        claim_id = blake3_hex(chunk_id, claim_text)
        records.append(
            ClaimRecord(
                claim_id=claim_id,
                chunk_id=chunk_id,
                document_id=document_id,
                source_rel_path=source_rel_path,
                claim_text=claim_text.strip(),
                subject_entity_id=subject,
                object_entity_id=obj,
                claim_type=claim_type,
                confidence=confidence,
                extraction_model=model_name,
                extracted_at=timestamp,
            )
        )

    return records


def _active_model(config: RuntimeConfig) -> str:
    """Return the model name that will be used for extraction."""
    kg_cfg = config.knowledge_graph
    if kg_cfg.claim_extraction_backend == "openai":
        return kg_cfg.claim_extraction_model
    if kg_cfg.claim_extraction_backend == "ollama":
        return kg_cfg.claim_extraction_fallback_model
    return "none"
