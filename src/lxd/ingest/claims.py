"""Extract fine-grained factual claims from chunks using LLM."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime

import structlog

from lxd.domain.ids import blake3_hex
from lxd.settings.models import RuntimeConfig
from lxd.stores.models import ClaimRecord
from lxd.stores.sqlite import insert_claims, load_chunk_ids_with_claims

_log = structlog.get_logger(__name__)

_VALID_CLAIM_TYPES = frozenset({"assertion", "definition", "comparison", "causal", "procedural"})

_SYSTEM_PROMPT = """You are a knowledge graph builder specialising in learning experience design (LxD), instructional design, and educational theory.

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


def extract_claims_for_chunks(
    connection: sqlite3.Connection,
    config: RuntimeConfig,
    *,
    force: bool = False,
) -> int:
    """Extract claims from all qualifying chunks.

    Qualifying chunks have ≥ claim_extraction_min_mentions entity mentions.
    Incremental: skips chunks that already have claims unless force=True.

    Returns:
        Number of claims extracted.
    """
    kg_cfg = config.knowledge_graph

    # Find qualifying chunks: those with enough entity mentions
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
        _log.info("claim extraction: no qualifying chunks found")
        return 0

    # Determine which chunks to skip (already have claims)
    existing_claim_chunks = load_chunk_ids_with_claims(connection) if not force else set()
    chunks_to_process = [
        row for row in qualifying_rows if str(row["chunk_id"]) not in existing_claim_chunks
    ]

    _log.info(
        "claim extraction starting",
        qualifying_chunks=len(qualifying_rows),
        already_extracted=len(qualifying_rows) - len(chunks_to_process),
        to_process=len(chunks_to_process),
    )

    total_claims = 0
    for idx, row in enumerate(chunks_to_process):
        chunk_id = str(row["chunk_id"])
        document_id = str(row["document_id"])
        source_rel_path = str(row["source_rel_path"])
        chunk_text = str(row["text"])

        # Load entity IDs for this chunk
        entity_rows = connection.execute(
            "SELECT DISTINCT entity_id FROM mention_rows WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchall()
        entity_ids = sorted(str(r["entity_id"]) for r in entity_rows)

        claims = _extract_claims_for_chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            source_rel_path=source_rel_path,
            chunk_text=chunk_text,
            entity_ids=entity_ids,
            config=config,
        )

        if claims:
            insert_claims(connection, claims)
            total_claims += len(claims)

        if (idx + 1) % 100 == 0:
            _log.info(
                "claim extraction progress",
                processed=idx + 1,
                total=len(chunks_to_process),
                claims_so_far=total_claims,
            )

    _log.info("claim extraction complete", total_claims=total_claims)
    return total_claims


def _extract_claims_for_chunk(
    *,
    chunk_id: str,
    document_id: str,
    source_rel_path: str,
    chunk_text: str,
    entity_ids: list[str],
    config: RuntimeConfig,
) -> list[ClaimRecord]:
    """Extract claims from a single chunk using LLM."""
    kg_cfg = config.knowledge_graph
    raw_claims = _call_with_fallback(chunk_text, entity_ids, config)
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
    kg_cfg = config.knowledge_graph
    if kg_cfg.claim_extraction_backend == "openai":
        return kg_cfg.claim_extraction_model
    if kg_cfg.claim_extraction_backend == "ollama":
        return kg_cfg.claim_extraction_fallback_model
    return "none"


def _build_user_prompt(chunk_text: str, entity_ids: list[str]) -> str:
    entities_str = "\n".join(f"  - {e}" for e in entity_ids)
    return f"Text:\n{chunk_text}\n\nEntity IDs found in this text:\n{entities_str}"


def _parse_response(raw_text: str) -> list[dict]:
    try:
        data = json.loads(raw_text)
        items = data.get("claims", []) if isinstance(data, dict) else []
        return [item for item in items if isinstance(item, dict)]
    except Exception:
        return []


def _call_with_fallback(
    chunk_text: str,
    entity_ids: list[str],
    config: RuntimeConfig,
) -> list[dict]:
    kg_cfg = config.knowledge_graph

    if kg_cfg.claim_extraction_backend == "openai":
        try:
            return _call_openai(chunk_text, entity_ids, config)
        except Exception as exc:
            _log.warning("OpenAI claim extraction failed, trying fallback: %s", exc)
            try:
                return _call_ollama(chunk_text, entity_ids, config)
            except Exception as fallback_exc:
                _log.warning("Ollama claim extraction fallback failed: %s", fallback_exc)
            return []

    if kg_cfg.claim_extraction_backend == "ollama":
        try:
            return _call_ollama(chunk_text, entity_ids, config)
        except Exception as exc:
            _log.warning("Ollama claim extraction failed: %s", exc)
            return []

    return []


def _call_openai(
    chunk_text: str,
    entity_ids: list[str],
    config: RuntimeConfig,
) -> list[dict]:
    import openai

    kg_cfg = config.knowledge_graph
    openai_cfg = config.openai
    api_key_env = openai_cfg.api_key_env if openai_cfg else "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env!r} is not set.")

    client = openai.OpenAI(api_key=api_key, timeout=float(kg_cfg.claim_extraction_timeout_secs))
    response = client.chat.completions.create(
        model=kg_cfg.claim_extraction_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(chunk_text, entity_ids)},
        ],
        temperature=kg_cfg.claim_extraction_temperature,
        response_format={"type": "json_object"},
        max_tokens=2000,
    )
    content = response.choices[0].message.content or ""
    return _parse_response(content)


def _call_ollama(
    chunk_text: str,
    entity_ids: list[str],
    config: RuntimeConfig,
) -> list[dict]:
    import ollama

    kg_cfg = config.knowledge_graph
    client = ollama.Client(
        host=str(config.ollama.url),
        timeout=float(kg_cfg.claim_extraction_timeout_secs),
    )
    response = client.chat(
        model=kg_cfg.claim_extraction_fallback_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(chunk_text, entity_ids)},
        ],
        options={"temperature": kg_cfg.claim_extraction_temperature},
        format="json",
    )
    content = (
        response["message"]["content"] if isinstance(response, dict) else response.message.content
    )
    return _parse_response(content or "")
