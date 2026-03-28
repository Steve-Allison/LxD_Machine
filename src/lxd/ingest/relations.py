"""Relation extraction from document chunks using LLM (OpenAI primary, Ollama fallback)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from lxd.domain.ids import blake3_hex
from lxd.settings.models import RuntimeConfig
from lxd.stores.models import ExtractedRelationRecord, MentionRecord

_LOG = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a knowledge graph builder specialising in learning experience design (LxD), instructional design, and educational theory.

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

    raw = _call_with_fallback(chunk_text, entity_ids, valid_predicates, config)
    if not raw:
        _LOG.debug("chunk=%s entities=%d raw_relations=0 stored=0", chunk_id[:8], len(entity_ids))
        return []

    timestamp = datetime.now(UTC).isoformat()
    entity_id_set = set(entity_ids)
    model_name = _active_model(config)
    records: list[ExtractedRelationRecord] = []
    dropped_predicate: list[str] = []

    for rel in raw[: cfg.max_relations_per_chunk]:
        if rel.subject not in entity_id_set or rel.object_ not in entity_id_set:
            continue
        if rel.predicate not in valid_predicates:
            dropped_predicate.append(rel.predicate)
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

    if dropped_predicate:
        _LOG.debug(
            "chunk=%s dropped predicates not in ontology: %s",
            chunk_id[:8],
            sorted(set(dropped_predicate)),
        )
    _LOG.debug(
        "chunk=%s entities=%d raw=%d stored=%d dropped_pred=%d",
        chunk_id[:8],
        len(entity_ids),
        len(raw),
        len(records),
        len(dropped_predicate),
    )
    return records


def build_valid_predicates(relation_records: list) -> frozenset[str]:
    """Derive the set of entity-to-entity predicate types from the loaded ontology."""
    return frozenset(r.relation_type for r in relation_records if r.origin_kind == "entity")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _active_model(config: RuntimeConfig) -> str:
    cfg = config.relation_extraction
    if cfg.backend == "openai":
        return cfg.openai_model
    return cfg.ollama_model


def _build_user_prompt(
    chunk_text: str,
    entity_ids: list[str],
    valid_predicates: frozenset[str],
) -> str:
    entities_str = "\n".join(f"  - {e}" for e in entity_ids)
    predicates_str = ", ".join(sorted(valid_predicates))
    return (
        f"Text:\n{chunk_text}\n\n"
        f"Entity IDs found in this text:\n{entities_str}\n\n"
        f"Valid predicates: {predicates_str}"
    )


def _parse_response(raw_text: str) -> list[_RawRelation]:
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


def _call_with_fallback(
    chunk_text: str,
    entity_ids: list[str],
    valid_predicates: frozenset[str],
    config: RuntimeConfig,
) -> list[_RawRelation]:
    cfg = config.relation_extraction

    if cfg.backend == "openai":
        try:
            return _call_openai(chunk_text, entity_ids, valid_predicates, config)
        except Exception as exc:
            _LOG.warning("OpenAI relation extraction failed, trying fallback: %s", exc)
            if cfg.fallback_backend == "ollama":
                try:
                    return _call_ollama(chunk_text, entity_ids, valid_predicates, config)
                except Exception as fallback_exc:
                    _LOG.warning("Ollama relation extraction fallback failed: %s", fallback_exc)
            return []

    if cfg.backend == "ollama":
        try:
            return _call_ollama(chunk_text, entity_ids, valid_predicates, config)
        except Exception as exc:
            _LOG.warning("Ollama relation extraction failed: %s", exc)
            return []

    return []


def _call_openai(
    chunk_text: str,
    entity_ids: list[str],
    valid_predicates: frozenset[str],
    config: RuntimeConfig,
) -> list[_RawRelation]:
    import openai  # lazy import — only needed for openai backend

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
            {"role": "system", "content": _SYSTEM_PROMPT},
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


def _call_ollama(
    chunk_text: str,
    entity_ids: list[str],
    valid_predicates: frozenset[str],
    config: RuntimeConfig,
) -> list[_RawRelation]:
    import ollama  # lazy import

    cfg = config.relation_extraction
    client = ollama.Client(host=str(config.ollama.url), timeout=float(cfg.timeout_secs))
    response = client.chat(
        model=cfg.ollama_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
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
