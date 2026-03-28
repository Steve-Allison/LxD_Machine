"""Tests for relation extraction record building and parsing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from lxd.ingest.relations import (
    _build_relation_records,
    _build_user_prompt,
    _parse_response,
    _RawRelation,
    build_valid_predicates,
)

# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


def test_parse_response_valid_json():
    """Valid JSON with relations array is parsed correctly."""
    raw = json.dumps(
        {
            "relations": [
                {
                    "subject": "entity_a",
                    "predicate": "teaches",
                    "object": "entity_b",
                    "confidence": 0.9,
                }
            ]
        }
    )
    result = _parse_response(raw)
    assert len(result) == 1
    assert result[0].subject == "entity_a"
    assert result[0].predicate == "teaches"
    assert result[0].object_ == "entity_b"


def test_parse_response_empty_relations():
    """Empty relations array returns empty list."""
    result = _parse_response('{"relations": []}')
    assert result == []


def test_parse_response_invalid_json():
    """Malformed JSON returns empty list."""
    result = _parse_response("not json")
    assert result == []


def test_parse_response_missing_fields_filtered():
    """Relations with missing subject/predicate/object are filtered."""
    raw = json.dumps(
        {
            "relations": [
                {"subject": "a", "predicate": "teaches", "object": "b", "confidence": 0.9},
                {"subject": "a", "predicate": "teaches"},  # missing object
                {"subject": "a", "object": "b"},  # missing predicate
            ]
        }
    )
    result = _parse_response(raw)
    assert len(result) == 1


def test_parse_response_non_string_confidence_coerced():
    """Non-float confidence values are coerced or defaulted."""
    raw = json.dumps(
        {
            "relations": [
                {"subject": "a", "predicate": "teaches", "object": "b", "confidence": "0.75"},
                {"subject": "c", "predicate": "teaches", "object": "d", "confidence": "invalid"},
            ]
        }
    )
    result = _parse_response(raw)
    assert len(result) == 2
    assert result[0].confidence == 0.75
    assert result[1].confidence == 0.5  # default


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------


def test_build_user_prompt_includes_all_components():
    """User prompt includes text, entity IDs, and predicates."""
    prompt = _build_user_prompt(
        "Some text.", ["entity_a", "entity_b"], frozenset({"teaches", "relates_to"})
    )
    assert "Some text." in prompt
    assert "entity_a" in prompt
    assert "entity_b" in prompt
    assert "teaches" in prompt
    assert "relates_to" in prompt


# ---------------------------------------------------------------------------
# _build_relation_records
# ---------------------------------------------------------------------------


def _make_config() -> Any:
    """Build a minimal mock config for relation record building."""
    return SimpleNamespace(
        relation_extraction=SimpleNamespace(
            backend="openai",
            openai_model="gpt-4o-mini",
            ollama_model="qwen3:14b",
            max_relations_per_chunk=15,
        ),
    )


def test_build_relation_records_valid():
    """Valid raw relations produce ExtractedRelationRecord objects."""
    raw = [_RawRelation(subject="a", predicate="teaches", object_="b", confidence=0.85)]
    records = _build_relation_records(
        raw=raw,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="t.md",
        entity_ids=["a", "b"],
        valid_predicates=frozenset({"teaches"}),
        config=_make_config(),
    )
    assert len(records) == 1
    assert records[0].subject_entity_id == "a"
    assert records[0].object_entity_id == "b"
    assert records[0].predicate == "teaches"


def test_build_relation_records_filters_unknown_entities():
    """Relations with entity IDs not in the provided set are dropped."""
    raw = [_RawRelation(subject="unknown", predicate="teaches", object_="b", confidence=0.8)]
    records = _build_relation_records(
        raw=raw,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="t.md",
        entity_ids=["a", "b"],
        valid_predicates=frozenset({"teaches"}),
        config=_make_config(),
    )
    assert len(records) == 0


def test_build_relation_records_filters_unknown_predicates():
    """Relations with predicates not in valid set are dropped."""
    raw = [_RawRelation(subject="a", predicate="unknown_pred", object_="b", confidence=0.8)]
    records = _build_relation_records(
        raw=raw,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="t.md",
        entity_ids=["a", "b"],
        valid_predicates=frozenset({"teaches"}),
        config=_make_config(),
    )
    assert len(records) == 0


def test_build_relation_records_filters_self_relations():
    """Relations where subject == object are dropped."""
    raw = [_RawRelation(subject="a", predicate="teaches", object_="a", confidence=0.8)]
    records = _build_relation_records(
        raw=raw,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="t.md",
        entity_ids=["a"],
        valid_predicates=frozenset({"teaches"}),
        config=_make_config(),
    )
    assert len(records) == 0


def test_build_relation_records_clamps_confidence():
    """Confidence is clamped to [0.0, 1.0]."""
    raw = [
        _RawRelation(subject="a", predicate="teaches", object_="b", confidence=1.5),
        _RawRelation(subject="a", predicate="teaches", object_="c", confidence=-0.1),
    ]
    records = _build_relation_records(
        raw=raw,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="t.md",
        entity_ids=["a", "b", "c"],
        valid_predicates=frozenset({"teaches"}),
        config=_make_config(),
    )
    assert records[0].confidence == 1.0
    assert records[1].confidence == 0.0


def test_build_relation_records_respects_max_per_chunk():
    """Only max_relations_per_chunk relations are processed."""
    config = _make_config()
    config.relation_extraction.max_relations_per_chunk = 2
    raw = [
        _RawRelation(subject="a", predicate="teaches", object_=f"e{i}", confidence=0.8)
        for i in range(5)
    ]
    entity_ids = ["a"] + [f"e{i}" for i in range(5)]
    records = _build_relation_records(
        raw=raw,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="t.md",
        entity_ids=entity_ids,
        valid_predicates=frozenset({"teaches"}),
        config=config,
    )
    assert len(records) == 2


# ---------------------------------------------------------------------------
# build_valid_predicates
# ---------------------------------------------------------------------------


def test_build_valid_predicates_filters_entity_origin():
    """Only entity-origin relations contribute predicates."""
    records = [
        SimpleNamespace(relation_type="teaches", origin_kind="entity"),
        SimpleNamespace(relation_type="references", origin_kind="file"),
        SimpleNamespace(relation_type="contains", origin_kind="entity"),
    ]
    result = build_valid_predicates(records)
    assert result == frozenset({"teaches", "contains"})
