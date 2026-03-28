"""Tests for async claim extraction via llm_client."""

from __future__ import annotations

import json
from typing import Any

from lxd.ingest.claims import _build_claim_records, _build_user_prompt, _parse_response

# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


def test_parse_response_valid_json():
    """Valid JSON with claims array is parsed correctly."""
    raw = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "Bloom's taxonomy classifies learning objectives",
                    "subject": "blooms_taxonomy",
                    "object": None,
                    "claim_type": "definition",
                    "confidence": 0.9,
                }
            ]
        }
    )
    result = _parse_response(raw)
    assert len(result) == 1
    assert result[0]["claim_text"] == "Bloom's taxonomy classifies learning objectives"


def test_parse_response_empty_claims():
    """Empty claims array returns empty list."""
    result = _parse_response('{"claims": []}')
    assert result == []


def test_parse_response_invalid_json():
    """Malformed JSON returns empty list."""
    result = _parse_response("not json at all")
    assert result == []


def test_parse_response_non_dict_items_filtered():
    """Non-dict items in claims array are filtered out."""
    raw = json.dumps({"claims": [{"claim_text": "valid"}, "invalid string", 42]})
    result = _parse_response(raw)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------


def test_build_user_prompt_includes_entities():
    """User prompt includes chunk text and entity IDs."""
    prompt = _build_user_prompt("Some chunk text.", ["entity_a", "entity_b"])
    assert "Some chunk text." in prompt
    assert "entity_a" in prompt
    assert "entity_b" in prompt


# ---------------------------------------------------------------------------
# _build_claim_records
# ---------------------------------------------------------------------------


def _make_config() -> Any:
    """Build a minimal mock config for claim record building."""
    from types import SimpleNamespace

    return SimpleNamespace(
        knowledge_graph=SimpleNamespace(
            claim_max_per_chunk=10,
            claim_extraction_backend="openai",
            claim_extraction_model="gpt-4o-mini",
            claim_extraction_fallback_model="qwen3:14b",
        ),
    )


def test_build_claim_records_valid_claims():
    """Valid raw claims produce ClaimRecord objects."""
    raw_claims = [
        {
            "claim_text": "Cognitive load theory applies to multimedia learning",
            "subject": "cognitive_load",
            "object": "multimedia_learning",
            "claim_type": "assertion",
            "confidence": 0.85,
        }
    ]
    records = _build_claim_records(
        raw_claims=raw_claims,
        chunk_id="chunk-001",
        document_id="doc-001",
        source_rel_path="test.md",
        entity_ids=["cognitive_load", "multimedia_learning"],
        config=_make_config(),
    )
    assert len(records) == 1
    assert records[0].claim_text == "Cognitive load theory applies to multimedia learning"
    assert records[0].subject_entity_id == "cognitive_load"
    assert records[0].object_entity_id == "multimedia_learning"
    assert records[0].confidence == 0.85


def test_build_claim_records_invalid_entity_ids_nullified():
    """Entity IDs not in the provided set are set to None."""
    raw_claims = [
        {
            "claim_text": "A claim",
            "subject": "unknown_entity",
            "object": "cognitive_load",
            "claim_type": "assertion",
            "confidence": 0.8,
        }
    ]
    records = _build_claim_records(
        raw_claims=raw_claims,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="t.md",
        entity_ids=["cognitive_load"],
        config=_make_config(),
    )
    assert len(records) == 1
    assert records[0].subject_entity_id is None
    assert records[0].object_entity_id == "cognitive_load"


def test_build_claim_records_clamps_confidence():
    """Confidence values are clamped to [0.0, 1.0]."""
    raw_claims = [
        {"claim_text": "Over-confident", "confidence": 1.5, "claim_type": "assertion"},
        {"claim_text": "Negative", "confidence": -0.3, "claim_type": "assertion"},
    ]
    records = _build_claim_records(
        raw_claims=raw_claims,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="t.md",
        entity_ids=[],
        config=_make_config(),
    )
    assert records[0].confidence == 1.0
    assert records[1].confidence == 0.0


def test_build_claim_records_invalid_claim_type_defaults():
    """Invalid claim type defaults to 'assertion'."""
    raw_claims = [
        {"claim_text": "A claim", "claim_type": "invalid_type", "confidence": 0.8},
    ]
    records = _build_claim_records(
        raw_claims=raw_claims,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="t.md",
        entity_ids=[],
        config=_make_config(),
    )
    assert records[0].claim_type == "assertion"


def test_build_claim_records_empty_text_skipped():
    """Claims with empty or non-string text are skipped."""
    raw_claims = [
        {"claim_text": "", "confidence": 0.8},
        {"claim_text": 42, "confidence": 0.8},
        {"claim_text": "Valid claim", "confidence": 0.8},
    ]
    records = _build_claim_records(
        raw_claims=raw_claims,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="t.md",
        entity_ids=[],
        config=_make_config(),
    )
    assert len(records) == 1
    assert records[0].claim_text == "Valid claim"


def test_build_claim_records_respects_max_per_chunk():
    """Only claim_max_per_chunk claims are processed."""
    config = _make_config()
    config.knowledge_graph.claim_max_per_chunk = 2
    raw_claims = [{"claim_text": f"Claim {i}", "confidence": 0.8} for i in range(5)]
    records = _build_claim_records(
        raw_claims=raw_claims,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="t.md",
        entity_ids=[],
        config=config,
    )
    assert len(records) == 2
