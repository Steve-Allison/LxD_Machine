from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import ahocorasick

from lxd.domain.ids import blake3_hex
from lxd.ontology.normalization import normalize_match_text


@dataclass(frozen=True, order=True)
class MatcherTermRecord:
    normalized_term: str
    entity_id: str
    term_source: str


def canonical_matcher_term_records(
    entity_definitions: Iterable[dict[str, Any]],
) -> list[MatcherTermRecord]:
    records: set[MatcherTermRecord] = set()
    for entity in entity_definitions:
        canonical_id = _coerce_required_str(entity, "canonical_id")
        for term_source, values in (
            ("canonical_id", [canonical_id]),
            ("alias", _coerce_str_list(entity.get("aliases", []))),
            ("indicator", _coerce_str_list(entity.get("indicators", []))),
        ):
            for value in values:
                normalized = normalize_match_text(value)
                if normalized:
                    records.add(
                        MatcherTermRecord(
                            normalized_term=normalized,
                            entity_id=canonical_id,
                            term_source=term_source,
                        )
                    )
    return sorted(records)


def matcher_termset_hash(records: Iterable[MatcherTermRecord]) -> str:
    lines = [
        json.dumps(
            {
                "entity_id": record.entity_id,
                "term_source": record.term_source,
                "normalized_term": record.normalized_term,
            },
            sort_keys=False,
            separators=(",", ":"),
        )
        for record in records
    ]
    return blake3_hex("\n".join(lines))


def build_automaton(
    records: Iterable[MatcherTermRecord],
) -> ahocorasick.Automaton:  # type: ignore[type-arg]
    automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_STRING)
    for record in records:
        payload = {
            "entity_id": record.entity_id,
            "term_source": record.term_source,
            "normalized_term": record.normalized_term,
        }
        automaton.add_word(record.normalized_term, payload)
    automaton.make_automaton()
    return automaton


def _coerce_required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string field: {key}")
    return value


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected list of strings")
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item)
    return result
