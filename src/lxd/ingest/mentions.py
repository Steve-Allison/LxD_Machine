"""Detect ontology mentions in chunk text spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lxd.ontology.normalization import normalize_match_text


@dataclass(frozen=True)
class Mention:
    """Detected mention span for an ontology term."""
    entity_id: str
    term_source: str
    surface_form: str
    start_char: int
    end_char: int


def detect_mentions(text: str, automaton: Any) -> list[Mention]:
    """Detect ontology term mentions in text.

    Args:
        text: Input text to process.
        automaton: Aho-Corasick automaton built from matcher terms.

    Returns:
        Non-overlapping mention spans sorted by position.
    """
    normalized = normalize_match_text(text)
    matches: list[Mention] = []
    for end_index, payload in automaton.iter(normalized):
        matched = payload["normalized_term"]
        start_index = end_index - len(matched) + 1
        matches.append(
            Mention(
                entity_id=str(payload["entity_id"]),
                term_source=str(payload["term_source"]),
                surface_form=matched,
                start_char=start_index,
                end_char=end_index + 1,
            )
        )
    return _resolve_overlaps(matches)


def _resolve_overlaps(matches: list[Mention]) -> list[Mention]:
    priority = {"canonical_id": 0, "alias": 1, "indicator": 2}
    ordered = sorted(
        matches,
        key=lambda item: (
            -(item.end_char - item.start_char),
            priority.get(item.term_source, 99),
            item.start_char,
            item.entity_id,
        ),
    )
    accepted: list[Mention] = []
    occupied: list[tuple[int, int]] = []
    for match in ordered:
        if any(not (match.end_char <= start or match.start_char >= end) for start, end in occupied):
            continue
        accepted.append(match)
        occupied.append((match.start_char, match.end_char))
    return sorted(accepted, key=lambda item: (item.start_char, item.end_char, item.entity_id))
