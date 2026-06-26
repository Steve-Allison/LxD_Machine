"""Detect ontology mentions in chunk text spans."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from operator import attrgetter
from typing import Any

from lxd.ontology.normalization import normalize_match_text


@dataclass(frozen=True, slots=True)
class Mention:
    """Detected mention span for an ontology term."""

    entity_id: str
    term_source: str
    surface_form: str
    start_char: int
    end_char: int


def detect_mentions(
    text: str,
    automaton: Any,
    *,
    ambiguous_map: dict[str, list[str]] | None = None,
    disambiguator: Callable[[str, list[str]], str | None] | None = None,
    context_radius: int = 200,
) -> list[Mention]:
    """Detect ontology term mentions in text.

    Args:
        text: Input text to process.
        automaton: Aho-Corasick automaton built from matcher terms.
        ambiguous_map: Optional ``{normalized_term: [entity_id, ...]}``
            for surface forms that map to >1 candidate (built once at
            ontology load via
            :func:`lxd.ontology.ambiguity.ambiguous_surface_forms_with_candidates`).
            When provided alongside ``disambiguator``, ambiguous matches
            get re-resolved via the surrounding context window. Both
            ``ambiguous_map`` and ``disambiguator`` must be set together;
            either one missing falls back to the upstream first-match
            policy (Aho-Corasick last-write-wins payload).
        disambiguator: Callable ``(window_text, candidates) -> entity_id |
            None``. ``None`` return means "could not decide"; the mention
            keeps the upstream entity_id.
        context_radius: ±characters of context around the ambiguous
            mention fed to the disambiguator. ±200 by default (B-KG-2
            spec).

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
    resolved = _resolve_overlaps(matches)
    if ambiguous_map and disambiguator is not None:
        resolved = _apply_disambiguator(
            resolved,
            text=normalized,
            ambiguous_map=ambiguous_map,
            disambiguator=disambiguator,
            context_radius=context_radius,
        )
    return resolved


def _apply_disambiguator(
    mentions: list[Mention],
    *,
    text: str,
    ambiguous_map: dict[str, list[str]],
    disambiguator: Callable[[str, list[str]], str | None],
    context_radius: int,
) -> list[Mention]:
    """Re-assign ``entity_id`` on ambiguous mentions using the disambiguator.

    Mentions whose surface form is unambiguous, or whose disambiguator
    returns ``None``, are left unchanged. The structural fields
    (``start_char``, ``end_char``, ``surface_form``, ``term_source``)
    are always preserved so chunk-level alignment is unaffected by the
    re-assignment.
    """
    out: list[Mention] = []
    for mention in mentions:
        candidates = ambiguous_map.get(mention.surface_form)
        if not candidates or len(candidates) < 2:
            out.append(mention)
            continue
        window = text[
            max(0, mention.start_char - context_radius) : min(
                len(text), mention.end_char + context_radius
            )
        ]
        chosen = disambiguator(window, candidates)
        if chosen is None or chosen == mention.entity_id:
            out.append(mention)
            continue
        out.append(replace(mention, entity_id=chosen))
    return out


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
    return sorted(accepted, key=attrgetter("start_char", "end_char", "entity_id"))
