"""Detect ontology mentions in chunk text spans."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from operator import attrgetter
from typing import Any

from lxd.ontology.loader.types import AnchorConstraint, RecognitionPattern
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
    recognition_patterns: Sequence[RecognitionPattern] | None = None,
    suppressed_terms: Mapping[str, frozenset[str]] | None = None,
    anchor_constraints: Sequence[AnchorConstraint] | None = None,
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
        recognition_patterns: Compiled library patterns applied after
            literal matching.
        suppressed_terms: Normalized negative surfaces that must not be
            kept for an entity.
        anchor_constraints: Context anchors a mention must satisfy.

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
    if recognition_patterns:
        matches.extend(_pattern_mentions(normalized, recognition_patterns))
    resolved = _resolve_overlaps(matches)
    if suppressed_terms:
        resolved = [
            mention
            for mention in resolved
            if not _is_suppressed(mention, normalized, suppressed_terms)
        ]
    if anchor_constraints:
        resolved = [
            mention
            for mention in resolved
            if _anchors_allow(mention, normalized, anchor_constraints)
        ]
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
    priority = {"canonical_id": 0, "alias": 1, "indicator": 2, "pattern": 3}
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


def _pattern_mentions(normalized: str, patterns: Sequence[RecognitionPattern]) -> list[Mention]:
    """Return mentions produced by compiled recognition patterns."""
    found: list[Mention] = []
    for spec in patterns:
        for match in spec.compiled.finditer(normalized):
            surface = normalize_match_text(match.group(0))
            if len(surface) < 3:
                continue
            found.append(
                Mention(
                    entity_id=spec.entity_id,
                    term_source="pattern",
                    surface_form=surface,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )
    return found


def _is_suppressed(
    mention: Mention,
    normalized: str,
    suppressed_terms: Mapping[str, frozenset[str]],
) -> bool:
    """Return whether a mention is a library negative surface."""
    negatives = suppressed_terms.get(mention.entity_id)
    if not negatives:
        return False
    if mention.surface_form in negatives:
        return True
    window = normalized[max(0, mention.start_char - 48) : mention.end_char + 48]
    return any(negative in window and mention.surface_form in negative for negative in negatives)


def _anchors_allow(
    mention: Mention,
    normalized: str,
    constraints: Sequence[AnchorConstraint],
) -> bool:
    """Return whether every applicable anchor rule is satisfied."""
    applicable = [
        rule
        for rule in constraints
        if rule.entity_id == mention.entity_id
        and (rule.surface is None or rule.surface == mention.surface_form)
    ]
    if not applicable:
        return True
    for rule in applicable:
        radius = rule.window_tokens * 8
        window = normalized[
            max(0, mention.start_char - radius) : min(len(normalized), mention.end_char + radius)
        ]
        if not any(anchor in window for anchor in rule.anchors):
            return False
    return True
