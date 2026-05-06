"""Tests for entity disambiguation (B-KG-2)."""

from __future__ import annotations

from lxd.ingest.mentions import detect_mentions
from lxd.ontology.ambiguity import ambiguous_surface_forms_with_candidates
from lxd.ontology.disambiguator import context_window
from lxd.ontology.matcher import MatcherTermRecord, build_automaton


def _records() -> list[MatcherTermRecord]:
    return [
        MatcherTermRecord(
            normalized_term="id",
            entity_id="instructional_design",
            term_source="alias",
        ),
        MatcherTermRecord(
            normalized_term="id",
            entity_id="identifier",
            term_source="alias",
        ),
        MatcherTermRecord(
            normalized_term="bloom",
            entity_id="bloom_taxonomy",
            term_source="canonical_id",
        ),
    ]


def test_ambiguous_map_excludes_unambiguous_terms() -> None:
    ambiguous = ambiguous_surface_forms_with_candidates(_records())

    assert "bloom" not in ambiguous, (
        "Unambiguous surface forms should not appear in the ambiguous map; saw 'bloom' included."
    )
    assert ambiguous["id"] == ["identifier", "instructional_design"], (
        "Candidate list should be sorted by entity_id for deterministic iteration; "
        f"saw {ambiguous['id']}"
    )


def test_ambiguous_map_returns_sorted_candidates() -> None:
    records = [
        MatcherTermRecord(normalized_term="x", entity_id="zebra", term_source="alias"),
        MatcherTermRecord(normalized_term="x", entity_id="apple", term_source="alias"),
        MatcherTermRecord(normalized_term="x", entity_id="mango", term_source="alias"),
    ]
    ambiguous = ambiguous_surface_forms_with_candidates(records)
    assert ambiguous["x"] == ["apple", "mango", "zebra"]


def test_detect_mentions_calls_disambiguator_only_on_ambiguous_terms() -> None:
    records = _records()
    automaton = build_automaton(records)
    ambiguous_map = ambiguous_surface_forms_with_candidates(records)

    calls: list[tuple[str, list[str]]] = []

    def fake_disambiguator(window: str, candidates: list[str]) -> str | None:
        calls.append((window, candidates))
        return "instructional_design"

    text = "id is the framework. bloom is also relevant."
    mentions = detect_mentions(
        text,
        automaton,
        ambiguous_map=ambiguous_map,
        disambiguator=fake_disambiguator,
    )

    assert len(calls) == 1, (
        f"Disambiguator should fire once for the ambiguous 'id' only; saw {calls}"
    )
    assert calls[0][1] == ["identifier", "instructional_design"]
    id_mentions = [m for m in mentions if m.surface_form == "id"]
    assert id_mentions, "Expected the 'id' mention to survive."
    assert id_mentions[0].entity_id == "instructional_design", (
        "Disambiguator's choice should overwrite the upstream entity_id."
    )


def test_detect_mentions_keeps_upstream_entity_when_disambiguator_returns_none() -> None:
    records = _records()
    automaton = build_automaton(records)
    ambiguous_map = ambiguous_surface_forms_with_candidates(records)

    def undecided_disambiguator(window: str, candidates: list[str]) -> str | None:
        return None

    text = "id matters here."
    mentions = detect_mentions(
        text,
        automaton,
        ambiguous_map=ambiguous_map,
        disambiguator=undecided_disambiguator,
    )

    id_mentions = [m for m in mentions if m.surface_form == "id"]
    assert id_mentions
    # The upstream Aho-Corasick payload wins (last-write or whichever
    # was registered last); we don't assert which entity, only that
    # *some* upstream entity is preserved rather than crashing.
    assert id_mentions[0].entity_id in {"instructional_design", "identifier"}


def test_detect_mentions_no_op_without_ambiguous_map() -> None:
    records = _records()
    automaton = build_automaton(records)

    def should_not_run(window: str, candidates: list[str]) -> str | None:
        raise AssertionError("disambiguator must not run when ambiguous_map is None")

    mentions = detect_mentions(
        "id and bloom",
        automaton,
        ambiguous_map=None,
        disambiguator=should_not_run,
    )
    assert mentions  # No crash, returns the upstream Aho-Corasick result.


def test_context_window_clips_to_text_bounds() -> None:
    text = "abcdefghij"
    assert context_window(text, start=0, end=2, radius=20) == "abcdefghij"
    assert context_window(text, start=4, end=6, radius=2) == "cdefgh"


def test_context_window_includes_mention_span() -> None:
    text = "lorem ipsum dolor sit amet"
    window = context_window(text, start=12, end=17, radius=5)
    assert "dolor" in window
