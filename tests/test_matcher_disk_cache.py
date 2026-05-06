"""Tests for the Aho-Corasick matcher disk cache (B-ROBUST-2)."""

from __future__ import annotations

from pathlib import Path

import ahocorasick

from lxd.ontology.matcher import (
    MatcherTermRecord,
    build_or_load_automaton,
    matcher_termset_hash,
)


def _records() -> list[MatcherTermRecord]:
    return [
        MatcherTermRecord(normalized_term="alpha", entity_id="alpha", term_source="canonical_id"),
        MatcherTermRecord(normalized_term="beta", entity_id="beta", term_source="canonical_id"),
        MatcherTermRecord(normalized_term="gamma", entity_id="gamma", term_source="alias"),
    ]


def test_first_call_writes_cache_file(tmp_path: Path) -> None:
    records = _records()
    cache_dir = tmp_path / "matcher_cache"

    automaton = build_or_load_automaton(records, cache_dir=cache_dir)

    assert isinstance(automaton, ahocorasick.Automaton)
    expected_path = cache_dir / f"matcher-{matcher_termset_hash(records)}.pkl"
    assert expected_path.is_file(), (
        f"Cache file should be written on cold-build; expected {expected_path}."
    )


def test_warm_load_returns_same_term_payloads(tmp_path: Path) -> None:
    records = _records()
    cache_dir = tmp_path / "matcher_cache"

    cold = build_or_load_automaton(records, cache_dir=cache_dir)
    warm = build_or_load_automaton(records, cache_dir=cache_dir)

    cold_terms = {key for key, _ in cold.items()}
    warm_terms = {key for key, _ in warm.items()}
    assert cold_terms == warm_terms == {"alpha", "beta", "gamma"}, (
        "Warm-load automaton must contain the same matcher terms as the cold build."
    )


def test_hash_mismatch_writes_separate_cache_file(tmp_path: Path) -> None:
    records_a = _records()
    records_b = [
        MatcherTermRecord(normalized_term="delta", entity_id="delta", term_source="canonical_id"),
    ]
    cache_dir = tmp_path / "matcher_cache"

    build_or_load_automaton(records_a, cache_dir=cache_dir)
    build_or_load_automaton(records_b, cache_dir=cache_dir)

    files = sorted(p.name for p in cache_dir.glob("matcher-*.pkl"))
    assert len(files) == 2, (
        f"Different ontologies should produce different cache filenames; saw {files}."
    )


def test_corrupt_cache_falls_through_to_fresh_build(tmp_path: Path) -> None:
    records = _records()
    cache_dir = tmp_path / "matcher_cache"
    cache_dir.mkdir()
    cache_path = cache_dir / f"matcher-{matcher_termset_hash(records)}.pkl"
    cache_path.write_bytes(b"this is not a pickled automaton")

    automaton = build_or_load_automaton(records, cache_dir=cache_dir)

    assert isinstance(automaton, ahocorasick.Automaton), (
        "A corrupt cache file must not crash the build path; fresh build should run."
    )
    assert {key for key, _ in automaton.items()} == {"alpha", "beta", "gamma"}
