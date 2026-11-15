"""Tests for BM25 lexical phrase and expansion query construction."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import duckdb
from lancedb.query import BooleanQuery, MatchQuery, PhraseQuery

from lxd.retrieval.lexical import (
    LexicalTerm,
    build_lexical_query,
    load_lexical_terms,
    phrases_for_question,
)
from lxd.settings.models import RuntimeConfig


def test_question_quotes_the_longest_overlapping_phrase() -> None:
    terms = (
        LexicalTerm("cognitive load", "wiki:cognitive-load", "wiki_owns"),
        LexicalTerm("cognitive load theory", "wiki:cognitive-load", "wiki_owns"),
    )

    phrases = phrases_for_question("explain cognitive load theory", terms)

    assert phrases == ("cognitive load theory",)


def test_short_alias_expands_to_the_owners_longest_phrase() -> None:
    terms = (
        LexicalTerm("wmc", "wiki:cognitive-load", "wiki_glossary"),
        LexicalTerm("working memory", "wiki:cognitive-load", "wiki_glossary"),
        LexicalTerm("working memory capacity", "wiki:cognitive-load", "wiki_glossary"),
    )

    phrases = phrases_for_question("what does WMC limit", terms)

    assert phrases == ("working memory capacity",)


def test_alias_inside_a_matched_phrase_does_not_expand_another_owner() -> None:
    terms = (
        LexicalTerm("cognitive load theory", "wiki:cognitive-load", "wiki_owns"),
        LexicalTerm("theory", "wiki:theory-of-change", "wiki_glossary"),
        LexicalTerm("theory of change", "wiki:theory-of-change", "wiki_glossary"),
    )

    phrases = phrases_for_question("explain cognitive load theory", terms)

    assert phrases == ("cognitive load theory",)


def test_label_does_not_expand_to_a_phrase_the_question_never_used() -> None:
    terms = (
        LexicalTerm("design", "ontology:coe:C05201", "label"),
        LexicalTerm("backward design", "ontology:coe:C05201", "alias"),
    )

    assert phrases_for_question("how do I design a course", terms) == ()


def test_build_lexical_query_keeps_words_adds_phrases_and_strips_curies() -> None:
    bare = build_lexical_query("what is backward design", ())
    quoted = build_lexical_query(
        "what is backward design coe:C05201",
        ("backward design",),
        expansion_terms=("Understanding by Design", "coe:C05202"),
    )

    assert isinstance(bare, MatchQuery)
    assert bare.query == "what is backward design"
    assert isinstance(quoted, BooleanQuery)
    match_queries = [clause for _, clause in quoted.queries if isinstance(clause, MatchQuery)]
    phrase_queries = [clause for _, clause in quoted.queries if isinstance(clause, PhraseQuery)]
    assert [clause.query for clause in match_queries] == [
        "what is backward design",
        "Understanding by Design",
    ]
    assert [clause.query for clause in phrase_queries] == ["backward design"]


def test_load_lexical_terms_reads_glossary_and_duckdb_terms_without_curies(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "_glossary.md").write_text(
        "\n".join(
            (
                "| Canonical term | Aliases | Owner |",
                "| --- | --- | --- |",
                "| working memory capacity | WMC | [[cognitive-load]] |",
                "",
            )
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "central.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE all_entities (
                id VARCHAR,
                collection_name VARCHAR,
                pref_label VARCHAR,
                definition VARCHAR
            );
            INSERT INTO all_entities VALUES
                ('coe:C05201', 'instructional_methods', 'Understanding by Design', 'Three-stage backward design.');

            CREATE TABLE all_terms (
                resource_id VARCHAR,
                written_form VARCHAR,
                term_kind VARCHAR
            );
            INSERT INTO all_terms VALUES
                ('coe:C05201', 'UbD', 'acronym');
            """
        )
    finally:
        con.close()

    config = SimpleNamespace(
        paths=SimpleNamespace(
            corpus_path=wiki,
            central_duckdb_path=db_path,
            central_configs_path=tmp_path / "missing-central",
        )
    )

    terms = load_lexical_terms(cast("RuntimeConfig", config))
    assert all("coe:c" not in term.normalized.lower() for term in terms)
    assert "working memory capacity" in phrases_for_question("what does WMC limit", terms)
    assert "understanding by design" in phrases_for_question("apply UbD stages", terms)
