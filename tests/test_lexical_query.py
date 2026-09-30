from pathlib import Path
from types import SimpleNamespace
from typing import cast

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


def test_canonical_id_does_not_expand_to_a_phrase_the_question_never_used() -> None:
    terms = (
        LexicalTerm("design", "ontology:backward_design", "canonical_id"),
        LexicalTerm("backward design", "ontology:backward_design", "alias"),
    )

    assert phrases_for_question("how do I design a course", terms) == ()


def test_build_lexical_query_keeps_the_original_words_and_adds_phrases() -> None:
    bare = build_lexical_query("what is backward design", ())
    quoted = build_lexical_query("what is backward design", ("backward design",))

    assert isinstance(bare, MatchQuery)
    assert bare.query == "what is backward design"
    assert isinstance(quoted, BooleanQuery)
    phrase_queries = [clause for _, clause in quoted.queries if isinstance(clause, PhraseQuery)]
    assert [clause.query for clause in phrase_queries] == ["backward design"]


def test_load_lexical_terms_reads_glossary_aliases(tmp_path: Path) -> None:
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
    config = SimpleNamespace(
        paths=SimpleNamespace(corpus_path=wiki, ontology_path=tmp_path / "missing-ontology")
    )

    terms = load_lexical_terms(cast("RuntimeConfig", config))
    phrases = phrases_for_question("what does WMC limit", terms)

    assert "working memory capacity" in phrases
