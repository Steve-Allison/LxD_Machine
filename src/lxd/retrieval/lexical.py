"""Build the BM25 query from wiki and ontology phrases.

The dense lane keeps the literal question, or the HyDE passage when that
lane fires. This module only shapes the text lane: a bag-of-words match
plus a phrase clause for each multi-word term the question actually
names. A short glossary alias such as ``WMC`` also pulls in the longest
multi-word phrase that shares its owner, so the keyword lane can find
``working memory capacity``.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from lancedb.query import (
    BooleanQuery,
    FullTextOperator,
    FullTextQuery,
    MatchQuery,
    Occur,
    PhraseQuery,
)

from lxd.ingest.wiki_metadata import load_wiki_matcher_phrases
from lxd.ontology.loader import load_configured_ontology
from lxd.ontology.normalization import normalize_match_text
from lxd.settings.models import RuntimeConfig

_TEXT_COLUMN = "text"
_ALIAS_SOURCES = frozenset({"wiki_glossary", "wiki_owns", "alias"})
_MIN_ALIAS_CHARS = 2


@dataclass(frozen=True, slots=True)
class LexicalTerm:
    """One surface form that may be quoted in the BM25 query."""

    normalized: str
    owner: str
    term_source: str


_TERM_CACHE: dict[tuple[str, str, int, int], tuple[LexicalTerm, ...]] = {}


def load_lexical_terms(config: RuntimeConfig) -> tuple[LexicalTerm, ...]:
    """Return wiki and ontology terms used to recognise phrases in a question.

    Cached until the wiki markdown or ontology YAML mtimes change. A
    missing corpus or ontology directory contributes no terms.
    """
    stamp = _term_stamp(config)
    cached = _TERM_CACHE.get(stamp)
    if cached is not None:
        return cached
    terms = (*_wiki_terms(config.paths.corpus_path), *_ontology_terms(config))
    _TERM_CACHE.clear()
    _TERM_CACHE[stamp] = terms
    return terms


def phrases_for_question(
    question: str, terms: tuple[LexicalTerm, ...] | list[LexicalTerm]
) -> tuple[str, ...]:
    """Return multi-word phrases the BM25 query should search as phrases.

    Longer phrases win when they overlap. A single-word glossary alias,
    owns phrase, or ontology alias also contributes the longest
    multi-word phrase of the same owner, unless that token already sits
    inside a phrase the question itself contains.
    """
    haystack = f" {normalize_match_text(question)} "
    occupied: list[tuple[int, int]] = []
    quoted: list[str] = []
    multiword = sorted(
        {term.normalized for term in terms if " " in term.normalized},
        key=len,
        reverse=True,
    )
    for phrase in multiword:
        span = _free_span(haystack, phrase, occupied)
        if span is None:
            continue
        occupied.append(span)
        quoted.append(phrase)

    by_owner: dict[str, list[LexicalTerm]] = defaultdict(list)
    for term in terms:
        by_owner[term.owner].append(term)
    already = set(quoted)
    for group in by_owner.values():
        expansion = _alias_expansion(haystack, group, occupied)
        if expansion and expansion not in already:
            quoted.append(expansion)
            already.add(expansion)
    return tuple(quoted)


def build_lexical_query(question: str, phrases: tuple[str, ...] | list[str]) -> FullTextQuery:
    """Bag-of-words match, plus one phrase clause per recognised term.

    Both clauses are optional. A chunk that contains the exact phrase
    scores the phrase and the original words. A chunk that only contains
    the words still matches.
    """
    match = MatchQuery(query=question.strip(), column=_TEXT_COLUMN, operator=FullTextOperator.OR)
    if not phrases:
        return match
    clauses: list[tuple[Occur, FullTextQuery]] = [(Occur.SHOULD, match)]
    clauses.extend(
        (Occur.SHOULD, PhraseQuery(query=phrase, column=_TEXT_COLUMN)) for phrase in phrases
    )
    return BooleanQuery(clauses)


def _alias_expansion(
    haystack: str,
    group: list[LexicalTerm],
    occupied: list[tuple[int, int]],
) -> str | None:
    singles = [
        term.normalized
        for term in group
        if term.term_source in _ALIAS_SOURCES
        and " " not in term.normalized
        and len(term.normalized) >= _MIN_ALIAS_CHARS
    ]
    if not any(_alias_outside_phrases(haystack, token, occupied) for token in singles):
        return None
    multiword = [term.normalized for term in group if " " in term.normalized]
    if not multiword:
        return None
    return max(multiword, key=len)


def _alias_outside_phrases(haystack: str, token: str, occupied: list[tuple[int, int]]) -> bool:
    needle = f" {token} "
    start = haystack.find(needle)
    while start >= 0:
        span = (start, start + len(needle))
        if not _overlaps(span, occupied):
            return True
        start = haystack.find(needle, start + 1)
    return False


def _free_span(
    haystack: str, phrase: str, occupied: list[tuple[int, int]]
) -> tuple[int, int] | None:
    needle = f" {phrase} "
    start = haystack.find(needle)
    while start >= 0:
        span = (start, start + len(needle))
        if not _overlaps(span, occupied):
            return span
        start = haystack.find(needle, start + 1)
    return None


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and span[1] > begin for begin, end in occupied)


def _wiki_terms(corpus_root: Path) -> tuple[LexicalTerm, ...]:
    terms: list[LexicalTerm] = []
    for phrase in load_wiki_matcher_phrases(corpus_root):
        normalized = normalize_match_text(phrase.term)
        if not normalized:
            continue
        terms.append(
            LexicalTerm(
                normalized=normalized,
                owner=f"wiki:{phrase.slug}",
                term_source=phrase.term_source,
            )
        )
    return tuple(terms)


def _ontology_terms(config: RuntimeConfig) -> tuple[LexicalTerm, ...]:
    if not config.paths.ontology_path.is_dir():
        return ()
    loaded = load_configured_ontology(config)
    terms: list[LexicalTerm] = []
    for record in loaded.matcher_records:
        if not record.normalized_term:
            continue
        terms.append(
            LexicalTerm(
                normalized=record.normalized_term,
                owner=f"ontology:{record.entity_id}",
                term_source=record.term_source,
            )
        )
    return tuple(terms)


def _term_stamp(config: RuntimeConfig) -> tuple[str, str, int, int]:
    corpus = config.paths.corpus_path
    ontology = config.paths.ontology_path
    return (
        str(corpus),
        str(ontology),
        _newest_mtime(corpus, ".md"),
        _newest_mtime(ontology, ".yaml"),
    )


def _newest_mtime(root: Path, suffix: str) -> int:
    if not root.is_dir():
        return 0
    newest = root.stat().st_mtime_ns
    for path in root.rglob(f"*{suffix}"):
        try:
            newest = max(newest, path.stat().st_mtime_ns)
        except OSError:
            continue
    return newest
