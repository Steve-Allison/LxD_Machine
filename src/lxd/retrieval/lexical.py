"""Build the BM25 query from wiki and ``Central_Configs`` v5.0.0 phrases.

The dense lane keeps the literal question, or the HyDE passage when that
lane fires. This module only shapes the text lane: a bag-of-words match
plus a phrase clause for each multi-word term the question actually
names, plus optional ``pref_label`` / ``all_terms.written_form`` lexical
expansion terms. Opaque ``coe:Cxxxxx`` CURIEs are never emitted into
BM25 queries.
"""

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from lancedb.query import (
    BooleanQuery,
    FullTextOperator,
    FullTextQuery,
    MatchQuery,
    Occur,
    PhraseQuery,
)

from lxd.ingest.wiki_metadata import load_wiki_matcher_phrases
from lxd.ontology.loader import load_ontology, resolve_central_duckdb_path
from lxd.ontology.normalization import normalize_match_text
from lxd.settings.models import RuntimeConfig

_TEXT_COLUMN: Final = "text"
_ALIAS_SOURCES: Final = frozenset({"wiki_glossary", "wiki_owns", "alias"})
_MIN_ALIAS_CHARS: Final = 2
_CURIE_TOKEN_RE: Final = re.compile(r"\b[A-Za-z][A-Za-z0-9_.-]*:C\d+\b")


@dataclass(frozen=True, slots=True)
class LexicalTerm:
    """One surface form that may be quoted in the BM25 query."""

    normalized: str
    owner: str
    term_source: str


_TERM_CACHE: dict[tuple[str, str, int, int], tuple[LexicalTerm, ...]] = {}


def load_lexical_terms(config: RuntimeConfig) -> tuple[LexicalTerm, ...]:
    """Return wiki and ``Central_Configs`` terms used to recognise phrases in a question.

    Cached until the wiki markdown or ``central.duckdb`` mtime changes. A
    missing corpus directory or ``central.duckdb`` file contributes no terms.
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


def build_lexical_query(
    question: str,
    phrases: tuple[str, ...] | list[str],
    expansion_terms: Sequence[str] = (),
) -> FullTextQuery:
    """Bag-of-words match, plus phrase clauses and human-readable lexical expansions.

    Opaque ``coe:Cxxxxx`` CURIEs are stripped defensively so structured IDs
    never pollute lexical BM25 scoring.
    """
    clean_question = _strip_curies(question).strip() or question.strip()
    match = MatchQuery(
        query=clean_question,
        column=_TEXT_COLUMN,
        boost=1.0,
        fuzziness=0,
        max_expansions=50,
        operator=FullTextOperator.OR,
        prefix_length=0,
        document_granularity=None,
    )
    clean_phrases = [
        cleaned
        for phrase in phrases
        if (cleaned := _strip_curies(phrase).strip())
    ]
    clean_expansion = [
        cleaned
        for term in expansion_terms
        if (cleaned := _strip_curies(term).strip())
    ]
    if not clean_phrases and not clean_expansion:
        return match
    clauses: list[tuple[Occur, FullTextQuery]] = [(Occur.SHOULD, match)]
    if clean_expansion:
        clauses.append(
            (
                Occur.SHOULD,
                MatchQuery(
                    query=" ".join(clean_expansion),
                    column=_TEXT_COLUMN,
                    boost=0.5,
                    fuzziness=0,
                    max_expansions=50,
                    operator=FullTextOperator.OR,
                    prefix_length=0,
                    document_granularity=None,
                ),
            )
        )
    clauses.extend(
        (
            Occur.SHOULD,
            PhraseQuery(query=phrase, column=_TEXT_COLUMN, slop=0, document_granularity=None),
        )
        for phrase in clean_phrases
    )
    return BooleanQuery(clauses)


def _strip_curies(text: str) -> str:
    return " ".join(_CURIE_TOKEN_RE.sub(" ", text).split())


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


def _resolve_db_path(config: RuntimeConfig) -> Path | None:
    paths = getattr(config, "paths", None)
    if paths is None:
        return None
    db_path_attr = getattr(paths, "central_duckdb_path", None)
    if db_path_attr is not None:
        return Path(db_path_attr)
    central_root = getattr(paths, "central_configs_path", None)
    if central_root is not None:
        return resolve_central_duckdb_path(Path(central_root))
    return None


def _ontology_terms(config: RuntimeConfig) -> tuple[LexicalTerm, ...]:
    db_path = _resolve_db_path(config)
    if db_path is None or not db_path.is_file():
        return ()
    loaded = load_ontology(db_path)
    terms: list[LexicalTerm] = []
    seen: set[tuple[str, str, str]] = set()
    for entity in loaded.entity_definitions:
        curie = str(entity.get("canonical_id") or "").strip()
        label = str(entity.get("label") or "").strip()
        if not curie or not label or _CURIE_TOKEN_RE.search(label):
            continue
        normalized = normalize_match_text(label)
        if not normalized:
            continue
        owner = f"ontology:{curie}"
        key = (normalized, owner, "label")
        if key not in seen:
            seen.add(key)
            terms.append(LexicalTerm(normalized=normalized, owner=owner, term_source="label"))
    for term in loaded.terms:
        if not term.resource_id or not term.written_form or _CURIE_TOKEN_RE.search(term.written_form):
            continue
        normalized = normalize_match_text(term.written_form)
        if not normalized:
            continue
        owner = f"ontology:{term.resource_id}"
        key = (normalized, owner, "alias")
        if key not in seen:
            seen.add(key)
            terms.append(LexicalTerm(normalized=normalized, owner=owner, term_source="alias"))
    return tuple(terms)


def _term_stamp(config: RuntimeConfig) -> tuple[str, str, int, int]:
    corpus = config.paths.corpus_path
    db_path = _resolve_db_path(config)
    db_mtime = db_path.stat().st_mtime_ns if db_path is not None and db_path.is_file() else 0
    return (
        str(corpus),
        str(db_path) if db_path is not None else "",
        _newest_mtime(corpus, ".md"),
        db_mtime,
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
