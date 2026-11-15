"""Expand user questions with ``Central_Configs`` v5.0.0 ontology-aware rewrite terms.

Lexical expansion emits ONLY human-readable ``pref_label`` and
``all_terms.written_form`` strings. Opaque ``coe:Cxxxxx`` CURIEs are returned
exclusively in ``matched_entity_ids`` for structured chunk/graph filtering and
are never injected into BM25 or dense rewrite text.
"""

import re
import sqlite3
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from lxd.ingest.mentions import detect_mentions
from lxd.ontology.disambiguator import DisambiguationCandidate, build_entity_catalog
from lxd.ontology.graph import OntologyGraph
from lxd.ontology.loader import OntologyLoadResult, resolve_central_duckdb_path
from lxd.ontology.matcher import build_or_load_automaton
from lxd.ontology.normalization import normalize_match_text
from lxd.settings.models import RuntimeConfig
from lxd.stores.sqlite._pool import pooled_connection
from lxd.stores.sqlite.chunks import load_corpus_related_entity_ids
from lxd.stores.sqlite.connection import build_store_paths

_CURIE_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*:C\d+$")


@dataclass(frozen=True, slots=True)
class ExpansionOutcome:
    """Query expansion text plus matched ontology entities."""

    expanded_question: str
    matched_entity_ids: list[str]
    added_terms: list[str]


@dataclass(frozen=True, slots=True)
class _ConceptSurface:
    phrase: str
    normalized: str
    entity_id: str
    case_exact: bool


@dataclass(frozen=True, slots=True)
class _OntologyRuntime:
    ontology: OntologyLoadResult
    automaton: object
    entity_by_id: dict[str, dict[str, object]]
    concept_surfaces: tuple[_ConceptSurface, ...] = ()
    entity_catalog: Mapping[str, DisambiguationCandidate] | None = None


def expand_entity_ids(
    graph: OntologyGraph,
    seed_entity_ids: list[str],
    *,
    hops: int = 1,
    max_entities: int = 50,
) -> list[str]:
    """Return related entity IDs reachable within `hops` from `seed_entity_ids`."""
    return _expand_entity_ids(graph, seed_entity_ids, hops=hops, max_entities=max_entities)


def expand_question(question: str, config: RuntimeConfig) -> ExpansionOutcome:
    """Expand a question with ontology-related human-readable concept terms.

    Args:
        question: User question text.
        config: Runtime configuration object.

    Returns:
        Expanded query text, matched ``coe:Cxxxxx`` entity IDs, and lexical terms.
    """
    runtime = _ontology_runtime(config)
    if runtime is None:
        return ExpansionOutcome(
            expanded_question=question,
            matched_entity_ids=[],
            added_terms=[],
        )
    mentions = detect_mentions(
        question,
        runtime.automaton,
        entity_catalog=getattr(runtime, "entity_catalog", None),
    )
    concept_ids = _match_concept_surfaces(
        question,
        getattr(runtime, "concept_surfaces", ()),
    )
    matched_entity_ids = _dedupe(
        [*[mention.entity_id for mention in mentions], *concept_ids]
    )
    if not matched_entity_ids:
        return ExpansionOutcome(
            expanded_question=question,
            matched_entity_ids=[],
            added_terms=[],
        )
    related_entity_ids = _expand_entity_ids(
        runtime.ontology.graph,
        matched_entity_ids,
        hops=config.expansion.hops,
        max_entities=config.expansion.max_terms,
    )
    corpus_related_ids = _expand_from_corpus(config, matched_entity_ids)
    all_related_ids = _dedupe([*related_entity_ids, *corpus_related_ids])
    added_terms = _terms_for_entities(
        runtime.entity_by_id,
        entity_ids=[*all_related_ids, *matched_entity_ids],
        question=question,
        max_terms=config.expansion.max_terms,
    )
    if not added_terms:
        return ExpansionOutcome(
            expanded_question=question,
            matched_entity_ids=matched_entity_ids,
            added_terms=[],
        )
    return ExpansionOutcome(
        expanded_question=f"{question}\n\nRelated concepts: {'; '.join(added_terms)}",
        matched_entity_ids=matched_entity_ids,
        added_terms=added_terms,
    )


def _expand_from_corpus(config: RuntimeConfig, entity_ids: list[str]) -> list[str]:
    """Return entity IDs related to `entity_ids` via extracted corpus relations.

    Returns an empty list silently if the store doesn't exist yet.
    """
    store_paths = build_store_paths(config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    try:
        with pooled_connection(store_paths.sqlite_path) as connection:
            return load_corpus_related_entity_ids(
                connection,
                entity_ids,
                max_results=config.expansion.max_terms,
            )
    except sqlite3.DatabaseError:
        return []


@lru_cache(maxsize=4)
def _build_ontology_runtime(
    db_path_str: str,
    db_mtime_ns: int,
    data_path_str: str,
) -> _OntologyRuntime:
    from lxd.ontology.loader import load_ontology

    _ = db_mtime_ns
    ontology = load_ontology(Path(db_path_str))
    proper_noun_curies = {rule.curie for rule in ontology.matcher_records}
    concept_surfaces = _build_concept_surfaces(ontology.entity_definitions, proper_noun_curies)
    return _OntologyRuntime(
        ontology=ontology,
        automaton=build_or_load_automaton(
            ontology.matcher_records,
            cache_dir=Path(data_path_str) / "matcher_cache",
        ),
        entity_by_id={entity["canonical_id"]: entity for entity in ontology.entity_definitions},
        concept_surfaces=concept_surfaces,
        entity_catalog=build_entity_catalog(ontology.entity_definitions),
    )


def _ontology_runtime(config: RuntimeConfig) -> _OntologyRuntime | None:
    paths = getattr(config, "paths", None)
    if paths is None:
        return None
    db_path_attr = getattr(paths, "central_duckdb_path", None)
    if db_path_attr is not None:
        db_path = Path(db_path_attr)
    else:
        central_root = getattr(paths, "central_configs_path", None)
        if central_root is None:
            return None
        db_path = resolve_central_duckdb_path(Path(central_root))
    if not db_path.is_file():
        return None
    return _build_ontology_runtime(
        str(db_path),
        db_path.stat().st_mtime_ns,
        str(paths.data_path),
    )


def _build_concept_surfaces(
    entity_definitions: list[dict[str, Any]],
    proper_noun_curies: set[str],
) -> tuple[_ConceptSurface, ...]:
    surfaces: list[_ConceptSurface] = []
    seen: set[tuple[str, str]] = set()
    for entity in entity_definitions:
        curie = entity.get("canonical_id")
        if not isinstance(curie, str) or not curie or curie in proper_noun_curies:
            continue
        for candidate in _entity_term_candidates(entity):
            normalized = normalize_match_text(candidate)
            if len(normalized) < 3:
                continue
            key = (normalized, curie)
            if key in seen:
                continue
            seen.add(key)
            case_exact = (
                " " not in candidate.strip()
                and len(candidate.strip()) <= 5
                and candidate.strip().isupper()
            )
            surfaces.append(
                _ConceptSurface(
                    phrase=candidate.strip(),
                    normalized=normalized,
                    entity_id=curie,
                    case_exact=case_exact,
                )
            )
    surfaces.sort(key=lambda item: (-len(item.normalized), item.entity_id))
    return tuple(surfaces)


def _match_concept_surfaces(
    question: str,
    surfaces: tuple[_ConceptSurface, ...],
) -> list[str]:
    if not surfaces:
        return []
    haystack = f" {normalize_match_text(question)} "
    occupied: list[tuple[int, int]] = []
    matched: list[str] = []
    for surface in surfaces:
        if surface.case_exact and not _contains_exact_token(question, surface.phrase):
            continue
        needle = f" {surface.normalized} "
        start = haystack.find(needle)
        while start >= 0:
            span = (start, start + len(needle))
            if not any(span[0] < end and span[1] > begin for begin, end in occupied):
                occupied.append(span)
                matched.append(surface.entity_id)
                break
            start = haystack.find(needle, start + 1)
    return matched


def _contains_exact_token(text: str, token: str) -> bool:
    start = text.find(token)
    while start >= 0:
        end = start + len(token)
        left_ok = start == 0 or not text[start - 1].isalnum()
        right_ok = end == len(text) or not text[end].isalnum()
        if left_ok and right_ok:
            return True
        start = text.find(token, start + 1)
    return False


def _expand_entity_ids(
    graph: OntologyGraph,
    seed_entity_ids: list[str],
    *,
    hops: int,
    max_entities: int,
) -> list[str]:
    if hops < 1 or max_entities < 1:
        return []
    visited = set(seed_entity_ids)
    queued = deque((entity_id, 0) for entity_id in seed_entity_ids)
    expanded: list[str] = []
    while queued and len(expanded) < max_entities:
        node_id, depth = queued.popleft()
        if depth >= hops or node_id not in graph:
            continue
        neighbors = set(graph.successors(node_id)) | set(graph.predecessors(node_id))
        for neighbor in sorted(neighbors):
            if neighbor in visited:
                continue
            neighbor_data = graph.nodes[neighbor]
            if neighbor_data.get("node_type") != "entity":
                continue
            visited.add(neighbor)
            expanded.append(str(neighbor))
            queued.append((str(neighbor), depth + 1))
            if len(expanded) >= max_entities:
                break
    return expanded


def _terms_for_entities(
    entity_by_id: dict[str, dict[str, object]],
    *,
    entity_ids: list[str],
    question: str,
    max_terms: int,
) -> list[str]:
    question_folded = question.casefold()
    terms: list[str] = []
    seen: set[str] = set()
    for entity_id in entity_ids:
        entity = entity_by_id.get(entity_id)
        if entity is None:
            continue
        for candidate in _entity_term_candidates(entity):
            folded = candidate.casefold()
            if folded in seen or folded in question_folded:
                continue
            seen.add(folded)
            terms.append(candidate)
            if len(terms) >= max_terms:
                return terms
    return terms


def _entity_term_candidates(entity: dict[str, Any]) -> list[str]:
    """Return strictly lexical expansion strings (``pref_label`` and ``written_form``).

    Opaque ``coe:Cxxxxx`` CURIEs are never emitted into lexical expansion terms.
    """
    candidates: list[str] = []
    label = entity.get("label")
    aliases = entity.get("aliases")
    if isinstance(label, str) and label.strip() and not _is_curie(label.strip()):
        candidates.append(label.strip())
    if isinstance(aliases, list):
        for item in aliases:
            if isinstance(item, str) and item.strip() and not _is_curie(item.strip()):
                candidates.append(item.strip())
    return _dedupe(candidates)


def _is_curie(value: str) -> bool:
    return bool(_CURIE_RE.match(value))


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        folded = value.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        result.append(value)
    return result
