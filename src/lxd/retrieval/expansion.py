"""Expand user questions with ontology-aware rewrite terms."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lxd.ingest.mentions import detect_mentions
from lxd.ontology.graph import OntologyGraph
from lxd.ontology.loader import OntologyLoadResult, load_ontology
from lxd.ontology.matcher import build_automaton
from lxd.settings.models import RuntimeConfig
from lxd.stores.sqlite import (
    build_store_paths,
    connect_sqlite,
    load_corpus_related_entity_ids,
)

_ONTOLOGY_CACHE: dict[tuple[str, tuple[str, ...], tuple[str, ...]], _OntologyRuntime] = {}


@dataclass(frozen=True)
class ExpansionOutcome:
    """Query expansion text plus matched ontology entities."""

    expanded_question: str
    matched_entity_ids: list[str]
    added_terms: list[str]


@dataclass(frozen=True)
class _OntologyRuntime:
    ontology: OntologyLoadResult
    automaton: object
    entity_by_id: dict[str, dict[str, object]]


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
    """Expand a question with ontology-related concept terms.

    Args:
        question: User question text.
        config: Runtime configuration object.

    Returns:
        Expanded query text and added ontology terms.
    """
    runtime = _ontology_runtime(config)
    mentions = detect_mentions(question, runtime.automaton)
    matched_entity_ids = _dedupe([mention.entity_id for mention in mentions])
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
        connection = connect_sqlite(store_paths.sqlite_path)
        try:
            return load_corpus_related_entity_ids(
                connection,
                entity_ids,
                max_results=config.expansion.max_terms,
            )
        finally:
            connection.close()
    except Exception:
        return []


def _ontology_runtime(config: RuntimeConfig) -> _OntologyRuntime:
    cache_key = (
        str(config.paths.ontology_path),
        tuple(config.ontology.include_globs),
        tuple(config.ontology.ignore_names),
    )
    cached = _ONTOLOGY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ontology = load_ontology(
        root=config.paths.ontology_path,
        include_globs=config.ontology.include_globs,
        ignore_names=config.ontology.ignore_names,
    )
    runtime = _OntologyRuntime(
        ontology=ontology,
        automaton=build_automaton(ontology.matcher_records),
        entity_by_id={entity["canonical_id"]: entity for entity in ontology.entity_definitions},
    )
    _ONTOLOGY_CACHE[cache_key] = runtime
    return runtime


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
        if depth >= hops:
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
    candidates: list[str] = []
    canonical_id = entity.get("canonical_id")
    label = entity.get("label")
    aliases = entity.get("aliases")
    if isinstance(canonical_id, str) and canonical_id.strip():
        candidates.append(canonical_id)
    if isinstance(label, str) and label.strip():
        candidates.append(label)
    if isinstance(aliases, list):
        candidates.extend(item for item in aliases if isinstance(item, str) and item.strip())
    return _dedupe(candidates)


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
