"""Build combined entity graph from ontology + corpus relations and compute centrality."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx
import structlog

from lxd.ontology.graph import OntologyGraph
from lxd.settings.models import RuntimeConfig
from lxd.stores.sqlite import load_all_extracted_relations

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CentralityScores:
    """Centrality metric values for one entity."""

    entity_id: str
    pagerank: float
    betweenness: float
    closeness: float
    in_degree: int
    out_degree: int
    eigenvector: float


@dataclass(frozen=True)
class EntityGraphResult:
    """Result of building the combined entity graph."""

    graph: nx.MultiDiGraph
    centrality: dict[str, CentralityScores]
    node_count: int
    edge_count: int


def build_combined_entity_graph(
    ontology_graph: OntologyGraph,
    connection: sqlite3.Connection,
    config: RuntimeConfig,
) -> EntityGraphResult:
    """Build combined entity graph from ontology edges + corpus relations.

    Loads extracted_relations, deduplicates by (subject, predicate, object),
    adds edges to the ontology graph, and computes all centrality metrics.
    """
    graph: nx.MultiDiGraph = ontology_graph.copy()  # type: ignore[assignment]
    min_confidence = config.knowledge_graph.min_relation_confidence

    # Load and deduplicate corpus relations
    extracted = load_all_extracted_relations(connection)
    corpus_edges_added = 0

    if extracted:
        groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in extracted:
            if row.confidence >= min_confidence:
                key = (row.subject_entity_id, row.predicate, row.object_entity_id)
                groups[key].append(row.confidence)

        for (subject, predicate, obj), confidences in groups.items():
            # Ensure nodes exist
            if subject not in graph:
                graph.add_node(
                    subject, node_type="entity", entity_id=subject, label=subject, metadata={}
                )
            if obj not in graph:
                graph.add_node(obj, node_type="entity", entity_id=obj, label=obj, metadata={})

            graph.add_edge(
                subject,
                obj,
                relation_type=predicate,
                origin_kind="corpus",
                weight=max(confidences),
                support_count=len(confidences),
            )
            corpus_edges_added += 1

    _log.info(
        "combined entity graph built",
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        corpus_edges_added=corpus_edges_added,
    )

    # Compute centrality metrics
    centrality = _compute_centrality(graph)

    return EntityGraphResult(
        graph=graph,
        centrality=centrality,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
    )


def _compute_centrality(graph: nx.MultiDiGraph) -> dict[str, CentralityScores]:
    """Compute all 6 centrality metrics on the directed MultiDiGraph."""
    if graph.number_of_nodes() == 0:
        return {}

    # PageRank — works natively on MultiDiGraph
    _log.info("computing PageRank")
    pagerank = nx.pagerank(graph)

    # Betweenness — unweighted (float weights are documented as unreliable)
    _log.info("computing betweenness centrality (unweighted)")
    betweenness = nx.betweenness_centrality(graph, weight=None)

    # Closeness — uses inward distance on directed graphs
    _log.info("computing closeness centrality")
    closeness = nx.closeness_centrality(graph)

    # Degree — raw counts, not normalised
    in_degrees: dict[str, int] = dict(graph.in_degree())
    out_degrees: dict[str, int] = dict(graph.out_degree())

    # Eigenvector — use NumPy/LAPACK solver for guaranteed convergence
    _log.info("computing eigenvector centrality (numpy)")
    try:
        eigenvector: dict[Any, float] = nx.eigenvector_centrality_numpy(graph)
    except Exception as exc:
        _log.warning("eigenvector centrality failed, using zeros: %s", exc)
        eigenvector = {node: 0.0 for node in graph.nodes()}

    result: dict[str, CentralityScores] = {}
    for node in graph.nodes():
        node_id = str(node)
        result[node_id] = CentralityScores(
            entity_id=node_id,
            pagerank=float(pagerank.get(node, 0.0)),
            betweenness=float(betweenness.get(node, 0.0)),
            closeness=float(closeness.get(node, 0.0)),
            in_degree=int(in_degrees.get(node, 0)),
            out_degree=int(out_degrees.get(node, 0)),
            eigenvector=float(eigenvector.get(node, 0.0)),
        )

    _log.info("centrality computation complete", entities=len(result))
    return result
