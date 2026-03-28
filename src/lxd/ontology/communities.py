"""Community detection on the combined entity graph."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import networkx as nx
import structlog

from lxd.settings.models import RuntimeConfig
from lxd.stores.models import EntityCommunityRecord
from lxd.stores.sqlite import replace_entity_communities

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CommunityDetectionResult:
    """Result of community detection."""

    algorithm: str
    resolution: float
    seed: int
    community_count: int
    assignments: dict[str, int]


def detect_communities(
    graph: nx.MultiDiGraph,
    config: RuntimeConfig,
) -> CommunityDetectionResult:
    """Partition entities into communities using the configured algorithm.

    Supports Leiden (via graspologic, requires undirected conversion) and
    Louvain (via NetworkX, supports directed graphs natively).
    """
    kg_cfg = config.knowledge_graph
    algorithm = kg_cfg.community_algorithm
    resolution = kg_cfg.community_resolution
    seed = kg_cfg.community_seed

    if graph.number_of_nodes() == 0:
        return CommunityDetectionResult(
            algorithm=algorithm,
            resolution=resolution,
            seed=seed,
            community_count=0,
            assignments={},
        )

    if algorithm == "leiden":
        assignments = _detect_leiden(graph, resolution=resolution, seed=seed)
    else:
        assignments = _detect_louvain(graph, resolution=resolution, seed=seed)

    community_count = len(set(assignments.values())) if assignments else 0
    _log.info(
        "community detection complete",
        algorithm=algorithm,
        resolution=resolution,
        communities=community_count,
        entities=len(assignments),
    )

    return CommunityDetectionResult(
        algorithm=algorithm,
        resolution=resolution,
        seed=seed,
        community_count=community_count,
        assignments=assignments,
    )


def persist_community_assignments(
    connection: sqlite3.Connection,
    assignments: dict[str, int],
) -> int:
    """Write community assignments to SQLite.

    Returns:
        Number of assignments written.
    """
    timestamp = datetime.now(UTC).isoformat()
    records = [
        EntityCommunityRecord(
            entity_id=entity_id,
            community_id=community_id,
            community_level=0,
            modularity_class=None,
            assigned_at=timestamp,
        )
        for entity_id, community_id in assignments.items()
    ]
    replace_entity_communities(connection, records)
    return len(records)


def _detect_leiden(
    graph: nx.MultiDiGraph,
    *,
    resolution: float,
    seed: int,
) -> dict[str, int]:
    """Run Leiden community detection via graspologic.

    Leiden does NOT support directed graphs — must convert to undirected.
    """
    try:
        from graspologic.partition import leiden  # type: ignore[import-not-found]
    except ImportError:
        _log.warning("graspologic not installed, falling back to Louvain")
        return _detect_louvain(graph, resolution=resolution, seed=seed)

    undirected = graph.to_undirected()
    # Convert MultiGraph to simple Graph for graspologic
    simple = nx.Graph(undirected)

    node_to_community = leiden(simple, resolution=resolution, random_seed=seed)

    return {str(node): int(community) for node, community in node_to_community.items()}


def _detect_louvain(
    graph: nx.MultiDiGraph,
    *,
    resolution: float,
    seed: int,
) -> dict[str, int]:
    """Run Louvain community detection via NetworkX.

    Louvain supports directed graphs natively (Directed Louvain modularity).
    """
    # Convert to DiGraph (Louvain doesn't support MultiDiGraph directly)
    simple = nx.DiGraph(graph)

    communities = nx.community.louvain_communities(
        simple,
        resolution=resolution,
        seed=seed,
    )

    assignments: dict[str, int] = {}
    for community_id, members in enumerate(communities):
        for node in members:
            assignments[str(node)] = community_id

    return assignments
