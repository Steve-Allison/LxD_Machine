"""Community detection on the combined entity graph.

Two scales of partitioning:

  - **Single-level**: the original Louvain pass at the configured resolution.
    Backwards-compatible; callers that only need flat communities use
    :func:`detect_communities` + :func:`persist_community_assignments`.
  - **Hierarchical (GraphRAG-style)**: multi-resolution Louvain produces a
    parent-child tree of communities, with the finest level at ``level=0``
    and coarser levels at increasing ``level`` numbers.
    :func:`detect_hierarchical_communities` returns a stack of
    :class:`CommunityDetectionResult` ordered finest → coarsest; each result
    carries its ``parent_community_id`` map keyed by ``community_id`` of the
    level below. ``persist_hierarchical_communities`` writes every level.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise

import networkx as nx
import structlog

from lxd.settings.models import RuntimeConfig
from lxd.stores.models import EntityCommunityRecord
from lxd.stores.sqlite.kg_profiles import replace_entity_communities

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CommunityDetectionResult:
    """Result of community detection at a single resolution / level."""

    algorithm: str
    resolution: float
    seed: int
    community_count: int
    assignments: dict[str, int]
    community_level: int = 0
    # parent_of[community_id] = community_id at the level above, or None for the top level.
    parent_of: dict[int, int | None] = field(default_factory=dict)


def detect_communities(
    graph: nx.MultiDiGraph,
    config: RuntimeConfig,
) -> CommunityDetectionResult:
    """Partition entities into communities at the configured resolution.

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

    assignments = _partition(graph, algorithm=algorithm, resolution=resolution, seed=seed)
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


def detect_hierarchical_communities(
    graph: nx.MultiDiGraph,
    config: RuntimeConfig,
    *,
    resolutions: tuple[float, ...] | None = None,
) -> list[CommunityDetectionResult]:
    """Run multi-resolution community detection to produce a community hierarchy.

    Args:
        graph: The combined entity graph.
        config: Runtime config — uses ``knowledge_graph.community_algorithm``,
            ``community_seed``, and (as level 0) ``community_resolution``.
        resolutions: Optional explicit resolution stack ordered finest →
            coarsest. Defaults to ``(community_resolution, r/2, r/4)`` for a
            three-level hierarchy. Resolutions must be strictly decreasing —
            higher = more, smaller communities.

    Returns:
        Stack of :class:`CommunityDetectionResult`, finest (level 0) first.
        Each carries ``community_level`` and ``parent_of`` mapping
        ``community_id at level → community_id at level + 1``. The top
        level's ``parent_of`` values are all ``None``.
    """
    kg_cfg = config.knowledge_graph
    algorithm = kg_cfg.community_algorithm
    seed = kg_cfg.community_seed
    if resolutions is None:
        base = kg_cfg.community_resolution
        resolutions = (base, base / 2.0, base / 4.0)
    if any(a <= b for a, b in pairwise(resolutions)):
        raise ValueError(
            f"resolutions must be strictly decreasing (finest first); got {resolutions}."
        )

    if graph.number_of_nodes() == 0:
        return []

    # First pass: partition at every resolution. Parent links computed below.
    per_level_assignments: list[dict[str, int]] = []
    for level_index, resolution in enumerate(resolutions):
        assignments = _partition(graph, algorithm=algorithm, resolution=resolution, seed=seed)
        per_level_assignments.append(assignments)
        _log.info(
            "hierarchical community level",
            level=level_index,
            resolution=resolution,
            communities=len(set(assignments.values())) if assignments else 0,
            entities=len(assignments),
        )

    # Second pass: for level N, parent_of maps our communities to the level-N+1
    # community that contains a majority of each. The coarsest level (last)
    # has no parent — its parent_of stays empty.
    levels: list[CommunityDetectionResult] = []
    for level_index, assignments in enumerate(per_level_assignments):
        if level_index + 1 < len(per_level_assignments):
            parent_of = _majority_parent(assignments, per_level_assignments[level_index + 1])
        else:
            parent_of = {}
        community_count = len(set(assignments.values())) if assignments else 0
        levels.append(
            CommunityDetectionResult(
                algorithm=algorithm,
                resolution=resolutions[level_index],
                seed=seed,
                community_count=community_count,
                assignments=assignments,
                community_level=level_index,
                parent_of=parent_of,
            )
        )

    return levels


def persist_community_assignments(
    connection: sqlite3.Connection,
    assignments: dict[str, int],
) -> int:
    """Write single-level (level 0) community assignments to SQLite.

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


def persist_hierarchical_communities(
    connection: sqlite3.Connection,
    levels: list[CommunityDetectionResult],
) -> int:
    """Write a hierarchical partition stack to SQLite.

    Truncates ``entity_communities`` and writes every level in one batch.
    The downstream community-report builder reads each level and writes one
    :class:`CommunityReportRecord` per ``(community_id, community_level)``.

    Returns:
        Total number of (entity, level) assignments written.
    """
    timestamp = datetime.now(UTC).isoformat()
    records: list[EntityCommunityRecord] = []
    for level in levels:
        for entity_id, community_id in level.assignments.items():
            records.append(
                EntityCommunityRecord(
                    entity_id=entity_id,
                    community_id=community_id,
                    community_level=level.community_level,
                    modularity_class=None,
                    assigned_at=timestamp,
                )
            )
    replace_entity_communities(connection, records)
    return len(records)


def _partition(
    graph: nx.MultiDiGraph,
    *,
    algorithm: str,
    resolution: float,
    seed: int,
) -> dict[str, int]:
    """Partition ``graph`` at the configured resolution; returns ``entity_id → community_id``."""
    if algorithm == "leiden":
        return _detect_leiden(graph, resolution=resolution, seed=seed)
    return _detect_louvain(graph, resolution=resolution, seed=seed)


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


def _majority_parent(
    fine_assignments: dict[str, int],
    coarse_assignments: dict[str, int],
) -> dict[int, int | None]:
    """For each fine-grained community, pick the coarse community by majority vote.

    A fine community whose members fall across multiple coarse communities is
    assigned to the coarse community that contains the most of its members.
    Communities with no representation in the coarse layer (rare; only if
    isolates were dropped) map to ``None``.
    """
    # Group fine community → list of coarse community IDs from its members.
    coarse_votes: dict[int, list[int]] = {}
    for entity_id, fine_community in fine_assignments.items():
        coarse_community = coarse_assignments.get(entity_id)
        if coarse_community is None:
            continue
        coarse_votes.setdefault(fine_community, []).append(coarse_community)

    parent_of: dict[int, int | None] = {}
    for fine_community, votes in coarse_votes.items():
        if not votes:
            parent_of[fine_community] = None
            continue
        winner, _ = Counter(votes).most_common(1)[0]
        parent_of[fine_community] = winner
    return parent_of
