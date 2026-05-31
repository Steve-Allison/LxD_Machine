"""Tests for hierarchical community detection + storage shape.

Schema migration v7→v8 changed both ``entity_communities`` and
``community_reports`` to composite primary keys keyed by
``(community_id, community_level)`` and added a ``parent_community_id``
column to anchor the hierarchy. These tests pin the resulting invariants.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import networkx as nx
import pytest

from lxd.ontology.communities import (
    _majority_parent,
    detect_hierarchical_communities,
)
from lxd.settings.models import (
    EmbeddingConfig,
    KnowledgeGraphConfig,
    ModelsConfig,
    RuntimeConfig,
)
from lxd.stores.schema import CURRENT_SCHEMA_VERSION, ensure_schema, get_schema_version

pytestmark = [pytest.mark.unit]


def _make_config(**kg_overrides: object) -> RuntimeConfig:
    """Build a minimal RuntimeConfig pinned to the fields the tests touch."""
    kg = KnowledgeGraphConfig(community_resolution=1.0, community_seed=42, **kg_overrides)  # type: ignore[arg-type]
    return RuntimeConfig.model_construct(
        models=ModelsConfig.model_construct(embed="text-embedding-3-small", embed_dims=1536),
        embedding=EmbeddingConfig.model_construct(),
        knowledge_graph=kg,
    )


def _two_clique_graph() -> nx.MultiDiGraph:
    """Two tight cliques joined by a single bridge edge."""
    g = nx.MultiDiGraph()
    # Clique A: a1, a2, a3
    for u, v in [("a1", "a2"), ("a2", "a3"), ("a3", "a1")]:
        g.add_edge(u, v)
        g.add_edge(v, u)
    # Clique B: b1, b2, b3
    for u, v in [("b1", "b2"), ("b2", "b3"), ("b3", "b1")]:
        g.add_edge(u, v)
        g.add_edge(v, u)
    # Bridge
    g.add_edge("a1", "b1")
    return g


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def test_schema_version_advances_to_v8(tmp_path: Path) -> None:
    db_path = tmp_path / "lxd.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        ensure_schema(connection)
        assert get_schema_version(connection) == CURRENT_SCHEMA_VERSION
        assert CURRENT_SCHEMA_VERSION >= 8
    finally:
        connection.close()


def test_community_reports_has_composite_pk_and_parent_column(tmp_path: Path) -> None:
    """v8 schema must accept the same (community_id, level) pair only once but allow per-level distinction."""
    db_path = tmp_path / "lxd.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        ensure_schema(connection)

        # community_id alone is no longer unique — the same id at different levels coexists.
        connection.execute(
            """
            INSERT INTO community_reports (
                community_id, community_level, parent_community_id, member_count,
                member_entity_ids_json, deterministic_summary, source_hash, generated_at
            ) VALUES (0, 0, NULL, 3, '["a"]', 'level 0', 'h0', '2026-05-31T00:00:00Z')
            """
        )
        connection.execute(
            """
            INSERT INTO community_reports (
                community_id, community_level, parent_community_id, member_count,
                member_entity_ids_json, deterministic_summary, source_hash, generated_at
            ) VALUES (0, 1, NULL, 6, '["a","b"]', 'level 1', 'h1', '2026-05-31T00:00:00Z')
            """
        )

        # parent_community_id is queryable.
        rows = connection.execute(
            "SELECT community_id, community_level, parent_community_id FROM community_reports ORDER BY community_level"
        ).fetchall()
        assert [(int(r["community_id"]), int(r["community_level"])) for r in rows] == [
            (0, 0),
            (0, 1),
        ]

        # Same (community_id, community_level) raises IntegrityError on duplicate insert.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO community_reports (
                    community_id, community_level, parent_community_id, member_count,
                    member_entity_ids_json, deterministic_summary, source_hash, generated_at
                ) VALUES (0, 0, NULL, 4, '["x"]', 'dup', 'h', '2026-05-31T00:00:00Z')
                """
            )
    finally:
        connection.close()


def test_entity_communities_allows_same_entity_at_multiple_levels(tmp_path: Path) -> None:
    db_path = tmp_path / "lxd.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        ensure_schema(connection)
        connection.execute(
            "INSERT INTO entity_communities (entity_id, community_id, community_level, assigned_at) "
            "VALUES ('a', 0, 0, '2026-05-31T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO entity_communities (entity_id, community_id, community_level, assigned_at) "
            "VALUES ('a', 7, 1, '2026-05-31T00:00:00Z')"
        )
        rows = connection.execute(
            "SELECT community_id, community_level FROM entity_communities WHERE entity_id = 'a'"
        ).fetchall()
        assert sorted((int(r["community_id"]), int(r["community_level"])) for r in rows) == [
            (0, 0),
            (7, 1),
        ]

        # Same (entity_id, level) raises IntegrityError.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO entity_communities (entity_id, community_id, community_level, assigned_at) "
                "VALUES ('a', 99, 0, '2026-05-31T00:00:00Z')"
            )
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Hierarchical detection
# ---------------------------------------------------------------------------


def test_detect_hierarchical_communities_returns_finest_first() -> None:
    config = _make_config()
    levels = detect_hierarchical_communities(_two_clique_graph(), config)
    # Default = three levels at 1.0, 0.5, 0.25
    assert [lvl.community_level for lvl in levels] == [0, 1, 2]
    assert [lvl.resolution for lvl in levels] == [1.0, 0.5, 0.25]
    # Coarser levels have fewer-or-equal communities than finer levels.
    counts = [lvl.community_count for lvl in levels]
    assert counts == sorted(counts, reverse=True)


def test_detect_hierarchical_communities_rejects_non_decreasing_resolutions() -> None:
    config = _make_config()
    with pytest.raises(ValueError, match="strictly decreasing"):
        detect_hierarchical_communities(
            _two_clique_graph(),
            config,
            resolutions=(0.5, 0.5),
        )


def test_detect_hierarchical_communities_empty_graph_returns_empty_list() -> None:
    config = _make_config()
    empty = nx.MultiDiGraph()
    assert detect_hierarchical_communities(empty, config) == []


def test_detect_hierarchical_communities_top_level_has_no_parent() -> None:
    config = _make_config()
    levels = detect_hierarchical_communities(_two_clique_graph(), config)
    # The coarsest (last) level's parent_of MUST be empty — there's no level above.
    assert levels[-1].parent_of == {}


def test_detect_hierarchical_communities_parent_of_is_majority_vote() -> None:
    config = _make_config()
    levels = detect_hierarchical_communities(_two_clique_graph(), config)
    if len(levels) < 2:
        pytest.skip("test graph collapsed to a single level — guard for tiny graphs")
    # Every fine community has a parent in the coarse level (it can't drop out
    # since every entity exists at every level).
    fine = levels[0]
    coarse_assignments = levels[1].assignments
    for fine_community in set(fine.assignments.values()):
        assert fine_community in fine.parent_of
        assert fine.parent_of[fine_community] in set(coarse_assignments.values())


# ---------------------------------------------------------------------------
# Majority parent helper
# ---------------------------------------------------------------------------


def test_majority_parent_picks_dominant_coarse_community() -> None:
    fine = {"a": 0, "b": 0, "c": 0, "d": 1}
    coarse = {"a": 10, "b": 10, "c": 20, "d": 99}
    parent_of = _majority_parent(fine, coarse)
    # Fine community 0: three members → 10, 10, 20 → majority is 10.
    assert parent_of[0] == 10
    assert parent_of[1] == 99


def test_majority_parent_handles_entity_missing_from_coarse() -> None:
    fine = {"a": 0, "b": 0}
    coarse = {"a": 10}  # b absent
    parent_of = _majority_parent(fine, coarse)
    # Fine community 0 only has one valid coarse vote (a→10).
    assert parent_of[0] == 10


def test_majority_parent_empty_input_returns_empty_map() -> None:
    assert _majority_parent({}, {}) == {}
