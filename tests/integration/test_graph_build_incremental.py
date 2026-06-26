"""Regression tests for the two incremental build-graph fixes.

Issue 2 — Profile / community-report ``source_hash`` is computed from
*rank positions*, not raw centrality floats. Adding one new file to the
corpus shifts everyone's PageRank by sub-rank amounts; if the hash were
sensitive to that noise, every profile would rebuild and every LLM
enrichment would re-fire on every routine update.

Issue 1 — ``_compute_entity_embeddings`` no longer wipes the LanceDB
``entity_embeddings`` table on every run. A per-entity ``source_hash``
(sorted ``chunk_ids`` + embedding model identity) gates the recompute;
unchanged entities keep their existing mean-pooled vector and entities
that fell below ``entity_embedding_min_mentions`` are evicted from both
LanceDB and the state table.

These tests fail on the pre-fix code (every profile rebuilt on float
noise; every entity re-embedded on every run; stale rows never evicted).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

# Private symbols are exercised here on purpose: these tests live in the same
# logical unit (``cli/graph.py`` and ``ontology/profiles.py``) and regress the
# fixes documented in the module docstring.
from lxd.cli import graph as _graph_module
from lxd.domain.ids import blake3_hex
from lxd.ontology import profiles as _profiles_module
from lxd.ontology.entity_graph import CentralityScores
from lxd.ontology.profiles import build_community_reports, build_entity_profiles
from lxd.settings.models import RuntimeConfig
from lxd.stores.lancedb import connect_lancedb, open_chunk_table, open_entity_table
from lxd.stores.models import StorePaths
from lxd.stores.schema import ensure_schema
from lxd.stores.sqlite.connection import connect_sqlite
from lxd.stores.sqlite.kg_profiles import (
    load_community_report_source_hashes,
    load_entity_embedding_state,
    load_entity_profile_source_hashes,
)

_compute_entity_embeddings = _graph_module._compute_entity_embeddings  # pyright: ignore[reportPrivateUsage]
_compute_ranks = _profiles_module._compute_ranks  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Fixtures and seeding helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def connection(tmp_path: Path) -> Generator[sqlite3.Connection]:
    """Fresh SQLite at the current schema version."""
    conn = connect_sqlite(tmp_path / "kg.sqlite3")
    ensure_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def _config(*, min_mentions: int = 1, max_chunks: int = 100, embed_dims: int = 4) -> RuntimeConfig:
    """Build a SimpleNamespace stub typed as RuntimeConfig.

    ``_compute_entity_embeddings`` reads only the four leaf attributes set
    here; a full Pydantic model would add ~80 lines of irrelevant config to
    every test. The cast is honest — the call surface is genuinely a subset.
    """
    return cast(
        "RuntimeConfig",
        SimpleNamespace(
            models=SimpleNamespace(embed="test-embed", embed_dims=embed_dims),
            knowledge_graph=SimpleNamespace(
                entity_embedding_min_mentions=min_mentions,
                entity_summary_max_chunks=max_chunks,
            ),
        ),
    )


def _seed_manifest_and_chunks(
    connection: sqlite3.Connection,
    chunk_specs: list[tuple[str, str, str]],
) -> None:
    """Seed manifest + chunk_rows so mention_rows FKs hold.

    Each spec: ``(chunk_id, source_rel_path, text)``.
    """
    sources = {spec[1] for spec in chunk_specs}
    for source_rel_path in sources:
        connection.execute(
            """
            INSERT INTO corpus_manifest (
                source_rel_path, absolute_path, source_type, source_domain,
                document_id, blake3_hash, file_size_bytes, lifecycle_status,
                retrieval_status, chunk_count, last_seen_at
            )
            VALUES (?, ?, 'markdown', 'wiki', 'doc1', 'hash', 0,
                    'complete', 'searchable', 0, '2026-06-26T00:00:00Z')
            """,
            (source_rel_path, f"/abs/{source_rel_path}"),
        )
    for chunk_id, source_rel_path, text in chunk_specs:
        connection.execute(
            """
            INSERT INTO chunk_rows (
                chunk_id, document_id, source_rel_path, source_filename,
                source_type, source_domain, source_hash, citation_label,
                chunk_index, chunk_occurrence, token_count, text, chunk_hash,
                score_hint, metadata_json, embedding_model, embedding_dims
            )
            VALUES (?, 'doc1', ?, ?, 'markdown', 'wiki', 'sh', ?, 0, 0,
                    10, ?, ?, 'none', '{}', 'test-embed', 4)
            """,
            (chunk_id, source_rel_path, source_rel_path, chunk_id, text, chunk_id),
        )
    connection.commit()


def _seed_mentions(
    connection: sqlite3.Connection,
    mentions: list[tuple[str, str, str]],
) -> None:
    """Insert mention_rows: list of ``(mention_id, entity_id, chunk_id)``."""
    rel_for_chunk: dict[str, str] = {}
    for chunk_id, source_rel_path in connection.execute(
        "SELECT chunk_id, source_rel_path FROM chunk_rows"
    ).fetchall():
        rel_for_chunk[str(chunk_id)] = str(source_rel_path)
    for mention_id, entity_id, chunk_id in mentions:
        connection.execute(
            """
            INSERT INTO mention_rows (
                mention_id, entity_id, term_source, source_domain,
                source_rel_path, source_filename, chunk_id, surface_form,
                start_char, end_char
            )
            VALUES (?, ?, 'matcher', 'wiki', ?, ?, ?, ?, 0, 5)
            """,
            (
                mention_id,
                entity_id,
                rel_for_chunk[chunk_id],
                rel_for_chunk[chunk_id],
                chunk_id,
                entity_id,
            ),
        )
    connection.commit()


def _three_entities() -> list[dict[str, Any]]:
    return [
        {"canonical_id": "alpha", "entity_type": "concept", "domain": "id"},
        {"canonical_id": "bravo", "entity_type": "concept", "domain": "id"},
        {"canonical_id": "charlie", "entity_type": "concept", "domain": "id"},
    ]


def _centrality(values: dict[str, tuple[float, float, float]]) -> dict[str, CentralityScores]:
    """Build a centrality dict from ``{entity_id: (pagerank, betweenness, closeness)}``."""
    return {
        eid: CentralityScores(
            entity_id=eid,
            pagerank=pr,
            betweenness=bt,
            closeness=cl,
            in_degree=0,
            out_degree=0,
            eigenvector=0.0,
        )
        for eid, (pr, bt, cl) in values.items()
    }


# ---------------------------------------------------------------------------
# Issue 2 — _compute_ranks behaviour
# ---------------------------------------------------------------------------


def test_compute_ranks_returns_one_based_descending() -> None:
    """Highest score is rank 1; ties resolve in stable sort order."""
    centrality = _centrality(
        {"alpha": (0.5, 0.0, 0.0), "bravo": (0.9, 0.0, 0.0), "charlie": (0.1, 0.0, 0.0)}
    )
    ranks = _compute_ranks(centrality, "pagerank")
    assert ranks == {"bravo": 1, "alpha": 2, "charlie": 3}


def test_compute_ranks_unknown_metric_raises() -> None:
    """Asking for a metric that isn't a CentralityScores field is a programmer error."""
    centrality = _centrality({"alpha": (0.5, 0.0, 0.0)})
    with pytest.raises(AttributeError):
        _compute_ranks(centrality, "not_a_metric")


# ---------------------------------------------------------------------------
# Issue 2 — Profile hash stability
# ---------------------------------------------------------------------------


def test_profile_hash_stable_under_subrank_centrality_noise(
    connection: sqlite3.Connection,
) -> None:
    """Sub-rank float shifts must NOT trigger a profile rebuild.

    Pre-fix: ``str(pagerank)`` in the hash made 4th-decimal centrality drift
    cascade into a full profile rebuild and an LLM re-enrichment per entity.
    """
    _seed_manifest_and_chunks(
        connection,
        [("chunk_a", "page_a.md", "text"), ("chunk_b", "page_b.md", "text")],
    )
    _seed_mentions(
        connection,
        [
            ("m1", "alpha", "chunk_a"),
            ("m2", "bravo", "chunk_a"),
            ("m3", "charlie", "chunk_b"),
        ],
    )

    centrality_before = _centrality(
        {
            "alpha": (0.50, 0.20, 0.30),
            "bravo": (0.30, 0.10, 0.20),
            "charlie": (0.10, 0.05, 0.10),
        }
    )
    build_entity_profiles(
        connection,
        _three_entities(),
        centrality_before,
        community_assignments={},
        config=_config(),
    )
    hashes_before = load_entity_profile_source_hashes(connection)
    assert len(hashes_before) == 3

    # Nudge every score by < 1% while preserving the descending order.
    centrality_after = _centrality(
        {
            "alpha": (0.5004, 0.2003, 0.3002),
            "bravo": (0.3001, 0.1001, 0.2001),
            "charlie": (0.1001, 0.0501, 0.1001),
        }
    )
    rebuilt = build_entity_profiles(
        connection,
        _three_entities(),
        centrality_after,
        community_assignments={},
        config=_config(),
    )
    hashes_after = load_entity_profile_source_hashes(connection)

    assert rebuilt == 0, (
        "Sub-rank float shifts must not rebuild any profile; pre-fix code "
        f"would have rebuilt all 3. Got {rebuilt}."
    )
    assert hashes_before == hashes_after


def test_profile_hash_changes_when_pagerank_rank_swaps(
    connection: sqlite3.Connection,
) -> None:
    """A true rank swap (alpha ↔ bravo) must dirty BOTH profiles, not just one."""
    _seed_manifest_and_chunks(connection, [("chunk_a", "page_a.md", "text")])
    _seed_mentions(
        connection,
        [
            ("m1", "alpha", "chunk_a"),
            ("m2", "bravo", "chunk_a"),
            ("m3", "charlie", "chunk_a"),
        ],
    )

    build_entity_profiles(
        connection,
        _three_entities(),
        _centrality(
            {
                "alpha": (0.50, 0.20, 0.30),
                "bravo": (0.30, 0.10, 0.20),
                "charlie": (0.10, 0.05, 0.10),
            }
        ),
        community_assignments={},
        config=_config(),
    )
    hashes_before = load_entity_profile_source_hashes(connection)

    swapped = build_entity_profiles(
        connection,
        _three_entities(),
        _centrality(
            {
                "alpha": (0.30, 0.20, 0.30),  # alpha dropped behind bravo
                "bravo": (0.50, 0.10, 0.20),
                "charlie": (0.10, 0.05, 0.10),
            }
        ),
        community_assignments={},
        config=_config(),
    )
    hashes_after = load_entity_profile_source_hashes(connection)

    assert swapped == 2, (
        f"Rank swap of alpha↔bravo should dirty both, charlie unchanged. Got {swapped}."
    )
    assert hashes_after["alpha"] != hashes_before["alpha"]
    assert hashes_after["bravo"] != hashes_before["bravo"]
    assert hashes_after["charlie"] == hashes_before["charlie"]


def test_profile_hash_changes_when_chunk_set_changes(
    connection: sqlite3.Connection,
) -> None:
    """Adding a chunk to an entity's mention set MUST trigger rebuild."""
    _seed_manifest_and_chunks(
        connection,
        [("chunk_a", "page_a.md", "text"), ("chunk_b", "page_b.md", "text")],
    )
    _seed_mentions(connection, [("m1", "alpha", "chunk_a")])
    centrality = _centrality({"alpha": (0.5, 0.2, 0.3)})

    build_entity_profiles(
        connection,
        [{"canonical_id": "alpha", "entity_type": "concept", "domain": "id"}],
        centrality,
        community_assignments={},
        config=_config(),
    )
    before = load_entity_profile_source_hashes(connection)["alpha"]

    _seed_mentions(connection, [("m2", "alpha", "chunk_b")])
    rebuilt = build_entity_profiles(
        connection,
        [{"canonical_id": "alpha", "entity_type": "concept", "domain": "id"}],
        centrality,
        community_assignments={},
        config=_config(),
    )
    after = load_entity_profile_source_hashes(connection)["alpha"]

    assert rebuilt == 1
    assert before != after


# ---------------------------------------------------------------------------
# Issue 2 — Community report hash cascade
# ---------------------------------------------------------------------------


def test_community_report_skips_when_member_profiles_unchanged(
    connection: sqlite3.Connection,
) -> None:
    """Without the cascade-skip the report was upserted every run (and llm_summary
    clobbered to NULL), forcing LLM re-enrichment on every routine build.
    """
    _seed_manifest_and_chunks(connection, [("chunk_a", "page_a.md", "text")])
    _seed_mentions(
        connection,
        [
            ("m1", "alpha", "chunk_a"),
            ("m2", "bravo", "chunk_a"),
        ],
    )

    centrality = _centrality({"alpha": (0.5, 0.2, 0.3), "bravo": (0.3, 0.1, 0.2)})
    assignments = {"alpha": 0, "bravo": 0}
    entity_defs = [
        {"canonical_id": "alpha", "entity_type": "concept", "domain": "id"},
        {"canonical_id": "bravo", "entity_type": "concept", "domain": "id"},
    ]
    build_entity_profiles(connection, entity_defs, centrality, assignments, _config())
    build_community_reports(connection, assignments, centrality)
    hashes_before = load_community_report_source_hashes(connection)
    assert (0, 0) in hashes_before

    # Simulate a routine no-op build: profiles and centrality unchanged.
    profiles_rebuilt = build_entity_profiles(
        connection, entity_defs, centrality, assignments, _config()
    )
    reports_rebuilt = build_community_reports(connection, assignments, centrality)
    hashes_after = load_community_report_source_hashes(connection)

    assert profiles_rebuilt == 0
    assert reports_rebuilt == 0, (
        f"Unchanged community report must be skipped; pre-fix upserted it. Got {reports_rebuilt}."
    )
    assert hashes_after == hashes_before


def test_community_report_force_bypasses_skip(connection: sqlite3.Connection) -> None:
    """``force=True`` must rebuild reports even when source_hash matches.

    Otherwise ``build-graph --full`` couldn't refresh enrichment.
    """
    _seed_manifest_and_chunks(connection, [("chunk_a", "page_a.md", "text")])
    _seed_mentions(connection, [("m1", "alpha", "chunk_a")])

    centrality = _centrality({"alpha": (0.5, 0.2, 0.3)})
    assignments = {"alpha": 0}
    entity_defs = [{"canonical_id": "alpha", "entity_type": "concept", "domain": "id"}]
    build_entity_profiles(connection, entity_defs, centrality, assignments, _config())
    build_community_reports(connection, assignments, centrality)

    forced = build_community_reports(connection, assignments, centrality, force=True)
    assert forced == 1


# ---------------------------------------------------------------------------
# Issue 1 — Entity embedding incremental
# ---------------------------------------------------------------------------


def _entity_embedding_setup(
    tmp_path: Path,
    connection: sqlite3.Connection,
    chunks: list[tuple[str, str, str]],
    mentions: list[tuple[str, str, str]],
    *,
    embed_dims: int = 4,
) -> StorePaths:
    """Seed corpus, chunks, mentions, profiles, and LanceDB chunk_vectors.

    Returns the ``StorePaths`` ``_compute_entity_embeddings`` accepts.
    """
    _seed_manifest_and_chunks(connection, chunks)
    _seed_mentions(connection, mentions)

    # Build profiles so qualifying-entity discovery works.
    entity_ids = sorted({mention[1] for mention in mentions})
    entity_defs = [
        {"canonical_id": eid, "entity_type": "concept", "domain": "id"} for eid in entity_ids
    ]
    centrality = _centrality({eid: (0.5, 0.2, 0.3) for eid in entity_ids})
    build_entity_profiles(connection, entity_defs, centrality, {}, _config(embed_dims=embed_dims))

    store_paths = StorePaths(
        sqlite_path=tmp_path / "kg.sqlite3",
        lancedb_path=tmp_path / "lance",
    )

    # Seed LanceDB chunk_vectors for the chunks we just created.
    db = connect_lancedb(store_paths.lancedb_path)
    chunk_table = open_chunk_table(db, vector_size=embed_dims)
    chunk_ids = sorted({chunk_id for chunk_id, _, _ in chunks})
    chunk_table.add(
        [
            {
                "chunk_id": chunk_id,
                "document_id": "doc1",
                "vector": [1.0] + [0.0] * (embed_dims - 1),
                "source_rel_path": "page_a.md",
                "source_filename": "page_a.md",
                "source_type": "markdown",
                "source_domain": "wiki",
                "source_hash": "sh",
                "citation_label": chunk_id,
                "chunk_index": 0,
                "chunk_occurrence": 0,
                "token_count": 10,
                "text": "text",
                "score_hint": "none",
                "metadata_json": "{}",
                "cited_sources_json": "[]",
                "wiki_links_json": "[]",
            }
            for chunk_id in chunk_ids
        ]
    )
    return store_paths


def test_entity_embeddings_skip_unchanged_chunk_set(
    tmp_path: Path, connection: sqlite3.Connection
) -> None:
    """Running the compute step twice with identical state must do zero work the second time.

    Pre-fix: ``reset_entity_table`` wiped LanceDB and every entity was re-pooled.
    """
    store_paths = _entity_embedding_setup(
        tmp_path,
        connection,
        chunks=[("chunk_a", "page_a.md", "t"), ("chunk_b", "page_a.md", "t")],
        mentions=[
            ("m1", "alpha", "chunk_a"),
            ("m2", "alpha", "chunk_b"),
            ("m3", "bravo", "chunk_a"),
        ],
    )

    first = _compute_entity_embeddings(connection, _config(), store_paths)
    assert first == 2, f"First run should embed both entities, got {first}."
    state_after_first = load_entity_embedding_state(connection)
    assert set(state_after_first) == {"alpha", "bravo"}

    second = _compute_entity_embeddings(connection, _config(), store_paths)
    assert second == 0, f"Second run with identical state must recompute nothing; got {second}."
    assert load_entity_embedding_state(connection) == state_after_first


def test_entity_embeddings_recompute_when_chunk_set_changes(
    tmp_path: Path, connection: sqlite3.Connection
) -> None:
    """Adding a chunk to one entity must recompute exactly that entity."""
    store_paths = _entity_embedding_setup(
        tmp_path,
        connection,
        chunks=[("chunk_a", "page_a.md", "t"), ("chunk_b", "page_a.md", "t")],
        mentions=[
            ("m1", "alpha", "chunk_a"),
            ("m2", "bravo", "chunk_a"),
        ],
    )
    _compute_entity_embeddings(connection, _config(), store_paths)
    state_before = load_entity_embedding_state(connection)

    # alpha now also mentions chunk_b — bravo unchanged.
    _seed_mentions(connection, [("m3", "alpha", "chunk_b")])
    # Profile must be rebuilt so mention_count and chunk_ids are current.
    build_entity_profiles(
        connection,
        [
            {"canonical_id": "alpha", "entity_type": "concept", "domain": "id"},
            {"canonical_id": "bravo", "entity_type": "concept", "domain": "id"},
        ],
        _centrality({"alpha": (0.5, 0.2, 0.3), "bravo": (0.3, 0.1, 0.2)}),
        {},
        _config(),
    )

    recomputed = _compute_entity_embeddings(connection, _config(), store_paths)
    state_after = load_entity_embedding_state(connection)

    assert recomputed == 1, f"Only alpha changed; expected 1 recompute, got {recomputed}."
    assert state_after["alpha"] != state_before["alpha"]
    assert state_after["bravo"] == state_before["bravo"]


def test_entity_embeddings_evict_no_longer_qualifying(
    tmp_path: Path, connection: sqlite3.Connection
) -> None:
    """When an entity falls below ``min_mentions`` it must be evicted from BOTH stores."""
    store_paths = _entity_embedding_setup(
        tmp_path,
        connection,
        chunks=[("chunk_a", "page_a.md", "t")],
        mentions=[
            ("m1", "alpha", "chunk_a"),
            ("m2", "bravo", "chunk_a"),
        ],
    )
    _compute_entity_embeddings(connection, _config(min_mentions=1), store_paths)
    assert set(load_entity_embedding_state(connection)) == {"alpha", "bravo"}

    # Raise the floor so neither qualifies.
    _compute_entity_embeddings(connection, _config(min_mentions=5), store_paths)
    assert load_entity_embedding_state(connection) == {}, (
        "State rows for entities that no longer qualify must be removed."
    )

    db = connect_lancedb(store_paths.lancedb_path)
    entity_table = open_entity_table(db, vector_size=4)
    rows = entity_table.search().limit(100).to_list()
    assert rows == [], "LanceDB entity_embeddings rows must be evicted alongside SQLite state."


def test_entity_embedding_source_hash_invalidates_on_model_change(
    tmp_path: Path, connection: sqlite3.Connection
) -> None:
    """Same chunks but a different embedding_model must trigger recompute.

    Otherwise switching embedding models would leave vectors mismatched with
    the chunk vectors they're meant to mean-pool.
    """
    store_paths = _entity_embedding_setup(
        tmp_path,
        connection,
        chunks=[("chunk_a", "page_a.md", "t")],
        mentions=[("m1", "alpha", "chunk_a")],
    )
    _compute_entity_embeddings(connection, _config(), store_paths)
    before = load_entity_embedding_state(connection)["alpha"]
    # Sanity-check the hash composition is what we expect.
    assert before == blake3_hex("test-embed", "4", "chunk_a")

    different_model = cast(
        "RuntimeConfig",
        SimpleNamespace(
            models=SimpleNamespace(embed="other-embed", embed_dims=4),
            knowledge_graph=SimpleNamespace(
                entity_embedding_min_mentions=1, entity_summary_max_chunks=100
            ),
        ),
    )
    recomputed = _compute_entity_embeddings(connection, different_model, store_paths)
    after = load_entity_embedding_state(connection)["alpha"]

    assert recomputed == 1
    assert after == blake3_hex("other-embed", "4", "chunk_a")
    assert after != before
