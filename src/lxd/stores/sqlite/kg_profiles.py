"""Entity profiles, entity communities, and community reports."""

from __future__ import annotations

import sqlite3

from lxd.stores._sqlite_rows import (
    community_report_from_row,
    entity_profile_from_row,
    optional_str,
    row_value,
)
from lxd.stores.models import (
    CommunityReportRecord,
    EntityCommunityRecord,
    EntityProfileRecord,
)


def upsert_entity_profile(connection: sqlite3.Connection, record: EntityProfileRecord) -> None:
    """Insert or update an entity profile."""
    with connection:
        connection.execute(
            """
            INSERT INTO entity_profiles (
                entity_id, label, entity_type, domain, aliases_json,
                deterministic_summary, llm_summary,
                chunk_count, doc_count, mention_count, claim_count,
                top_predicates_json, top_claims_json,
                pagerank, betweenness, closeness,
                in_degree, out_degree, eigenvector,
                community_id, source_hash, generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                label = excluded.label,
                entity_type = excluded.entity_type,
                domain = excluded.domain,
                aliases_json = excluded.aliases_json,
                deterministic_summary = excluded.deterministic_summary,
                llm_summary = excluded.llm_summary,
                chunk_count = excluded.chunk_count,
                doc_count = excluded.doc_count,
                mention_count = excluded.mention_count,
                claim_count = excluded.claim_count,
                top_predicates_json = excluded.top_predicates_json,
                top_claims_json = excluded.top_claims_json,
                pagerank = excluded.pagerank,
                betweenness = excluded.betweenness,
                closeness = excluded.closeness,
                in_degree = excluded.in_degree,
                out_degree = excluded.out_degree,
                eigenvector = excluded.eigenvector,
                community_id = excluded.community_id,
                source_hash = excluded.source_hash,
                generated_at = excluded.generated_at
            """,
            (
                record.entity_id,
                record.label,
                record.entity_type,
                record.domain,
                record.aliases_json,
                record.deterministic_summary,
                record.llm_summary,
                record.chunk_count,
                record.doc_count,
                record.mention_count,
                record.claim_count,
                record.top_predicates_json,
                record.top_claims_json,
                record.pagerank,
                record.betweenness,
                record.closeness,
                record.in_degree,
                record.out_degree,
                record.eigenvector,
                record.community_id,
                record.source_hash,
                record.generated_at,
            ),
        )


def load_entity_profile(
    connection: sqlite3.Connection, entity_id: str
) -> EntityProfileRecord | None:
    """Load a single entity profile by ID."""
    row = connection.execute(
        "SELECT * FROM entity_profiles WHERE entity_id = ?", (entity_id,)
    ).fetchone()
    if row is None:
        return None
    return entity_profile_from_row(row)


def load_all_entity_profiles(connection: sqlite3.Connection) -> list[EntityProfileRecord]:
    """Load all entity profiles, ordered by PageRank descending."""
    rows = connection.execute("SELECT * FROM entity_profiles ORDER BY pagerank DESC").fetchall()
    return [entity_profile_from_row(row) for row in rows]


def search_entity_profiles(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
) -> list[EntityProfileRecord]:
    """Search entity profiles by label or alias substring, ranked by PageRank."""
    pattern = f"%{query}%"
    rows = connection.execute(
        """
        SELECT * FROM entity_profiles
        WHERE label LIKE ? OR aliases_json LIKE ?
        ORDER BY pagerank DESC
        LIMIT ?
        """,
        (pattern, pattern, limit),
    ).fetchall()
    return [entity_profile_from_row(row) for row in rows]


def load_top_entities_by_pagerank(
    connection: sqlite3.Connection, *, limit: int = 20
) -> list[EntityProfileRecord]:
    """Load top entities ranked by PageRank."""
    rows = connection.execute(
        "SELECT * FROM entity_profiles ORDER BY pagerank DESC LIMIT ?", (limit,)
    ).fetchall()
    return [entity_profile_from_row(row) for row in rows]


def load_top_entities_by_betweenness(
    connection: sqlite3.Connection, *, limit: int = 20
) -> list[EntityProfileRecord]:
    """Load top entities ranked by betweenness centrality."""
    rows = connection.execute(
        "SELECT * FROM entity_profiles ORDER BY betweenness DESC LIMIT ?", (limit,)
    ).fetchall()
    return [entity_profile_from_row(row) for row in rows]


def load_top_entities_by_closeness(
    connection: sqlite3.Connection, *, limit: int = 20
) -> list[EntityProfileRecord]:
    """Load top entities ranked by closeness centrality."""
    rows = connection.execute(
        "SELECT * FROM entity_profiles ORDER BY closeness DESC LIMIT ?", (limit,)
    ).fetchall()
    return [entity_profile_from_row(row) for row in rows]


def load_entity_profile_source_hashes(
    connection: sqlite3.Connection,
) -> dict[str, str]:
    """Load entity_id → source_hash mapping for incremental rebuild."""
    rows = connection.execute("SELECT entity_id, source_hash FROM entity_profiles").fetchall()
    return {str(row["entity_id"]): str(row["source_hash"]) for row in rows}


def replace_entity_communities(
    connection: sqlite3.Connection, records: list[EntityCommunityRecord]
) -> None:
    """Replace all community assignments (truncate and rebuild)."""
    with connection:
        connection.execute("DELETE FROM entity_communities")
        if records:
            connection.executemany(
                """
                INSERT INTO entity_communities (
                    entity_id, community_id, community_level, modularity_class, assigned_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.entity_id,
                        r.community_id,
                        r.community_level,
                        r.modularity_class,
                        r.assigned_at,
                    )
                    for r in records
                ],
            )


def load_entity_community(
    connection: sqlite3.Connection, entity_id: str
) -> EntityCommunityRecord | None:
    """Load community assignment for one entity."""
    row = connection.execute(
        "SELECT * FROM entity_communities WHERE entity_id = ?", (entity_id,)
    ).fetchone()
    if row is None:
        return None
    return EntityCommunityRecord(
        entity_id=str(row["entity_id"]),
        community_id=int(row["community_id"]),
        community_level=int(row["community_level"]),
        modularity_class=optional_str(row["modularity_class"]),
        assigned_at=str(row["assigned_at"]),
    )


def load_community_members(connection: sqlite3.Connection, community_id: int) -> list[str]:
    """Return entity IDs belonging to a community."""
    rows = connection.execute(
        "SELECT entity_id FROM entity_communities WHERE community_id = ?",
        (community_id,),
    ).fetchall()
    return [str(row["entity_id"]) for row in rows]


def upsert_community_report(connection: sqlite3.Connection, record: CommunityReportRecord) -> None:
    """Insert or update a community report."""
    with connection:
        connection.execute(
            """
            INSERT INTO community_reports (
                community_id, community_level, member_count, member_entity_ids_json,
                deterministic_summary, llm_summary,
                top_entities_json, top_claims_json,
                intra_community_edge_count, source_hash, generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(community_id) DO UPDATE SET
                community_level = excluded.community_level,
                member_count = excluded.member_count,
                member_entity_ids_json = excluded.member_entity_ids_json,
                deterministic_summary = excluded.deterministic_summary,
                llm_summary = excluded.llm_summary,
                top_entities_json = excluded.top_entities_json,
                top_claims_json = excluded.top_claims_json,
                intra_community_edge_count = excluded.intra_community_edge_count,
                source_hash = excluded.source_hash,
                generated_at = excluded.generated_at
            """,
            (
                record.community_id,
                record.community_level,
                record.member_count,
                record.member_entity_ids_json,
                record.deterministic_summary,
                record.llm_summary,
                record.top_entities_json,
                record.top_claims_json,
                record.intra_community_edge_count,
                record.source_hash,
                record.generated_at,
            ),
        )


def load_community_report(
    connection: sqlite3.Connection, community_id: int
) -> CommunityReportRecord | None:
    """Load a single community report."""
    row = connection.execute(
        "SELECT * FROM community_reports WHERE community_id = ?", (community_id,)
    ).fetchone()
    if row is None:
        return None
    return community_report_from_row(row)


def load_all_community_reports(
    connection: sqlite3.Connection,
) -> list[CommunityReportRecord]:
    """Load all community reports."""
    rows = connection.execute(
        "SELECT * FROM community_reports ORDER BY member_count DESC"
    ).fetchall()
    return [community_report_from_row(row) for row in rows]


def delete_stale_community_reports(connection: sqlite3.Connection) -> int:
    """Remove community reports whose community_id no longer exists in entity_communities."""
    with connection:
        cursor = connection.execute(
            """
            DELETE FROM community_reports
            WHERE community_id NOT IN (
                SELECT DISTINCT community_id FROM entity_communities
            )
            """
        )
    return cursor.rowcount


def count_entity_profiles(connection: sqlite3.Connection) -> int:
    """Return total entity profile count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM entity_profiles").fetchone()
    return int(row_value(row, "cnt"))


def count_communities(connection: sqlite3.Connection) -> int:
    """Return number of distinct communities."""
    row = connection.execute(
        "SELECT COUNT(DISTINCT community_id) AS cnt FROM entity_communities"
    ).fetchone()
    return int(row_value(row, "cnt"))


def count_community_reports(connection: sqlite3.Connection) -> int:
    """Return total community report count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM community_reports").fetchone()
    return int(row_value(row, "cnt"))
