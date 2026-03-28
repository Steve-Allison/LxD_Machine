"""CLI commands and state machine for the knowledge graph build pipeline."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
import typer
from rich.console import Console
from rich.table import Table

from lxd.app.bootstrap import bootstrap_app
from lxd.ingest.claims import extract_claims_for_chunks
from lxd.ontology.communities import detect_communities, persist_community_assignments
from lxd.ontology.entity_graph import build_combined_entity_graph
from lxd.ontology.evidence import consolidate_relations
from lxd.ontology.loader import load_ontology
from lxd.ontology.profiles import (
    build_community_reports,
    build_entity_profiles,
    enrich_entity_profiles_with_llm,
)
from lxd.settings.models import RuntimeConfig
from lxd.stores.lancedb import (
    connect_lancedb,
    replace_entity_embeddings,
    reset_entity_table,
)
from lxd.stores.sqlite import (
    begin_graph_build,
    build_store_paths,
    connect_sqlite,
    count_canonical_relations,
    count_claims,
    count_communities,
    count_community_reports,
    count_entity_profiles,
    count_relation_evidence,
    delete_stale_community_reports,
    finish_graph_build,
    initialize_schema,
    load_all_entity_profiles,
    load_chunk_ids_for_entity,
    load_graph_metadata,
    load_graph_version,
    load_latest_graph_build_state,
    update_graph_build_phase,
    upsert_graph_metadata,
)

_log = structlog.get_logger(__name__)
_console = Console()

# Phase execution order (serial default)
_PHASE_ORDER = [
    "evidence",
    "claims",
    "entity_graph",
    "centrality",
    "communities",
    "entity_profiles",
    "community_reports",
    "complete",
]

_LLM_ENRICHMENT_PHASE = "llm_enrichment"

_FULL_OPTION = typer.Option(False, "--full", help="Force regeneration of all phases")
_ENRICH_OPTION = typer.Option(False, "--enrich", help="Include optional LLM enrichment")
_DRY_RUN_OPTION = typer.Option(False, "--dry-run", help="Preview without writing")
_PHASE_OPTION = typer.Option(None, "--phase", help="Run only a specific phase")
_PROFILE_OPTION = typer.Option(None, "--profile", help="Config profile name")
_CONFIG_OPTION = typer.Option(None, "--config", help="Config file path")


def build_graph_command(
    full: bool = _FULL_OPTION,
    enrich: bool = _ENRICH_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
    phase: str | None = _PHASE_OPTION,
    profile: str | None = _PROFILE_OPTION,
    config_path: Path | None = _CONFIG_OPTION,
) -> None:
    """Build or update the knowledge graph from ingested data."""
    ctx = bootstrap_app(profile=profile, config_path=config_path)
    config = ctx.config
    store_paths = build_store_paths(config.paths.data_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    initialize_schema(connection)

    if dry_run:
        _dry_run_report(connection, config)
        return

    # Load ontology for entity definitions and graph
    ontology = load_ontology(
        config.paths.ontology_path,
        include_globs=config.ontology.include_globs,
        ignore_names=config.ontology.ignore_names,
    )

    run_id = str(uuid4())
    started_at = datetime.now(UTC).isoformat()
    graph_version = load_graph_version(connection)
    if full:
        graph_version += 1

    begin_graph_build(
        connection,
        run_id=run_id,
        started_at=started_at,
        graph_version=graph_version,
    )

    notes: list[str] = []
    try:
        phases_to_run = _PHASE_ORDER if phase is None else [phase]

        # Phase: evidence (relations consolidation)
        if "evidence" in phases_to_run:
            update_graph_build_phase(connection, run_id=run_id, current_phase="evidence")
            _console.print("[bold]Phase: relations consolidation + evidence[/bold]")
            rel_count, ev_count = consolidate_relations(connection)
            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="evidence",
                relations_consolidated=rel_count,
                evidence_rows_built=ev_count,
            )
            notes.append(f"relations={rel_count} evidence={ev_count}")

        # Phase: claims
        if "claims" in phases_to_run:
            update_graph_build_phase(connection, run_id=run_id, current_phase="claims")
            _console.print("[bold]Phase: claim extraction[/bold]")
            claims_count = extract_claims_for_chunks(connection, config, force=full)
            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="claims",
                claims_extracted=claims_count,
            )
            notes.append(f"claims={claims_count}")

        # Phase: entity_graph + centrality
        combined_result = None
        if "entity_graph" in phases_to_run or "centrality" in phases_to_run:
            update_graph_build_phase(connection, run_id=run_id, current_phase="entity_graph")
            _console.print("[bold]Phase: combined entity graph + centrality[/bold]")
            combined_result = build_combined_entity_graph(
                ontology.graph,
                connection,
                config,
            )
            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="centrality",
                centrality_computed=len(combined_result.centrality),
            )
            notes.append(
                f"graph_nodes={combined_result.node_count} "
                f"graph_edges={combined_result.edge_count} "
                f"centrality_entities={len(combined_result.centrality)}"
            )

        # Phase: communities
        community_assignments: dict[str, int] = {}
        if "communities" in phases_to_run:
            if combined_result is None:
                _console.print(
                    "[yellow]communities phase requires entity_graph — building[/yellow]"
                )
                combined_result = build_combined_entity_graph(
                    ontology.graph,
                    connection,
                    config,
                )

            update_graph_build_phase(connection, run_id=run_id, current_phase="communities")
            _console.print("[bold]Phase: community detection[/bold]")
            detection = detect_communities(combined_result.graph, config)
            community_assignments = detection.assignments
            persist_community_assignments(connection, community_assignments)
            delete_stale_community_reports(connection)
            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="communities",
                communities_detected=detection.community_count,
            )
            notes.append(f"communities={detection.community_count} algorithm={detection.algorithm}")

        # Phase: entity_profiles (includes entity embeddings)
        if "entity_profiles" in phases_to_run or "profiles" in phases_to_run:
            if combined_result is None:
                combined_result = build_combined_entity_graph(
                    ontology.graph,
                    connection,
                    config,
                )
            if not community_assignments:
                # Load from database if communities phase was skipped
                comm_rows = connection.execute(
                    "SELECT entity_id, community_id FROM entity_communities"
                ).fetchall()
                community_assignments = {
                    str(r["entity_id"]): int(r["community_id"]) for r in comm_rows
                }

            update_graph_build_phase(connection, run_id=run_id, current_phase="entity_profiles")
            _console.print("[bold]Phase: entity profiles[/bold]")
            profiles_built = build_entity_profiles(
                connection,
                ontology.entity_definitions,
                combined_result.centrality,
                community_assignments,
                config,
                force=full,
            )
            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="entity_profiles",
                entity_profiles_built=profiles_built,
            )
            notes.append(f"profiles={profiles_built}")

            # Entity embeddings
            _console.print("[bold]Phase: entity embeddings[/bold]")
            embeddings_count = _compute_entity_embeddings(connection, config, store_paths)
            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="entity_profiles",
                entity_embeddings_computed=embeddings_count,
            )
            notes.append(f"entity_embeddings={embeddings_count}")

        # Phase: community_reports
        if "community_reports" in phases_to_run:
            if combined_result is None:
                combined_result = build_combined_entity_graph(
                    ontology.graph,
                    connection,
                    config,
                )
            if not community_assignments:
                comm_rows = connection.execute(
                    "SELECT entity_id, community_id FROM entity_communities"
                ).fetchall()
                community_assignments = {
                    str(r["entity_id"]): int(r["community_id"]) for r in comm_rows
                }

            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="community_reports",
            )
            _console.print("[bold]Phase: community reports[/bold]")
            reports_built = build_community_reports(
                connection,
                community_assignments,
                combined_result.centrality,
                force=full,
            )
            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="community_reports",
                community_reports_built=reports_built,
            )
            notes.append(f"community_reports={reports_built}")

        # Optional: LLM enrichment
        if enrich and (_LLM_ENRICHMENT_PHASE in phases_to_run or phase is None):
            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="llm_enrichment",
            )
            _console.print("[bold]Phase: LLM enrichment (optional)[/bold]")
            enriched = enrich_entity_profiles_with_llm(connection, config, force=full)
            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="llm_enrichment",
                llm_enrichment_count=enriched,
            )
            notes.append(f"llm_enriched={enriched}")

        # Update graph metadata
        finished_at = datetime.now(UTC).isoformat()
        upsert_graph_metadata(connection, "graph_version", str(graph_version), finished_at)
        upsert_graph_metadata(connection, "last_build_at", finished_at, finished_at)

        finish_graph_build(
            connection,
            run_id=run_id,
            finished_at=finished_at,
            status="complete",
            notes=notes,
        )
        _console.print(f"\n[green]Graph build complete.[/green] Version: {graph_version}")
        for note in notes:
            _console.print(f"  {note}")

    except Exception as exc:
        finished_at = datetime.now(UTC).isoformat()
        notes.append(f"error: {exc}")
        finish_graph_build(
            connection,
            run_id=run_id,
            finished_at=finished_at,
            status="failed",
            notes=notes,
        )
        _console.print(f"[red]Graph build failed: {exc}[/red]")
        raise


def graph_status_command(
    profile: str | None = _PROFILE_OPTION,
    config_path: Path | None = _CONFIG_OPTION,
) -> None:
    """Display knowledge graph build state and statistics."""
    ctx = bootstrap_app(profile=profile, config_path=config_path)
    config = ctx.config
    store_paths = build_store_paths(config.paths.data_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    initialize_schema(connection)

    metadata = load_graph_metadata(connection)
    state = load_latest_graph_build_state(connection)

    table = Table(title="Knowledge Graph Status")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Graph version", metadata.get("graph_version", "0"))
    table.add_row("Last build", metadata.get("last_build_at", "never"))
    table.add_row("Canonical relations", str(count_canonical_relations(connection)))
    table.add_row("Relation evidence rows", str(count_relation_evidence(connection)))
    table.add_row("Claims", str(count_claims(connection)))
    table.add_row("Entity profiles", str(count_entity_profiles(connection)))
    table.add_row("Communities", str(count_communities(connection)))
    table.add_row("Community reports", str(count_community_reports(connection)))

    if state:
        table.add_row("Last build status", state.status)
        table.add_row("Current phase", state.current_phase)
        table.add_row("Started at", state.started_at)
        table.add_row("Finished at", state.finished_at or "in progress")

    _console.print(table)


def _dry_run_report(connection: sqlite3.Connection, config: RuntimeConfig) -> None:
    """Print a dry-run preview of what the graph build would do."""
    _console.print("[bold]Dry run — no writes, no API calls[/bold]\n")

    # Count qualifying chunks for claims
    min_mentions = config.knowledge_graph.claim_extraction_min_mentions
    row = connection.execute(
        """
        SELECT COUNT(DISTINCT c.chunk_id) AS cnt
        FROM chunk_rows c
        JOIN mention_rows m ON c.chunk_id = m.chunk_id
        GROUP BY c.chunk_id
        HAVING COUNT(DISTINCT m.entity_id) >= ?
        """,
        (min_mentions,),
    ).fetchall()
    qualifying_chunks = len(row) if row else 0

    existing_claims = count_claims(connection)
    existing_profiles = count_entity_profiles(connection)

    table = Table(title="Graph Build Preview")
    table.add_column("Phase", style="bold")
    table.add_column("Estimated Work")
    table.add_column("Est. API Calls")

    table.add_row("Evidence consolidation", "Pure SQLite — no API calls", "0")
    table.add_row(
        "Claim extraction",
        f"{qualifying_chunks} qualifying chunks ({existing_claims} existing claims)",
        str(max(0, qualifying_chunks - existing_claims)),
    )
    table.add_row("Entity graph + centrality", "In-memory computation", "0")
    table.add_row("Community detection", "In-memory computation", "0")
    table.add_row("Entity profiles", f"{existing_profiles} existing profiles", "0")
    table.add_row("Community reports", "Deterministic — no API calls", "0")

    _console.print(table)


def _compute_entity_embeddings(
    connection: sqlite3.Connection,
    config: RuntimeConfig,
    store_paths: Any,
) -> int:
    """Compute entity embeddings from chunk embeddings and store in LanceDB.

    Returns:
        Number of entity embeddings computed.
    """
    min_mentions = config.knowledge_graph.entity_embedding_min_mentions
    max_chunks = config.knowledge_graph.entity_summary_max_chunks
    vector_size = config.models.embed_dims

    profiles = load_all_entity_profiles(connection)
    qualifying = [p for p in profiles if p.mention_count >= min_mentions]

    if not qualifying:
        return 0

    db = connect_lancedb(store_paths.lancedb_path)
    entity_table = reset_entity_table(db, vector_size=vector_size)

    records: list[dict[str, object]] = []
    for profile in qualifying:
        chunk_ids = load_chunk_ids_for_entity(connection, profile.entity_id, limit=max_chunks)
        if not chunk_ids:
            continue

        # Load chunk vectors from SQLite
        placeholders = ",".join("?" * len(chunk_ids))
        chunk_rows = connection.execute(
            f"SELECT vector_json FROM chunk_rows WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()

        if not chunk_rows:
            continue

        # Mean-pool chunk embeddings
        vectors: list[list[float]] = []
        for row in chunk_rows:
            vec = json.loads(str(row["vector_json"]))
            if isinstance(vec, list) and len(vec) == vector_size:
                vectors.append(vec)

        if not vectors:
            continue

        mean_vector = [sum(v[i] for v in vectors) / len(vectors) for i in range(vector_size)]

        records.append(
            {
                "entity_id": profile.entity_id,
                "label": profile.label,
                "community_id": profile.community_id if profile.community_id is not None else -1,
                "vector": mean_vector,
            }
        )

    if records:
        replace_entity_embeddings(entity_table, records)

    _log.info("entity embeddings computed", count=len(records))
    return len(records)
