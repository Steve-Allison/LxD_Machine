"""CLI commands and state machine for the knowledge graph build pipeline."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import structlog
import typer
from rich.console import Console
from rich.table import Table

from lxd.app.bootstrap import bootstrap_app
from lxd.domain.ids import blake3_hex
from lxd.ingest.claims import (
    collect_claims_batch,
    extract_claims_for_chunks,
    prepare_claims_batch_jsonl,
    submit_claims_batch,
)
from lxd.ingest.llm_client import poll_batch as _poll_batch
from lxd.ontology.communities import (
    CommunityDetectionResult,
    detect_hierarchical_communities,
    persist_hierarchical_communities,
)
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
    load_vectors_by_chunk_ids,
    open_chunk_table,
    open_entity_table,
    upsert_entity_embeddings,
)
from lxd.stores.sqlite.chunks import load_chunk_ids_for_entity
from lxd.stores.sqlite.claims import count_claims
from lxd.stores.sqlite.connection import build_store_paths, connect_sqlite, initialize_schema
from lxd.stores.sqlite.kg_profiles import (
    count_communities,
    count_community_reports,
    count_entity_profiles,
    delete_entity_embedding_state,
    delete_stale_community_reports,
    load_all_entity_profiles,
    load_entity_embedding_state,
    upsert_entity_embedding_state,
)
from lxd.stores.sqlite.kg_relations import (
    begin_graph_build,
    count_canonical_relations,
    count_relation_evidence,
    finish_graph_build,
    load_graph_metadata,
    load_graph_version,
    load_latest_graph_build_state,
    update_graph_build_phase,
    upsert_graph_metadata,
)

_log = structlog.get_logger(__name__)
_console = Console()

# Phase execution order (serial default)
_PHASE_ORDER: Final = [
    "evidence",
    "claims",
    "entity_graph",
    "centrality",
    "communities",
    "entity_profiles",
    "community_reports",
    "complete",
]

_LLM_ENRICHMENT_PHASE: Final = "llm_enrichment"

_FULL_OPTION: Final = typer.Option(False, "--full", help="Force regeneration of all phases")
_ENRICH_OPTION: Final = typer.Option(False, "--enrich", help="Include optional LLM enrichment")
_DRY_RUN_OPTION: Final = typer.Option(False, "--dry-run", help="Preview without writing")
_BATCH_OPTION: Final = typer.Option(
    False, "--batch", help="Submit claims to OpenAI Batch API instead of async"
)
_PHASE_OPTION: Final = typer.Option(None, "--phase", help="Run only a specific phase")
_PROFILE_OPTION: Final = typer.Option(None, "--profile", help="Config profile name")
_CONFIG_OPTION: Final = typer.Option(None, "--config", help="Config file path")


def build_graph_command(
    full: bool = _FULL_OPTION,
    enrich: bool = _ENRICH_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
    batch: bool = _BATCH_OPTION,
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

    if full:
        claim_count = connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        profile_count = connection.execute("SELECT COUNT(*) FROM entity_profiles").fetchone()[0]
        if claim_count > 0 or profile_count > 0:
            _console.print(
                f"[bold red]--full will re-extract {claim_count:,} claims and rebuild"
                f" {profile_count:,} entity profiles.[/bold red]"
            )
            _console.print(
                "This costs API calls and time. Incremental build is usually sufficient."
            )
            if not typer.confirm("Proceed with full rebuild?"):
                raise typer.Abort()

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
            if batch:
                _console.print("[bold]Phase: claim extraction (Batch API)[/bold]")
                batch_dir = config.paths.data_path / "batch"
                jsonl_path = prepare_claims_batch_jsonl(connection, config, batch_dir, force=full)
                batch_id = submit_claims_batch(jsonl_path, config)
                _console.print(f"[green]Batch submitted:[/green] {batch_id}")
                _console.print(f"JSONL: {jsonl_path}")
                _console.print("Run [bold]collect-batch[/bold] when complete.")
                notes.append(f"claims_batch={batch_id}")
                # Skip remaining phases — batch results need collecting first
                finish_graph_build(
                    connection,
                    run_id=run_id,
                    finished_at=datetime.now(UTC).isoformat(),
                    status="batch_submitted",
                    notes=notes,
                )
                return
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
        community_levels: list[CommunityDetectionResult] | None = None
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
            _console.print("[bold]Phase: hierarchical community detection[/bold]")
            community_levels = detect_hierarchical_communities(combined_result.graph, config)
            persist_hierarchical_communities(connection, community_levels)
            level_zero = community_levels[0] if community_levels else None
            community_assignments = level_zero.assignments if level_zero is not None else {}
            delete_stale_community_reports(connection)
            level_counts = ",".join(str(lvl.community_count) for lvl in community_levels)
            top_level_count = community_levels[-1].community_count if community_levels else 0
            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="communities",
                communities_detected=top_level_count,
            )
            algorithm_name = level_zero.algorithm if level_zero is not None else "n/a"
            notes.append(
                f"communities=[{level_counts}] (finest→coarsest) algorithm={algorithm_name}"
            )

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
            _console.print("[bold]Phase: community reports (per hierarchy level)[/bold]")
            levels_to_report = _community_levels_for_reports(
                community_levels=community_levels,
                connection=connection,
                level_zero_assignments=community_assignments,
            )
            reports_built = 0
            for level_assignments, level_idx, parent_of in levels_to_report:
                reports_built += build_community_reports(
                    connection,
                    level_assignments,
                    combined_result.centrality,
                    force=full,
                    community_level=level_idx,
                    parent_of=parent_of,
                )
            update_graph_build_phase(
                connection,
                run_id=run_id,
                current_phase="community_reports",
                community_reports_built=reports_built,
            )
            notes.append(f"community_reports={reports_built} levels={len(levels_to_report)}")

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


def _community_levels_for_reports(
    *,
    community_levels: list[CommunityDetectionResult] | None,
    connection: sqlite3.Connection,
    level_zero_assignments: dict[str, int],
) -> list[tuple[dict[str, int], int, dict[int, int | None]]]:
    """Return the per-level ``(assignments, level, parent_of)`` tuples for report build.

    Three sources, in priority order:

    1. ``community_levels`` from the in-process hierarchical detection — used
       when the ``communities`` phase ran in the same invocation.
    2. ``entity_communities`` table grouped by ``community_level`` — used when
       ``community_reports`` runs as a standalone phase against a previously
       persisted hierarchy.
    3. Fallback to ``level_zero_assignments`` as a single-level partition —
       only fires when neither (1) nor (2) yielded any levels (legacy DB
       state).
    """
    if community_levels:
        return [
            (level.assignments, level.community_level, level.parent_of)
            for level in community_levels
        ]

    rows = connection.execute(
        "SELECT entity_id, community_id, community_level FROM entity_communities ORDER BY community_level"
    ).fetchall()
    if not rows:
        return [(level_zero_assignments, 0, {})]

    by_level: dict[int, dict[str, int]] = {}
    for row in rows:
        level = int(row["community_level"])
        by_level.setdefault(level, {})[str(row["entity_id"])] = int(row["community_id"])

    ordered = sorted(by_level.items())
    result: list[tuple[dict[str, int], int, dict[int, int | None]]] = []
    for index, (level, assignments) in enumerate(ordered):
        if index + 1 < len(ordered):
            coarser_assignments = ordered[index + 1][1]
            parent_of: dict[int, int | None] = _reconstruct_parent_of(
                assignments, coarser_assignments
            )
        else:
            parent_of = {}
        result.append((assignments, level, parent_of))
    return result


def _reconstruct_parent_of(
    fine_assignments: dict[str, int],
    coarse_assignments: dict[str, int],
) -> dict[int, int | None]:
    """Majority-vote parent reconstruction from two assignment maps."""
    from collections import Counter

    votes: dict[int, list[int]] = {}
    for entity_id, fine_id in fine_assignments.items():
        coarse_id = coarse_assignments.get(entity_id)
        if coarse_id is None:
            continue
        votes.setdefault(fine_id, []).append(coarse_id)
    parent_of: dict[int, int | None] = {}
    for fine_id, ballots in votes.items():
        if not ballots:
            parent_of[fine_id] = None
            continue
        parent_of[fine_id] = Counter(ballots).most_common(1)[0][0]
    return parent_of


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
    """Compute entity embeddings incrementally.

    The mean-pooled L2-normalised vector for one entity is a pure function of
    its (sorted chunk_ids, embedding_model, embedding_dims). When that tuple
    is unchanged since the previous build, the existing LanceDB row is left
    in place — no chunk-vector fetch, no mean-pool, no LanceDB write. Entities
    whose mention count fell below ``entity_embedding_min_mentions`` are
    evicted from both LanceDB and the state table.

    Returns:
        Number of entity embeddings (re)computed this run. Excludes skip-hits.
    """
    min_mentions = config.knowledge_graph.entity_embedding_min_mentions
    max_chunks = config.knowledge_graph.entity_summary_max_chunks
    vector_size = config.models.embed_dims
    embedding_model = config.models.embed

    profiles = load_all_entity_profiles(connection)
    qualifying = [p for p in profiles if p.mention_count >= min_mentions]

    db = connect_lancedb(store_paths.lancedb_path)
    entity_table = open_entity_table(db, vector_size=vector_size)

    existing_state = load_entity_embedding_state(connection)
    qualifying_ids = {p.entity_id for p in qualifying}
    stale_entity_ids = sorted(set(existing_state) - qualifying_ids)

    if stale_entity_ids:
        delete_entity_embedding_state(connection, stale_entity_ids)

    if not qualifying:
        # Still evict stale LanceDB rows so the table reflects current truth.
        if stale_entity_ids:
            upsert_entity_embeddings(entity_table, [], removed_entity_ids=stale_entity_ids)
        return 0

    chunk_table = open_chunk_table(db, vector_size=vector_size)
    timestamp = datetime.now(UTC).isoformat()
    records: list[dict[str, object]] = []
    pending_state: list[tuple[str, str, int]] = []  # (entity_id, source_hash, chunk_count)
    skipped = 0

    for profile in qualifying:
        chunk_ids = load_chunk_ids_for_entity(connection, profile.entity_id, limit=max_chunks)
        if not chunk_ids:
            continue

        # Source hash is purely structural — sorted chunk_ids + model identity.
        # The chunk vector itself is content-addressed in the embedding cache,
        # so identical chunk_id always implies identical vector at the same
        # (model, dims). Same source_hash here means the previous mean-pool is
        # still correct.
        sorted_chunk_ids = sorted(chunk_ids)
        source_hash = blake3_hex(
            embedding_model,
            str(vector_size),
            *sorted_chunk_ids,
        )

        if existing_state.get(profile.entity_id) == source_hash:
            skipped += 1
            continue

        # Fetch vectors from LanceDB (native float arrays, no JSON parsing)
        vectors_by_id = load_vectors_by_chunk_ids(chunk_table, chunk_ids)
        vectors = [v for v in vectors_by_id.values() if len(v) == vector_size]

        if not vectors:
            continue

        mean_vector = [sum(v[i] for v in vectors) / len(vectors) for i in range(vector_size)]
        # L2-normalise so cosine similarity search works correctly
        magnitude = sum(x * x for x in mean_vector) ** 0.5
        if magnitude > 0:
            mean_vector = [x / magnitude for x in mean_vector]

        records.append(
            {
                "entity_id": profile.entity_id,
                "label": profile.label,
                "community_id": profile.community_id if profile.community_id is not None else -1,
                "vector": mean_vector,
            }
        )
        pending_state.append((profile.entity_id, source_hash, len(sorted_chunk_ids)))

    if records or stale_entity_ids:
        upsert_entity_embeddings(entity_table, records, removed_entity_ids=stale_entity_ids)

    for entity_id, source_hash, chunk_count in pending_state:
        upsert_entity_embedding_state(
            connection,
            entity_id=entity_id,
            source_hash=source_hash,
            chunk_count=chunk_count,
            embedding_model=embedding_model,
            embedding_dims=vector_size,
            updated_at=timestamp,
        )

    _log.info(
        "entity embeddings computed",
        recomputed=len(records),
        skipped=skipped,
        evicted=len(stale_entity_ids),
    )
    return len(records)


# ---------------------------------------------------------------------------
# Batch API commands
# ---------------------------------------------------------------------------

_BATCH_ID_ARG: Final = typer.Argument(help="OpenAI batch ID to collect or check")


def collect_batch_command(
    batch_id: str = _BATCH_ID_ARG,
    profile: str | None = _PROFILE_OPTION,
    config_path: Path | None = _CONFIG_OPTION,
) -> None:
    """Collect results from a completed OpenAI Batch API job and insert into SQLite."""

    ctx = bootstrap_app(profile=profile, config_path=config_path)
    config = ctx.config
    store_paths = build_store_paths(config.paths.data_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    initialize_schema(connection)

    batch_dir = config.paths.data_path / "batch"
    chunks_meta_path = batch_dir / "claims_batch_chunks.json"
    if not chunks_meta_path.exists():
        # Fall back to durable copy in SQLite graph_metadata
        gm = load_graph_metadata(connection)
        stored = gm.get("claims_batch_chunks")
        if stored is None:
            _console.print(
                f"[red]Chunk metadata not found:[/red] {chunks_meta_path} "
                "(and no durable copy in SQLite)"
            )
            raise typer.Exit(1)
        chunks_meta_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_meta_path.write_text(stored)
        _console.print("[yellow]Restored batch metadata from SQLite.[/yellow]")

    status = _poll_batch(batch_id)
    if status["status"] != "completed":
        _console.print(f"[yellow]Batch status:[/yellow] {status['status']}")
        _console.print(f"  Completed: {status['request_counts']['completed']}")
        _console.print(f"  Failed: {status['request_counts']['failed']}")
        _console.print(f"  Total: {status['request_counts']['total']}")
        raise typer.Exit(1)

    claims_count = collect_claims_batch(batch_id, connection, config, chunks_meta_path)
    _console.print(f"[green]Collected {claims_count} claims from batch {batch_id}[/green]")

    # Now resume build-graph from the next phase
    _console.print("\nRun [bold]build-graph --phase entity_graph[/bold] to continue the pipeline.")


def batch_status_command(
    batch_id: str = _BATCH_ID_ARG,
    profile: str | None = _PROFILE_OPTION,
    config_path: Path | None = _CONFIG_OPTION,
) -> None:
    """Check the status of an OpenAI Batch API job."""
    bootstrap_app(profile=profile, config_path=config_path)

    status = _poll_batch(batch_id)
    table = Table(title=f"Batch {batch_id}")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Status", status["status"])
    table.add_row("Total requests", str(status["request_counts"]["total"]))
    table.add_row("Completed", str(status["request_counts"]["completed"]))
    table.add_row("Failed", str(status["request_counts"]["failed"]))
    if status.get("output_file_id"):
        table.add_row("Output file", status["output_file_id"])
    if status.get("error_file_id"):
        table.add_row("Error file", status["error_file_id"])
    _console.print(table)
