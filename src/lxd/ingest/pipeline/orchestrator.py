"""Top-level ingest run orchestrator: plan, execute, persist, finish."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

from lxd.domain.status import LifecycleStatus, RetrievalStatus
from lxd.ingest.assets import infer_asset_parent
from lxd.ingest.budget import BudgetExceededError, IngestBudgetTracker
from lxd.ingest.contextual_chunker import open_summary_cache_table
from lxd.ingest.embedder import probe_embedder
from lxd.ingest.embedding_cache import open_cache_table
from lxd.ingest.error_classification import (
    CircuitBreakerTripped,
    PersistentCircuitBreaker,
    classify,
)
from lxd.ingest.pipeline.moves import (
    can_skip_unchanged_source,
    clone_source_records,
    find_move_source,
    resolve_document_id,
)
from lxd.ingest.pipeline.sources import build_manifest_record, build_source_records
from lxd.ingest.relations import build_valid_predicates
from lxd.ingest.scanner import ScannedCorpusFile, scan_corpus
from lxd.ingest.wiki_relations import build_slug_index, derive_wiki_link_relations
from lxd.ontology.loader import OntologyLoadResult, load_ontology
from lxd.ontology.matcher import build_or_load_automaton
from lxd.settings.models import RuntimeConfig
from lxd.stores.lancedb import (
    connect_lancedb,
    open_chunk_table,
    refresh_fts_index,
)
from lxd.stores.lancedb import delete_source as delete_vector_source
from lxd.stores.lancedb import (
    replace_source_chunks as replace_vector_source_chunks,
)
from lxd.stores.models import (
    AssetLinkRecord,
    CorpusStatusSummary,
    IngestConfigSnapshotRecord,
    ManifestRecord,
    OntologySnapshotRecord,
    OntologySourceRecord,
)
from lxd.stores.sqlite.chunks import (
    replace_source_chunks as replace_sqlite_source_chunks,
)
from lxd.stores.sqlite.connection import (
    build_store_paths,
    connect_sqlite,
    initialize_schema,
)
from lxd.stores.sqlite.manifest import (
    delete_source as delete_sqlite_source,
)
from lxd.stores.sqlite.manifest import (
    load_manifest_by_content_hash,
    load_manifest_index,
    upsert_asset_link,
    upsert_manifest_record,
)
from lxd.stores.sqlite.ontology import (
    replace_ingest_config_snapshot,
    replace_ontology_snapshot,
    replace_ontology_sources,
)
from lxd.stores.sqlite.runs import (
    begin_ingest_run,
    finish_ingest_run,
    update_ingest_run_progress,
)
from lxd.stores.sqlite.summary import summarize_store

_log = structlog.get_logger(__name__)

_RECOVERABLE_SOURCE_ERRORS = (
    FileNotFoundError,
    OSError,
    RuntimeError,
    ValueError,
    sqlite3.Error,
)


@dataclass(frozen=True, slots=True)
class IngestPlan:
    """Resolved scan and ontology inputs for an ingest run."""

    scanned_files: list[ScannedCorpusFile]
    ontology: OntologyLoadResult


@dataclass(frozen=True, slots=True)
class IngestRunResult:
    """Outcome details and counters from an ingest run."""

    run_id: str
    summary: CorpusStatusSummary
    entity_count: int
    warnings: list[str]
    reembedded_text_sources: int
    reused_move_sources: int
    snapshot_path: Path


def validate_project_paths(config: RuntimeConfig) -> None:
    """Validate configuration and apply runtime settings."""
    if not config.paths.corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus path: {config.paths.corpus_path}")
    if not config.paths.ontology_path.exists():
        raise FileNotFoundError(f"Missing ontology path: {config.paths.ontology_path}")
    config.paths.data_path.mkdir(parents=True, exist_ok=True)


def build_ingest_plan(config: RuntimeConfig) -> IngestPlan:
    """Build an ingest plan from corpus scan and ontology load."""
    validate_project_paths(config)
    scanned_files = scan_corpus(
        corpus_root=config.paths.corpus_path,
        text_extensions=config.corpus.text_extensions,
        asset_extensions=config.corpus.asset_extensions,
        ignore_names=config.corpus.ignore_names,
    )
    ontology = load_ontology(
        root=config.paths.ontology_path,
        include_globs=config.ontology.include_globs,
        ignore_names=config.ontology.ignore_names,
    )
    return IngestPlan(scanned_files=scanned_files, ontology=ontology)


def run_ingest(config: RuntimeConfig, *, full_rebuild: bool = False) -> IngestRunResult:
    """Execute the ingestion pipeline and persist results."""
    plan = build_ingest_plan(config)
    _validate_ingest_dependencies(config)

    warnings: list[str] = []
    automaton = build_or_load_automaton(
        plan.ontology.matcher_records,
        cache_dir=config.paths.data_path / "matcher_cache",
    )
    valid_predicates = build_valid_predicates(plan.ontology.relation_records)
    slug_index = build_slug_index(plan.ontology.entity_definitions)
    budget_tracker = IngestBudgetTracker(config.ingest_budget)
    wiki_dangling_total: set[str] = set()
    wiki_pages_without_subject_total: set[str] = set()
    store_paths = build_store_paths(config.paths.data_path)
    sqlite_connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(sqlite_connection)
        vector_db = connect_lancedb(store_paths.lancedb_path)
        vector_table = open_chunk_table(vector_db, vector_size=config.models.embed_dims)
        cache_table = open_cache_table(vector_db, vector_size=config.models.embed_dims)
        contextual_summary_table = (
            open_summary_cache_table(vector_db)
            if config.chunking.contextual_summary_enabled
            else None
        )
        circuit_breaker = PersistentCircuitBreaker(sqlite_connection, threshold=3)
        cache_hit_total = 0
        cache_miss_total = 0

        run_id = f"ingest-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        timestamp = utc_now()
        begin_ingest_run(
            sqlite_connection,
            run_id=run_id,
            started_at=timestamp,
            mode="full" if full_rebuild else "incremental",
            files_total=len(plan.scanned_files),
        )

        replace_ontology_sources(
            sqlite_connection,
            [
                OntologySourceRecord(
                    file_rel_path=source.file_rel_path,
                    blake3_hash=source.blake3_hash,
                    last_seen_at=timestamp,
                )
                for source in plan.ontology.sources
            ],
        )
        replace_ontology_snapshot(
            sqlite_connection,
            OntologySnapshotRecord(
                snapshot_id="current",
                ontology_root=str(config.paths.ontology_path),
                snapshot_hash=plan.ontology.snapshot_hash,
                matcher_termset_hash=plan.ontology.matcher_termset_hash,
                matcher_term_count=len(plan.ontology.matcher_records),
                source_file_count=len(plan.ontology.sources),
                entity_file_count=len(plan.ontology.sources),
                entity_count=len(plan.ontology.entity_definitions),
                coverage_path_count=plan.ontology.coverage_report.discovered_path_count,
                graph_relation_count=len(plan.ontology.relation_records),
                validation_issue_count=len(plan.ontology.validation_issues),
                validation_issues_json=json.dumps(
                    [issue.message for issue in plan.ontology.validation_issues],
                    ensure_ascii=True,
                ),
                last_loaded_at=timestamp,
            ),
        )

        # Always load the manifest so we can clean up deleted sources.
        # For full rebuild, skip hash lookups so every file is force-reprocessed.
        current_manifest = load_manifest_index(sqlite_connection)
        existing_by_path = {} if full_rebuild else current_manifest
        existing_by_hash = {} if full_rebuild else load_manifest_by_content_hash(sqlite_connection)
        manifest_by_rel_path = dict(existing_by_path)
        scanned_rel_paths = {item.relative_path for item in plan.scanned_files}

        for missing_rel_path in sorted(set(current_manifest) - scanned_rel_paths):
            missing_manifest = current_manifest[missing_rel_path]
            delete_sqlite_source(sqlite_connection, missing_manifest.source_rel_path)
            delete_vector_source(vector_table, missing_manifest.source_rel_path)

        reembedded_text_sources = 0
        reused_move_sources = 0
        files_completed = 0
        searchable_files_rebuilt = 0
        asset_files_processed = 0
        unchanged_files_skipped = 0
        failed_files = 0
        chunks_written = 0

        try:
            for scanned in plan.scanned_files:
                unchanged = existing_by_path.get(scanned.relative_path)
                if (
                    unchanged is not None
                    and unchanged.content_hash == scanned.content_hash
                    and not full_rebuild
                    and can_skip_unchanged_source(sqlite_connection, scanned, unchanged)
                ):
                    manifest_by_rel_path[scanned.relative_path] = unchanged
                    files_completed += 1
                    unchanged_files_skipped += 1
                    _persist_ingest_progress(
                        sqlite_connection=sqlite_connection,
                        run_id=run_id,
                        files_completed=files_completed,
                        searchable_files_rebuilt=searchable_files_rebuilt,
                        asset_files_processed=asset_files_processed,
                        unchanged_files_skipped=unchanged_files_skipped,
                        failed_files=failed_files,
                        chunks_written=chunks_written,
                        warnings=warnings,
                    )
                    continue

                if scanned.source_type == "image_png":
                    manifest_record = build_manifest_record(
                        scanned=scanned,
                        document_id=None,
                        parent_source_rel_path=None,
                        chunk_count=0,
                        timestamp=timestamp,
                        lifecycle_status=LifecycleStatus.PROCESSING,
                        retrieval_status=RetrievalStatus.ASSET_ONLY,
                        error_message=None,
                    )
                    upsert_manifest_record(sqlite_connection, manifest_record)
                    asset_link = infer_asset_parent(scanned.relative_path)
                    parent_manifest = (
                        manifest_by_rel_path.get(asset_link.parent_rel_path)
                        if asset_link.parent_rel_path
                        else None
                    )
                    committed_manifest = ManifestRecord(
                        source_rel_path=manifest_record.source_rel_path,
                        absolute_path=manifest_record.absolute_path,
                        source_type=manifest_record.source_type,
                        source_domain=manifest_record.source_domain,
                        document_id=manifest_record.document_id,
                        file_size_bytes=manifest_record.file_size_bytes,
                        content_hash=manifest_record.content_hash,
                        parent_source_rel_path=parent_manifest.source_rel_path
                        if parent_manifest
                        else None,
                        chunk_count=manifest_record.chunk_count,
                        last_seen_at=manifest_record.last_seen_at,
                        last_processed_at=timestamp,
                        last_committed_at=timestamp,
                        error_message=None,
                        lifecycle_status=LifecycleStatus.COMPLETE,
                        retrieval_status=RetrievalStatus.ASSET_ONLY,
                    )
                    upsert_manifest_record(sqlite_connection, committed_manifest)
                    upsert_asset_link(
                        sqlite_connection,
                        AssetLinkRecord(
                            asset_rel_path=scanned.relative_path,
                            asset_filename=scanned.absolute_path.name,
                            source_domain=scanned.source_domain,
                            parent_source_rel_path=parent_manifest.source_rel_path
                            if parent_manifest
                            else None,
                            parent_document_id=parent_manifest.document_id
                            if parent_manifest
                            else None,
                            page_no=asset_link.page_no,
                            asset_index=None,
                            link_method=asset_link.link_method,
                            blake3_hash=scanned.content_hash,
                            last_committed_at=timestamp,
                        ),
                    )
                    manifest_by_rel_path[scanned.relative_path] = committed_manifest
                    files_completed += 1
                    asset_files_processed += 1
                    _persist_ingest_progress(
                        sqlite_connection=sqlite_connection,
                        run_id=run_id,
                        files_completed=files_completed,
                        searchable_files_rebuilt=searchable_files_rebuilt,
                        asset_files_processed=asset_files_processed,
                        unchanged_files_skipped=unchanged_files_skipped,
                        failed_files=failed_files,
                        chunks_written=chunks_written,
                        warnings=warnings,
                    )
                    continue

                previous_manifest = existing_by_path.get(scanned.relative_path)
                move_source = (
                    None
                    if full_rebuild
                    else find_move_source(scanned, existing_by_hash, scanned_rel_paths)
                )
                document_id = resolve_document_id(scanned, previous_manifest, move_source)
                processing_manifest = build_manifest_record(
                    scanned=scanned,
                    document_id=document_id,
                    parent_source_rel_path=None,
                    chunk_count=0,
                    timestamp=timestamp,
                    lifecycle_status=LifecycleStatus.PROCESSING,
                    retrieval_status=RetrievalStatus.NOT_SEARCHABLE,
                    error_message=None,
                )
                upsert_manifest_record(sqlite_connection, processing_manifest)

                try:
                    if move_source is not None:
                        cloned_chunks, cloned_mentions = clone_source_records(
                            sqlite_connection=sqlite_connection,
                            vector_table=vector_table,
                            old_manifest=move_source,
                            new_scanned=scanned,
                            document_id=document_id,
                        )
                        delete_sqlite_source(sqlite_connection, move_source.source_rel_path)
                        delete_vector_source(vector_table, move_source.source_rel_path)
                        # The page's wiki_links travel with the cloned chunks, but
                        # the subject (resolved from the new filename stem) may
                        # change for cross-directory moves. Re-derive against the
                        # current path so KG edges follow the file.
                        cloned_wiki = derive_wiki_link_relations(
                            chunk_records=cloned_chunks,
                            slug_index=slug_index,
                            extracted_at=utc_now(),
                        )
                        wiki_dangling_total.update(cloned_wiki.dangling_slugs)
                        wiki_pages_without_subject_total.update(cloned_wiki.pages_without_subject)
                        replace_sqlite_source_chunks(
                            sqlite_connection,
                            source_rel_path=scanned.relative_path,
                            chunk_records=cloned_chunks,
                            mention_records=cloned_mentions,
                            relation_records=cloned_wiki.relations,
                        )
                        replace_vector_source_chunks(
                            vector_table, scanned.relative_path, cloned_chunks
                        )
                        chunk_records = cloned_chunks
                        mention_records = cloned_mentions
                        reused_move_sources += 1
                    else:
                        (
                            chunk_records,
                            mention_records,
                            relation_records,
                            file_cache_hits,
                            file_cache_misses,
                            file_wiki_dangling,
                            file_wiki_no_subject,
                        ) = build_source_records(
                            scanned=scanned,
                            document_id=document_id,
                            config=config,
                            automaton=automaton,
                            valid_predicates=valid_predicates,
                            slug_index=slug_index,
                            budget_tracker=budget_tracker,
                            cache_table=cache_table,
                            contextual_summary_table=contextual_summary_table,
                        )
                        cache_hit_total += file_cache_hits
                        cache_miss_total += file_cache_misses
                        wiki_dangling_total.update(file_wiki_dangling)
                        wiki_pages_without_subject_total.update(file_wiki_no_subject)

                        # LanceDB FIRST. If LanceDB succeeds and SQLite fails,
                        # the LanceDB write is left in place — it's content-
                        # addressed and a future re-ingest of the same file
                        # will replace it cleanly. The compensating delete
                        # below restores LanceDB to its pre-write state if
                        # SQLite fails so query results stay consistent with
                        # the (older) SQLite manifest.
                        replace_vector_source_chunks(
                            vector_table, scanned.relative_path, chunk_records
                        )
                        try:
                            replace_sqlite_source_chunks(
                                sqlite_connection,
                                source_rel_path=scanned.relative_path,
                                chunk_records=chunk_records,
                                mention_records=mention_records,
                                relation_records=relation_records,
                            )
                        except sqlite3.Error:
                            # Compensate: undo the LanceDB write so the two
                            # stores stay coherent. Best-effort — if the
                            # delete itself fails we propagate the original
                            # SQLite error.
                            with contextlib.suppress(FileNotFoundError, ValueError, RuntimeError):
                                delete_vector_source(vector_table, scanned.relative_path)
                            raise
                        reembedded_text_sources += 1

                    committed_manifest = build_manifest_record(
                        scanned=scanned,
                        document_id=document_id,
                        parent_source_rel_path=None,
                        chunk_count=len(chunk_records),
                        timestamp=timestamp,
                        lifecycle_status=LifecycleStatus.COMPLETE,
                        retrieval_status=RetrievalStatus.SEARCHABLE,
                        error_message=None,
                    )
                    upsert_manifest_record(sqlite_connection, committed_manifest)
                    manifest_by_rel_path[scanned.relative_path] = committed_manifest
                    files_completed += 1
                    searchable_files_rebuilt += 1
                    chunks_written += len(chunk_records)
                    circuit_breaker.record_success()
                except _RECOVERABLE_SOURCE_ERRORS as exc:
                    failed_manifest = build_manifest_record(
                        scanned=scanned,
                        document_id=document_id,
                        parent_source_rel_path=None,
                        chunk_count=0,
                        timestamp=timestamp,
                        lifecycle_status=LifecycleStatus.FAILED,
                        retrieval_status=RetrievalStatus.NOT_SEARCHABLE,
                        error_message=str(exc),
                    )
                    upsert_manifest_record(sqlite_connection, failed_manifest)
                    err_class = classify(exc).value
                    warnings.append(f"{scanned.relative_path}: [{err_class}] {exc}")
                    files_completed += 1
                    failed_files += 1
                    # May raise CircuitBreakerTripped on the Nth consecutive
                    # systemic failure. Caught at the outer try below.
                    circuit_breaker.record_failure(exc)
                _persist_ingest_progress(
                    sqlite_connection=sqlite_connection,
                    run_id=run_id,
                    files_completed=files_completed,
                    searchable_files_rebuilt=searchable_files_rebuilt,
                    asset_files_processed=asset_files_processed,
                    unchanged_files_skipped=unchanged_files_skipped,
                    failed_files=failed_files,
                    chunks_written=chunks_written,
                    warnings=warnings,
                )

            replace_ingest_config_snapshot(sqlite_connection, _config_snapshot_records(config))
            summary = summarize_store(
                sqlite_connection,
                ontology_file_count=len(plan.ontology.sources),
                matcher_term_count=len(plan.ontology.matcher_records),
                matcher_termset_hash=plan.ontology.matcher_termset_hash,
                ontology_snapshot_hash=plan.ontology.snapshot_hash,
                ontology_coverage_path_count=plan.ontology.coverage_report.discovered_path_count,
                ontology_graph_relation_count=len(plan.ontology.relation_records),
                ontology_validation_issue_count=len(plan.ontology.validation_issues),
                ontology_validation_issue_samples=[
                    issue.message for issue in plan.ontology.validation_issues[:10]
                ],
            )
            snapshot_path = persist_ingest_snapshot(
                config,
                summary=summary,
                entity_count=len(plan.ontology.entity_definitions),
            )
            finish_ingest_run(
                sqlite_connection,
                run_id=run_id,
                finished_at=utc_now(),
                status="complete" if not warnings else "complete_with_warnings",
                files_completed=files_completed,
                searchable_files_rebuilt=searchable_files_rebuilt,
                asset_files_processed=asset_files_processed,
                unchanged_files_skipped=unchanged_files_skipped,
                failed_files=failed_files,
                chunks_written=chunks_written,
                notes=warnings,
                embedding_cache_hits=cache_hit_total,
                embedding_cache_misses=cache_miss_total,
            )
            # Native LanceDB FTS does not auto-include rows added after
            # index creation; rebuild the index once at the end of the run
            # so retrieval BM25 sees every chunk written this run.
            refresh_fts_index(vector_table)
            if wiki_dangling_total or wiki_pages_without_subject_total:
                _log.info(
                    "wiki_relation_derivation_diagnostics",
                    dangling_slug_count=len(wiki_dangling_total),
                    dangling_slug_samples=sorted(wiki_dangling_total)[:20],
                    pages_without_subject_count=len(wiki_pages_without_subject_total),
                    pages_without_subject_samples=sorted(wiki_pages_without_subject_total)[:20],
                )
            return IngestRunResult(
                run_id=run_id,
                summary=summary,
                entity_count=len(plan.ontology.entity_definitions),
                warnings=warnings,
                reembedded_text_sources=reembedded_text_sources,
                reused_move_sources=reused_move_sources,
                snapshot_path=snapshot_path,
            )
        except CircuitBreakerTripped as exc:
            # Systemic failure: stop spending API budget and surface loudly.
            failure_notes = [
                *warnings,
                f"aborted: circuit-breaker tripped after {exc.count} consecutive systemic errors",
                f"last error: {type(exc.last_error).__name__}: {exc.last_error}",
            ]
            finish_ingest_run(
                sqlite_connection,
                run_id=run_id,
                finished_at=utc_now(),
                status="aborted",
                files_completed=files_completed,
                searchable_files_rebuilt=searchable_files_rebuilt,
                asset_files_processed=asset_files_processed,
                unchanged_files_skipped=unchanged_files_skipped,
                failed_files=failed_files,
                chunks_written=chunks_written,
                notes=failure_notes,
                embedding_cache_hits=cache_hit_total,
                embedding_cache_misses=cache_miss_total,
            )
            raise
        except BudgetExceededError as exc:
            failure_notes = [
                *warnings,
                f"aborted: {exc}",
                f"llm_calls_at_abort={budget_tracker.llm_calls}",
            ]
            finish_ingest_run(
                sqlite_connection,
                run_id=run_id,
                finished_at=utc_now(),
                status="aborted_budget",
                files_completed=files_completed,
                searchable_files_rebuilt=searchable_files_rebuilt,
                asset_files_processed=asset_files_processed,
                unchanged_files_skipped=unchanged_files_skipped,
                failed_files=failed_files,
                chunks_written=chunks_written,
                notes=failure_notes,
                embedding_cache_hits=cache_hit_total,
                embedding_cache_misses=cache_miss_total,
            )
            raise
        except _RECOVERABLE_SOURCE_ERRORS as exc:
            failure_notes = [*warnings, f"fatal: {exc}"]
            finish_ingest_run(
                sqlite_connection,
                run_id=run_id,
                finished_at=utc_now(),
                status="failed",
                files_completed=files_completed,
                searchable_files_rebuilt=searchable_files_rebuilt,
                asset_files_processed=asset_files_processed,
                unchanged_files_skipped=unchanged_files_skipped,
                failed_files=failed_files + 1,
                chunks_written=chunks_written,
                notes=failure_notes,
                embedding_cache_hits=cache_hit_total,
                embedding_cache_misses=cache_miss_total,
            )
            raise
    finally:
        sqlite_connection.close()


def _persist_ingest_progress(
    *,
    sqlite_connection: sqlite3.Connection,
    run_id: str,
    files_completed: int,
    searchable_files_rebuilt: int,
    asset_files_processed: int,
    unchanged_files_skipped: int,
    failed_files: int,
    chunks_written: int,
    warnings: list[str],
) -> None:
    update_ingest_run_progress(
        sqlite_connection,
        run_id=run_id,
        files_completed=files_completed,
        searchable_files_rebuilt=searchable_files_rebuilt,
        asset_files_processed=asset_files_processed,
        unchanged_files_skipped=unchanged_files_skipped,
        failed_files=failed_files,
        chunks_written=chunks_written,
        notes=warnings,
    )


def persist_ingest_snapshot(
    config: RuntimeConfig,
    *,
    summary: CorpusStatusSummary,
    entity_count: int,
) -> Path:
    """Write the latest ingest summary snapshot JSON."""
    config.paths.data_path.mkdir(parents=True, exist_ok=True)
    output_path = config.paths.data_path / "ingest_snapshot.json"
    payload = {
        "corpus_counts": {
            "total": summary.corpus_file_count,
            "text": summary.text_file_count,
            "asset": summary.asset_file_count,
        },
        "retrieval_role_counts": summary.retrieval_role_counts,
        "chunk_count": summary.chunk_count,
        "mention_count": summary.mention_count,
        "ontology_file_count": summary.ontology_file_count,
        "entity_count": entity_count,
        "matcher_term_count": summary.matcher_term_count,
        "ontology_snapshot_hash": summary.ontology_snapshot_hash,
        "matcher_termset_hash": summary.matcher_termset_hash,
        "ontology_coverage_path_count": summary.ontology_coverage_path_count,
        "ontology_graph_relation_count": summary.ontology_graph_relation_count,
        "ontology_validation_issue_count": summary.ontology_validation_issue_count,
        "ontology_validation_issue_samples": summary.ontology_validation_issue_samples,
        "config_drift_warnings": summary.config_drift_warnings,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _validate_ingest_dependencies(config: RuntimeConfig) -> None:
    embed_probe = probe_embedder(config)
    if not embed_probe.ok:
        raise RuntimeError(f"Embedding readiness probe failed: {embed_probe.warning}")


def _config_snapshot_records(config: RuntimeConfig) -> list[IngestConfigSnapshotRecord]:
    snapshot = {
        "paths.corpus_path": str(config.paths.corpus_path),
        "paths.ontology_path": str(config.paths.ontology_path),
        "paths.data_path": str(config.paths.data_path),
        "models.embed": config.models.embed,
        "models.embed_backend": config.models.embed_backend,
        "models.embed_dims": str(config.models.embed_dims),
        "chunking.strategy": config.chunking.strategy,
        "chunking.chunk_size": str(config.chunking.chunk_size),
        "chunking.chunk_overlap": str(config.chunking.chunk_overlap),
        "chunking.min_tokens": str(config.chunking.min_tokens),
        "chunking.tokenizer_backend": config.chunking.tokenizer_backend,
        "chunking.tokenizer_name": config.chunking.tokenizer_name,
    }
    return [
        IngestConfigSnapshotRecord(key=key, value=value) for key, value in sorted(snapshot.items())
    ]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
