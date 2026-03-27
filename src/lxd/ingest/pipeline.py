"""Plan and execute end-to-end ingestion and persistence steps."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lxd.domain.citations import make_citation_label
from lxd.domain.ids import blake3_hex, make_chunk_id
from lxd.domain.status import LifecycleStatus, RetrievalStatus
from lxd.ingest.assets import infer_asset_parent
from lxd.ingest.chunking import (
    TextChunk,
    build_tokenizer,
    chunk_document,
    split_chunk_for_context,
    token_count_with_tokenizer,
)
from lxd.ingest.docling import load_docling_document
from lxd.ingest.embedder import EmbeddingContextError, embed_chunk_text, probe_embedder
from lxd.ingest.markdown import ExtractedDocument, load_markdown_document
from lxd.ingest.mentions import detect_mentions
from lxd.ingest.relations import build_valid_predicates, extract_relations_for_chunk
from lxd.ingest.scanner import ScannedCorpusFile, scan_corpus
from lxd.ontology.loader import OntologyLoadResult, load_ontology
from lxd.ontology.matcher import build_automaton
from lxd.settings.models import RuntimeConfig
from lxd.stores.lancedb import (
    connect_lancedb,
    open_chunk_table,
    reset_chunk_table,
)
from lxd.stores.lancedb import delete_source as delete_vector_source
from lxd.stores.lancedb import (
    replace_source_chunks as replace_vector_source_chunks,
)
from lxd.stores.models import (
    AssetLinkRecord,
    ChunkRecord,
    CorpusStatusSummary,
    ExtractedRelationRecord,
    IngestConfigSnapshotRecord,
    ManifestRecord,
    MentionRecord,
    OntologySnapshotRecord,
    OntologySourceRecord,
)
from lxd.stores.sqlite import (
    begin_ingest_run,
    build_store_paths,
    connect_sqlite,
    finish_ingest_run,
    initialize_schema,
    load_chunk_records_for_source,
    load_manifest_by_content_hash,
    load_manifest_index,
    load_mentions_for_source,
    replace_ingest_config_snapshot,
    replace_ontology_snapshot,
    replace_ontology_sources,
    reset_store,
    summarize_store,
    update_ingest_run_progress,
    upsert_asset_link,
    upsert_manifest_record,
)
from lxd.stores.sqlite import (
    delete_source as delete_sqlite_source,
)
from lxd.stores.sqlite import (
    replace_source_chunks as replace_sqlite_source_chunks,
)

_RECOVERABLE_SOURCE_ERRORS = (
    FileNotFoundError,
    OSError,
    RuntimeError,
    ValueError,
    sqlite3.Error,
)


@dataclass(frozen=True)
class IngestPlan:
    """Resolved scan and ontology inputs for an ingest run."""
    scanned_files: list[ScannedCorpusFile]
    ontology: OntologyLoadResult


@dataclass(frozen=True)
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
    """Validate configuration and apply runtime settings.

    Args:
        config: Runtime configuration object.
    """
    if not config.paths.corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus path: {config.paths.corpus_path}")
    if not config.paths.ontology_path.exists():
        raise FileNotFoundError(f"Missing ontology path: {config.paths.ontology_path}")
    config.paths.data_path.mkdir(parents=True, exist_ok=True)


def build_ingest_plan(config: RuntimeConfig) -> IngestPlan:
    """Build an ingest plan from corpus scan and ontology load.

    Args:
        config: Runtime configuration object.

    Returns:
        Planned corpus scan and ontology snapshot.
    """
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
    """Execute the ingestion pipeline and persist results.

    Args:
        config: Runtime configuration object.
        full_rebuild: Whether to rebuild stores from scratch.

    Returns:
        Completed ingest run summary and diagnostics.
    """
    plan = build_ingest_plan(config)
    _validate_ingest_dependencies(config)

    warnings: list[str] = []
    automaton = build_automaton(plan.ontology.matcher_records)
    valid_predicates = build_valid_predicates(plan.ontology.relation_records)
    store_paths = build_store_paths(config.paths.data_path)
    sqlite_connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(sqlite_connection)
        vector_db = connect_lancedb(store_paths.lancedb_path)
        vector_table = (
            reset_chunk_table(vector_db, vector_size=config.models.embed_dims)
            if full_rebuild
            else open_chunk_table(vector_db, vector_size=config.models.embed_dims)
        )
        if full_rebuild:
            reset_store(sqlite_connection)

        run_id = f"ingest-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        timestamp = _utc_now()
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
                    file_path=str(source.file_path),
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

        existing_by_path = {} if full_rebuild else load_manifest_index(sqlite_connection)
        existing_by_hash = {} if full_rebuild else load_manifest_by_content_hash(sqlite_connection)
        manifest_by_rel_path = dict(existing_by_path)
        scanned_rel_paths = {item.relative_path for item in plan.scanned_files}

        if not full_rebuild:
            for missing_rel_path in sorted(set(existing_by_path) - scanned_rel_paths):
                missing_manifest = existing_by_path[missing_rel_path]
                delete_sqlite_source(sqlite_connection, missing_manifest.absolute_path)
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
                    and _can_skip_unchanged_source(sqlite_connection, scanned, unchanged)
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
                    manifest_record = _manifest_record(
                        scanned=scanned,
                        document_id=None,
                        parent_source_path=None,
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
                        parent_source_path=parent_manifest.absolute_path if parent_manifest else None,
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
                        scanned.absolute_path.as_posix(),
                        AssetLinkRecord(
                            asset_rel_path=scanned.relative_path,
                            asset_filename=scanned.absolute_path.name,
                            source_domain=scanned.source_domain,
                            parent_source_path=parent_manifest.absolute_path
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
                    else _find_move_source(scanned, existing_by_hash, scanned_rel_paths)
                )
                document_id = _resolve_document_id(scanned, previous_manifest, move_source, timestamp)
                processing_manifest = _manifest_record(
                    scanned=scanned,
                    document_id=document_id,
                    parent_source_path=None,
                    chunk_count=0,
                    timestamp=timestamp,
                    lifecycle_status=LifecycleStatus.PROCESSING,
                    retrieval_status=RetrievalStatus.NOT_SEARCHABLE,
                    error_message=None,
                )
                upsert_manifest_record(sqlite_connection, processing_manifest)

                try:
                    if move_source is not None:
                        cloned_chunks, cloned_mentions = _clone_source_records(
                            sqlite_connection=sqlite_connection,
                            old_manifest=move_source,
                            new_scanned=scanned,
                            document_id=document_id,
                        )
                        delete_sqlite_source(sqlite_connection, move_source.absolute_path)
                        delete_vector_source(vector_table, move_source.source_rel_path)
                        replace_sqlite_source_chunks(
                            sqlite_connection,
                            absolute_source_path=scanned.absolute_path.as_posix(),
                            chunk_records=cloned_chunks,
                            mention_records=cloned_mentions,
                            relation_records=[],
                        )
                        replace_vector_source_chunks(
                            vector_table, scanned.relative_path, cloned_chunks
                        )
                        chunk_records = cloned_chunks
                        mention_records = cloned_mentions
                        reused_move_sources += 1
                    else:
                        chunk_records, mention_records, relation_records = _build_source_records(
                            scanned=scanned,
                            document_id=document_id,
                            config=config,
                            automaton=automaton,
                            valid_predicates=valid_predicates,
                        )
                        replace_sqlite_source_chunks(
                            sqlite_connection,
                            absolute_source_path=scanned.absolute_path.as_posix(),
                            chunk_records=chunk_records,
                            mention_records=mention_records,
                            relation_records=relation_records,
                        )
                        replace_vector_source_chunks(
                            vector_table, scanned.relative_path, chunk_records
                        )
                        reembedded_text_sources += 1

                    committed_manifest = _manifest_record(
                        scanned=scanned,
                        document_id=document_id,
                        parent_source_path=None,
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
                except _RECOVERABLE_SOURCE_ERRORS as exc:
                    failed_manifest = _manifest_record(
                        scanned=scanned,
                        document_id=document_id,
                        parent_source_path=None,
                        chunk_count=0,
                        timestamp=timestamp,
                        lifecycle_status=LifecycleStatus.FAILED,
                        retrieval_status=RetrievalStatus.NOT_SEARCHABLE,
                        error_message=str(exc),
                    )
                    upsert_manifest_record(sqlite_connection, failed_manifest)
                    warnings.append(f"{scanned.relative_path}: {exc}")
                    files_completed += 1
                    failed_files += 1
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
                finished_at=_utc_now(),
                status="complete" if not warnings else "complete_with_warnings",
                files_completed=files_completed,
                searchable_files_rebuilt=searchable_files_rebuilt,
                asset_files_processed=asset_files_processed,
                unchanged_files_skipped=unchanged_files_skipped,
                failed_files=failed_files,
                chunks_written=chunks_written,
                notes=warnings,
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
        except _RECOVERABLE_SOURCE_ERRORS as exc:
            failure_notes = [*warnings, f"fatal: {exc}"]
            finish_ingest_run(
                sqlite_connection,
                run_id=run_id,
                finished_at=_utc_now(),
                status="failed",
                files_completed=files_completed,
                searchable_files_rebuilt=searchable_files_rebuilt,
                asset_files_processed=asset_files_processed,
                unchanged_files_skipped=unchanged_files_skipped,
                failed_files=failed_files + 1,
                chunks_written=chunks_written,
                notes=failure_notes,
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
    """Write the latest ingest summary snapshot JSON.

    Args:
        config: Runtime configuration object.
        summary: Current corpus status summary.
        entity_count: Total ontology entity count.

    Returns:
        Path to the written ingest snapshot JSON.
    """
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


def _build_source_records(
    *,
    scanned: ScannedCorpusFile,
    document_id: str,
    config: RuntimeConfig,
    automaton: object,
    valid_predicates: frozenset[str],
) -> tuple[list[ChunkRecord], list[MentionRecord], list[ExtractedRelationRecord]]:
    extracted_document = _load_extracted_document(scanned)
    initial_chunks = chunk_document(
        extracted_document,
        document_id=document_id,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        min_tokens=config.chunking.min_tokens,
        tokenizer_backend=config.chunking.tokenizer_backend,
        tokenizer_name=config.chunking.tokenizer_name,
        strategy=config.chunking.strategy,
    )
    text_chunks, embeddings = _embed_with_context_refinement(initial_chunks, document_id, config)
    chunk_records: list[ChunkRecord] = []
    mention_records: list[MentionRecord] = []
    relation_records: list[ExtractedRelationRecord] = []
    for chunk, vector in zip(text_chunks, embeddings, strict=True):
        chunk_record = ChunkRecord(
            chunk_id=chunk.chunk_id,
            document_id=document_id,
            source_rel_path=chunk.source_rel_path,
            source_path=scanned.absolute_path.as_posix(),
            source_filename=scanned.absolute_path.name,
            source_type=chunk.source_type,
            source_domain=scanned.source_domain,
            source_hash=scanned.content_hash,
            citation_label=chunk.citation_label,
            chunk_index=chunk.chunk_index,
            chunk_occurrence=chunk.chunk_occurrence,
            token_count=chunk.token_count,
            text=chunk.text,
            chunk_hash=chunk.chunk_hash,
            score_hint=chunk.score_hint,
            metadata_json=chunk.metadata_json,
            vector=vector,
            embedding_model=config.models.embed,
            embedding_dims=config.models.embed_dims,
        )
        chunk_records.append(chunk_record)
        chunk_mentions = list(
            MentionRecord(
                chunk_id=chunk_record.chunk_id,
                entity_id=mention.entity_id,
                term_source=mention.term_source,
                surface_form=mention.surface_form,
                start_char=mention.start_char,
                end_char=mention.end_char,
            )
            for mention in detect_mentions(chunk.text, automaton)
        )
        mention_records.extend(chunk_mentions)
        relation_records.extend(
            extract_relations_for_chunk(
                chunk_id=chunk_record.chunk_id,
                document_id=document_id,
                source_rel_path=chunk_record.source_rel_path,
                chunk_text=chunk.text,
                mention_records=chunk_mentions,
                valid_predicates=valid_predicates,
                config=config,
            )
        )
    return chunk_records, mention_records, relation_records


def _load_extracted_document(scanned: ScannedCorpusFile) -> ExtractedDocument:
    if scanned.source_type in ("markdown", "docling_md"):
        return load_markdown_document(
            scanned.absolute_path,
            scanned.relative_path,
            source_type=scanned.source_type,
        )
    return load_docling_document(scanned.absolute_path, scanned.relative_path)


def _embed_with_context_refinement(
    chunks: list[TextChunk],
    document_id: str,
    config: RuntimeConfig,
) -> tuple[list[TextChunk], list[list[float]]]:
    token_counter = token_count_with_tokenizer(
        build_tokenizer(config.chunking.tokenizer_backend, config.chunking.tokenizer_name)
    )
    resolved_chunks: list[TextChunk] = []
    vectors: list[list[float]] = []

    for chunk in chunks:
        refined_chunks, refined_vectors = _embed_chunk_recursively(
            chunk,
            config,
            token_counter=token_counter,
        )
        resolved_chunks.extend(refined_chunks)
        vectors.extend(refined_vectors)

    reindexed_chunks = _reindex_chunks(resolved_chunks, document_id)
    return reindexed_chunks, vectors


def _embed_chunk_recursively(
    chunk: TextChunk,
    config: RuntimeConfig,
    *,
    token_counter: Callable[[str], int],
) -> tuple[list[TextChunk], list[list[float]]]:
    try:
        return [chunk], [embed_chunk_text(config, chunk.text)]
    except EmbeddingContextError:
        split_chunks = split_chunk_for_context(chunk, token_counter=token_counter)
        if len(split_chunks) == 1 and split_chunks[0].text == chunk.text:
            raise
        resolved_chunks: list[TextChunk] = []
        vectors: list[list[float]] = []
        for split_chunk in split_chunks:
            nested_chunks, nested_vectors = _embed_chunk_recursively(
                split_chunk,
                config,
                token_counter=token_counter,
            )
            resolved_chunks.extend(nested_chunks)
            vectors.extend(nested_vectors)
        return resolved_chunks, vectors


def _reindex_chunks(chunks: list[TextChunk], document_id: str) -> list[TextChunk]:
    if not chunks:
        return []
    occurrences: dict[str, int] = {}
    reindexed: list[TextChunk] = []
    for index, chunk in enumerate(chunks):
        chunk_occurrence = occurrences.get(chunk.chunk_hash, 0)
        occurrences[chunk.chunk_hash] = chunk_occurrence + 1
        reindexed.append(
            TextChunk(
                chunk_id=make_chunk_id(document_id, chunk.chunk_hash, chunk_occurrence),
                document_id=document_id,
                source_rel_path=chunk.source_rel_path,
                source_type=chunk.source_type,
                citation_label=chunk.citation_label,
                chunk_index=index,
                chunk_occurrence=chunk_occurrence,
                token_count=chunk.token_count,
                text=chunk.text,
                chunk_hash=chunk.chunk_hash,
                score_hint=chunk.score_hint,
                metadata_json=chunk.metadata_json,
            )
        )
    return reindexed


def _manifest_record(
    *,
    scanned: ScannedCorpusFile,
    document_id: str | None,
    parent_source_path: str | None,
    chunk_count: int,
    timestamp: str,
    lifecycle_status: LifecycleStatus,
    retrieval_status: RetrievalStatus,
    error_message: str | None,
) -> ManifestRecord:
    return ManifestRecord(
        source_rel_path=scanned.relative_path,
        absolute_path=scanned.absolute_path.as_posix(),
        source_type=scanned.source_type,
        source_domain=scanned.source_domain,
        document_id=document_id,
        file_size_bytes=scanned.file_size_bytes,
        content_hash=scanned.content_hash,
        parent_source_path=parent_source_path,
        chunk_count=chunk_count,
        last_seen_at=timestamp,
        last_processed_at=timestamp,
        last_committed_at=timestamp if lifecycle_status == LifecycleStatus.COMPLETE else None,
        error_message=error_message,
        lifecycle_status=lifecycle_status,
        retrieval_status=retrieval_status,
    )


def _find_move_source(
    scanned: ScannedCorpusFile,
    existing_by_hash: dict[str, list[ManifestRecord]],
    scanned_paths: set[str],
) -> ManifestRecord | None:
    candidates = existing_by_hash.get(scanned.content_hash, [])
    for candidate in candidates:
        if candidate.source_rel_path == scanned.relative_path:
            continue
        if candidate.source_rel_path in scanned_paths:
            continue
        if candidate.source_type != scanned.source_type:
            continue
        return candidate
    return None


def _can_skip_unchanged_source(
    sqlite_connection: sqlite3.Connection,
    scanned: ScannedCorpusFile,
    manifest: ManifestRecord,
) -> bool:
    if scanned.source_type == "image_png":
        return True
    if manifest.retrieval_status != RetrievalStatus.SEARCHABLE or manifest.chunk_count <= 0:
        return False
    committed_chunks = load_chunk_records_for_source(sqlite_connection, manifest.absolute_path)
    return len(committed_chunks) == manifest.chunk_count


def _resolve_document_id(
    scanned: ScannedCorpusFile,
    existing_manifest: ManifestRecord | None,
    move_source: ManifestRecord | None,
    timestamp: str,
) -> str:
    if existing_manifest is not None and existing_manifest.document_id is not None:
        return existing_manifest.document_id
    if move_source is not None and move_source.document_id is not None:
        return move_source.document_id
    return blake3_hex(scanned.relative_path, scanned.content_hash, timestamp)


def _clone_source_records(
    *,
    sqlite_connection: sqlite3.Connection,
    old_manifest: ManifestRecord,
    new_scanned: ScannedCorpusFile,
    document_id: str,
) -> tuple[list[ChunkRecord], list[MentionRecord]]:
    old_chunks = load_chunk_records_for_source(sqlite_connection, old_manifest.absolute_path)
    mentions_by_chunk = load_mentions_for_source(sqlite_connection, old_manifest.absolute_path)
    chunk_id_map: dict[str, str] = {}
    cloned_chunks: list[ChunkRecord] = []
    for old_chunk in old_chunks:
        chunk_id = make_chunk_id(document_id, old_chunk.chunk_hash, old_chunk.chunk_occurrence)
        chunk_id_map[old_chunk.chunk_id] = chunk_id
        cloned_chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                source_rel_path=new_scanned.relative_path,
                source_path=new_scanned.absolute_path.as_posix(),
                source_filename=new_scanned.absolute_path.name,
                source_type=old_chunk.source_type,
                source_domain=new_scanned.source_domain,
                source_hash=new_scanned.content_hash,
                citation_label=_clone_citation_label(
                    old_chunk.citation_label,
                    old_manifest.source_rel_path,
                    new_scanned.relative_path,
                ),
                chunk_index=old_chunk.chunk_index,
                chunk_occurrence=old_chunk.chunk_occurrence,
                token_count=old_chunk.token_count,
                text=old_chunk.text,
                chunk_hash=old_chunk.chunk_hash,
                score_hint=old_chunk.score_hint,
                metadata_json=old_chunk.metadata_json,
                vector=old_chunk.vector,
                embedding_model=old_chunk.embedding_model,
                embedding_dims=old_chunk.embedding_dims,
            )
        )
    cloned_mentions: list[MentionRecord] = []
    for old_chunk_id, mentions in mentions_by_chunk.items():
        new_chunk_id = chunk_id_map.get(old_chunk_id)
        if new_chunk_id is None:
            continue
        cloned_mentions.extend(
            MentionRecord(
                chunk_id=new_chunk_id,
                entity_id=mention.entity_id,
                term_source=mention.term_source,
                surface_form=mention.surface_form,
                start_char=mention.start_char,
                end_char=mention.end_char,
            )
            for mention in mentions
        )
    return cloned_chunks, cloned_mentions


def _clone_citation_label(
    old_label: str, old_source_rel_path: str, new_source_rel_path: str
) -> str:
    page_fragment = ""
    if old_label.startswith(old_source_rel_path) and "#page=" in old_label:
        page_fragment = old_label.split("#page=", 1)[1]
    if page_fragment:
        try:
            return make_citation_label(new_source_rel_path, int(page_fragment))
        except ValueError:
            return f"{new_source_rel_path}#page={page_fragment}"
    return make_citation_label(new_source_rel_path)


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
