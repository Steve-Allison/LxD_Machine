from __future__ import annotations

from lxd.stores.models import ChunkRecord, ManifestRecord, MentionRecord, OntologySnapshotRecord
from lxd.stores.sqlite import (
    begin_ingest_run,
    build_store_paths,
    connect_sqlite,
    finish_ingest_run,
    initialize_schema,
    load_chunk_records_for_source,
    load_mentions_for_source,
    load_ontology_snapshot,
    replace_ontology_snapshot,
    replace_source_chunks,
    summarize_store,
    update_ingest_run_progress,
    upsert_manifest_record,
)


def test_sqlite_store_round_trip(tmp_path) -> None:
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        upsert_manifest_record(
            connection,
            ManifestRecord(
                source_rel_path="Guides/example.md",
                absolute_path="/tmp/example.md",
                source_type="markdown",
                source_domain="guides",
                document_id="doc-guides",
                file_size_bytes=123,
                content_hash="abc123",
                parent_source_path=None,
                chunk_count=1,
                last_seen_at="2026-03-10T00:00:00+00:00",
                last_processed_at="2026-03-10T00:00:00+00:00",
                last_committed_at="2026-03-10T00:00:00+00:00",
                error_message=None,
            ),
        )
        replace_source_chunks(
            connection,
            absolute_source_path="/tmp/example.md",
            chunk_records=[
                ChunkRecord(
                    chunk_id="chunk-1",
                    document_id="doc-guides",
                    source_rel_path="Guides/example.md",
                    source_path="/tmp/example.md",
                    source_filename="example.md",
                    source_type="markdown",
                    source_domain="guides",
                    source_hash="abc123",
                    citation_label="Guides/example.md",
                    chunk_index=0,
                    chunk_occurrence=0,
                    token_count=2,
                    text="Example text",
                    chunk_hash="hash-1",
                    score_hint="Example text",
                    metadata_json="{}",
                    vector=[0.1, 0.2, 0.3],
                    embedding_model="test-embed",
                    embedding_dims=3,
                )
            ],
            mention_records=[
                MentionRecord(
                    chunk_id="chunk-1",
                    entity_id="mayer_principles",
                    term_source="alias",
                    surface_form="mayer principles",
                    start_char=0,
                    end_char=16,
                )
            ],
        )

        loaded_chunks = load_chunk_records_for_source(connection, "/tmp/example.md")
        loaded_mentions = load_mentions_for_source(connection, "/tmp/example.md")
        summary = summarize_store(
            connection,
            ontology_file_count=10,
            matcher_term_count=25,
            matcher_termset_hash="matcher-hash",
            ontology_snapshot_hash="ontology-hash",
        )
    finally:
        connection.close()

    assert len(loaded_chunks) == 1
    assert loaded_chunks[0].document_id == "doc-guides"
    assert loaded_chunks[0].vector == [0.1, 0.2, 0.3]
    assert list(loaded_mentions) == ["chunk-1"]
    assert len(loaded_mentions["chunk-1"]) == 1
    assert loaded_mentions["chunk-1"][0].term_source == "alias"
    assert summary.corpus_file_count == 1
    assert summary.text_file_count == 1
    assert summary.asset_file_count == 0
    assert summary.chunk_count == 1
    assert summary.mention_count == 1
    assert summary.matcher_termset_hash == "matcher-hash"


def test_ingest_run_progress_round_trip(tmp_path) -> None:
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        begin_ingest_run(
            connection,
            run_id="ingest-1",
            started_at="2026-03-10T00:00:00+00:00",
            mode="incremental",
            files_total=10,
        )
        update_ingest_run_progress(
            connection,
            run_id="ingest-1",
            files_completed=4,
            searchable_files_rebuilt=2,
            asset_files_processed=1,
            unchanged_files_skipped=1,
            failed_files=0,
            chunks_written=17,
            notes=["warning-a"],
        )
        finish_ingest_run(
            connection,
            run_id="ingest-1",
            finished_at="2026-03-10T00:05:00+00:00",
            status="complete_with_warnings",
            files_completed=10,
            searchable_files_rebuilt=3,
            asset_files_processed=6,
            unchanged_files_skipped=1,
            failed_files=1,
            chunks_written=23,
            notes=["warning-a", "warning-b"],
        )
        row = connection.execute(
            """
            SELECT
                status,
                files_total,
                files_completed,
                searchable_files_rebuilt,
                asset_files_processed,
                unchanged_files_skipped,
                failed_files,
                chunks_written,
                notes,
                finished_at
            FROM ingest_runs
            WHERE run_id = 'ingest-1'
            """
        ).fetchone()
    finally:
        connection.close()

    assert row is not None
    assert row["status"] == "complete_with_warnings"
    assert row["files_total"] == 10
    assert row["files_completed"] == 10
    assert row["searchable_files_rebuilt"] == 3
    assert row["asset_files_processed"] == 6
    assert row["unchanged_files_skipped"] == 1
    assert row["failed_files"] == 1
    assert row["chunks_written"] == 23
    assert row["notes"] == '["warning-a","warning-b"]'
    assert row["finished_at"] == "2026-03-10T00:05:00+00:00"


def test_ontology_snapshot_round_trip_includes_validation_fields(tmp_path) -> None:
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        replace_ontology_snapshot(
            connection,
            OntologySnapshotRecord(
                snapshot_id="current",
                ontology_root="/tmp/ontology",
                snapshot_hash="ontology-hash",
                matcher_termset_hash="matcher-hash",
                matcher_term_count=25,
                source_file_count=10,
                entity_file_count=4,
                entity_count=12,
                coverage_path_count=123,
                graph_relation_count=456,
                validation_issue_count=2,
                validation_issues_json='["issue-a","issue-b"]',
                last_loaded_at="2026-03-10T00:00:00+00:00",
            ),
        )
        loaded = load_ontology_snapshot(connection)
    finally:
        connection.close()

    assert loaded is not None
    assert loaded.coverage_path_count == 123
    assert loaded.graph_relation_count == 456
    assert loaded.validation_issue_count == 2
    assert loaded.validation_issues_json == '["issue-a","issue-b"]'
