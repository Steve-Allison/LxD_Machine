from __future__ import annotations

from types import SimpleNamespace

from lxd.app.status import load_committed_status
from lxd.stores.models import OntologySnapshotRecord
from lxd.stores.sqlite import (
    build_store_paths,
    connect_sqlite,
    initialize_schema,
    replace_ontology_snapshot,
)


def test_load_committed_status_uses_live_plan_entity_count_for_legacy_snapshot(tmp_path) -> None:
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        replace_ontology_snapshot(
            connection,
            OntologySnapshotRecord(
                snapshot_id="current",
                ontology_root="/tmp/ontology",
                snapshot_hash="snapshot-hash",
                matcher_termset_hash="matcher-hash",
                matcher_term_count=2,
                source_file_count=1,
                entity_file_count=1,
                entity_count=0,
                coverage_path_count=0,
                graph_relation_count=0,
                validation_issue_count=0,
                validation_issues_json="[]",
                last_loaded_at="2026-03-10T00:00:00+00:00",
            ),
        )

        config = SimpleNamespace(
            paths=SimpleNamespace(corpus_path=tmp_path / "corpus", ontology_path=tmp_path / "ontology", data_path=tmp_path),
            chunking=SimpleNamespace(
                chunk_overlap=10,
                chunk_size=100,
                min_tokens=5,
                strategy="hybrid_docling",
                tokenizer_backend="tiktoken",
                tokenizer_name="cl100k_base",
            ),
            models=SimpleNamespace(embed="embed-model", embed_backend="ollama", embed_dims=3),
        )
        plan = SimpleNamespace(
            ontology=SimpleNamespace(
                sources=[object()],
                matcher_records=[object(), object()],
                matcher_termset_hash="live-matcher-hash",
                snapshot_hash="live-snapshot-hash",
                coverage_report=SimpleNamespace(discovered_path_count=4),
                relation_records=[object(), object(), object()],
                validation_issues=[],
                entity_definitions=[{"canonical_id": "a"}, {"canonical_id": "b"}],
            )
        )

        status_snapshot = load_committed_status(
            connection,
            config=config,
            plan_provider=lambda: plan,
        )
    finally:
        connection.close()

    assert status_snapshot is not None
    assert status_snapshot.entity_count == 2
    assert status_snapshot.summary.matcher_termset_hash == "live-matcher-hash"
    assert status_snapshot.summary.ontology_snapshot_hash == "live-snapshot-hash"


def test_load_committed_status_ignores_malformed_validation_issue_json(tmp_path) -> None:
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        replace_ontology_snapshot(
            connection,
            OntologySnapshotRecord(
                snapshot_id="current",
                ontology_root="/tmp/ontology",
                snapshot_hash="snapshot-hash",
                matcher_termset_hash="matcher-hash",
                matcher_term_count=1,
                source_file_count=1,
                entity_file_count=1,
                entity_count=1,
                coverage_path_count=1,
                graph_relation_count=1,
                validation_issue_count=2,
                validation_issues_json="{not-json",
                last_loaded_at="2026-03-10T00:00:00+00:00",
            ),
        )
        config = SimpleNamespace(
            paths=SimpleNamespace(corpus_path=tmp_path / "corpus", ontology_path=tmp_path / "ontology", data_path=tmp_path),
            chunking=SimpleNamespace(
                chunk_overlap=10,
                chunk_size=100,
                min_tokens=5,
                strategy="hybrid_docling",
                tokenizer_backend="tiktoken",
                tokenizer_name="cl100k_base",
            ),
            models=SimpleNamespace(embed="embed-model", embed_backend="ollama", embed_dims=3),
        )
        plan = SimpleNamespace(
            ontology=SimpleNamespace(
                sources=[],
                matcher_records=[],
                matcher_termset_hash="",
                snapshot_hash="",
                coverage_report=SimpleNamespace(discovered_path_count=0),
                relation_records=[],
                validation_issues=[],
                entity_definitions=[],
            )
        )

        status_snapshot = load_committed_status(
            connection,
            config=config,
            plan_provider=lambda: plan,
        )
    finally:
        connection.close()

    assert status_snapshot is not None
    assert status_snapshot.summary.ontology_validation_issue_samples == []
