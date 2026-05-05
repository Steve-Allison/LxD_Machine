"""Tests for ingest-lifecycle invariants.

Covers:

- ``load_manifest_by_content_hash`` orders by ``source_rel_path`` so the
  incremental move-detection branch finds existing manifests.
- The ingest deletion branch passes ``source_rel_path`` (not the absolute
  path) to ``delete_sqlite_source``, so missing-from-corpus files are
  marked deleted in SQLite and the LanceDB rows are dropped.
- ``resolve_document_id`` is a pure function of relative path + content
  hash, so full rebuilds yield identical ``document_id`` values and
  downstream tables (claims, relations, profiles, communities) remain
  stable.
"""

from __future__ import annotations

from pathlib import Path

from lxd.ingest.pipeline.moves import resolve_document_id
from lxd.ingest.scanner import ScannedCorpusFile
from lxd.stores.models import ManifestRecord
from lxd.stores.sqlite.connection import build_store_paths, connect_sqlite, initialize_schema
from lxd.stores.sqlite.manifest import (
    delete_source,
    load_manifest_by_content_hash,
    load_manifest_index,
    upsert_manifest_record,
)


def _make_manifest(
    *,
    source_rel_path: str,
    absolute_path: str,
    content_hash: str,
    document_id: str | None = None,
) -> ManifestRecord:
    return ManifestRecord(
        source_rel_path=source_rel_path,
        absolute_path=absolute_path,
        source_type="markdown",
        source_domain="guides",
        document_id=document_id,
        file_size_bytes=123,
        content_hash=content_hash,
        parent_source_rel_path=None,
        chunk_count=0,
        last_seen_at="2026-03-27T00:00:00+00:00",
        last_processed_at="2026-03-27T00:00:00+00:00",
        last_committed_at="2026-03-27T00:00:00+00:00",
        error_message=None,
    )


def test_load_manifest_by_content_hash_uses_existing_column(tmp_path: Path) -> None:
    """Regression: query must ORDER BY ``source_rel_path`` (not ``file_rel_path``).

    The previous query referenced a column that does not exist on
    ``corpus_manifest``; this test would have raised
    ``sqlite3.OperationalError: no such column: file_rel_path`` before the fix.
    """
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        upsert_manifest_record(
            connection,
            _make_manifest(
                source_rel_path="Guides/alpha.md",
                absolute_path="/tmp/Guides/alpha.md",
                content_hash="hash-alpha",
                document_id="doc-alpha",
            ),
        )
        upsert_manifest_record(
            connection,
            _make_manifest(
                source_rel_path="Guides/beta.md",
                absolute_path="/tmp/Guides/beta.md",
                content_hash="hash-beta",
                document_id="doc-beta",
            ),
        )

        grouped = load_manifest_by_content_hash(connection)
    finally:
        connection.close()

    assert set(grouped.keys()) == {"hash-alpha", "hash-beta"}
    assert grouped["hash-alpha"][0].source_rel_path == "Guides/alpha.md"
    assert grouped["hash-beta"][0].source_rel_path == "Guides/beta.md"


def test_delete_source_removes_manifest_when_given_relative_path(tmp_path: Path) -> None:
    """Regression: deletion must use ``source_rel_path`` (not absolute path).

    Previously the pipeline's deletion branch passed ``absolute_path`` into
    ``delete_sqlite_source``; the filter compared against ``source_rel_path``
    and produced a silent no-op, so the manifest row remained and drifted
    against LanceDB.
    """
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        manifest = _make_manifest(
            source_rel_path="Guides/old.md",
            absolute_path="/tmp/Guides/old.md",
            content_hash="hash-old",
            document_id="doc-old",
        )
        upsert_manifest_record(connection, manifest)

        delete_source(connection, manifest.absolute_path)
        after_wrong_key = load_manifest_index(connection)
        still_present = "Guides/old.md" in after_wrong_key
        assert still_present, (
            "Using absolute_path should not delete a row keyed on source_rel_path."
        )

        delete_source(connection, manifest.source_rel_path)
        after_correct_key = load_manifest_index(connection)
    finally:
        connection.close()

    assert "Guides/old.md" in after_correct_key
    assert after_correct_key["Guides/old.md"].lifecycle_status == "deleted"


def testresolve_document_id_is_deterministic_across_full_rebuilds(tmp_path: Path) -> None:
    """`document_id` must be a pure function of path + content hash."""
    scanned = ScannedCorpusFile(
        absolute_path=tmp_path / "Guides" / "doc.md",
        relative_path="Guides/doc.md",
        source_type="markdown",
        file_size_bytes=42,
        content_hash="hash-doc",
        source_domain="guides",
    )

    first = resolve_document_id(
        scanned,
        existing_manifest=None,
        move_source=None,
    )
    second = resolve_document_id(
        scanned,
        existing_manifest=None,
        move_source=None,
    )

    assert first == second


def testresolve_document_id_prefers_existing_manifest_id(tmp_path: Path) -> None:
    """Existing manifest ``document_id`` wins over any fresh derivation."""
    scanned = ScannedCorpusFile(
        absolute_path=tmp_path / "Guides" / "doc.md",
        relative_path="Guides/doc.md",
        source_type="markdown",
        file_size_bytes=42,
        content_hash="hash-doc",
        source_domain="guides",
    )
    existing = _make_manifest(
        source_rel_path="Guides/doc.md",
        absolute_path=str(scanned.absolute_path),
        content_hash="hash-doc",
        document_id="stable-doc-id",
    )

    resolved = resolve_document_id(
        scanned,
        existing_manifest=existing,
        move_source=None,
    )

    assert resolved == "stable-doc-id"
