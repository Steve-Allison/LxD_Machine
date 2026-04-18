"""Regression tests for Wave 0 P0 correctness fixes.

Covers three bugs introduced into the ingest/store layer:

1. ``load_manifest_by_content_hash`` previously ordered rows by a non-existent
   column (``file_rel_path``) and raised ``sqlite3.OperationalError`` the first
   time an incremental ingest exercised the move-detection branch.
2. The ingest pipeline's deletion branch previously passed the absolute path
   (``missing_manifest.absolute_path``) to ``delete_sqlite_source``, which
   filters by ``source_rel_path``; the call silently no-opped, leaving
   orphan manifest/chunk rows and causing drift against LanceDB.
3. ``_resolve_document_id`` previously mixed the ingest wall-clock timestamp
   into the BLAKE3 hash, so two full rebuilds of the same corpus produced
   different ``document_id`` values; downstream tables keyed on
   ``document_id`` (claims, relations, profiles, communities) could not be
   safely rebuilt.

Each test exercises the fix directly against a temp SQLite instance or a
pure-Python call, to catch regressions without the full ingest stack.
"""

from __future__ import annotations

from pathlib import Path

from lxd.ingest.pipeline import _resolve_document_id
from lxd.ingest.scanner import ScannedCorpusFile
from lxd.stores.models import ManifestRecord
from lxd.stores.sqlite import (
    build_store_paths,
    connect_sqlite,
    delete_source,
    initialize_schema,
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


def test_resolve_document_id_is_deterministic_across_full_rebuilds(tmp_path: Path) -> None:
    """Regression: ``document_id`` must be a pure function of path + content hash.

    The pre-fix implementation folded the ingest wall-clock timestamp into the
    BLAKE3 hash; two full rebuilds of the same corpus produced different IDs
    and broke every downstream table keyed on ``document_id``.
    """
    scanned = ScannedCorpusFile(
        absolute_path=tmp_path / "Guides" / "doc.md",
        relative_path="Guides/doc.md",
        source_type="markdown",
        file_size_bytes=42,
        content_hash="hash-doc",
        source_domain="guides",
    )

    first = _resolve_document_id(
        scanned,
        existing_manifest=None,
        move_source=None,
        timestamp="2026-03-27T00:00:00+00:00",
    )
    second = _resolve_document_id(
        scanned,
        existing_manifest=None,
        move_source=None,
        timestamp="2027-01-01T12:34:56+00:00",
    )

    assert first == second, "document_id must not depend on wall-clock timestamp."


def test_resolve_document_id_prefers_existing_manifest_id(tmp_path: Path) -> None:
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

    resolved = _resolve_document_id(
        scanned,
        existing_manifest=existing,
        move_source=None,
        timestamp="2026-03-27T00:00:00+00:00",
    )

    assert resolved == "stable-doc-id"
