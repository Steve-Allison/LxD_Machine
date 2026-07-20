"""Unit tests for LanceDB source-path snapshot / restore compensate helpers."""

from pathlib import Path

import pytest

from lxd.stores.lancedb import (
    connect_lancedb,
    load_source_chunk_rows,
    open_chunk_table,
    replace_source_chunks,
    restore_source_chunk_rows,
)
from lxd.stores.models import ChunkRecord

pytestmark = [pytest.mark.unit]


def _chunk(chunk_id: str, source: str, text: str, vector: list[float]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_rel_path=source,
        source_filename="page.md",
        source_type="markdown",
        source_domain="test",
        source_hash="hash",
        citation_label="page.md",
        chunk_index=0,
        chunk_occurrence=0,
        token_count=3,
        text=text,
        chunk_hash=f"ch-{chunk_id}",
        score_hint="",
        metadata_json="{}",
        vector=vector,
        embedding_model="test",
        embedding_dims=len(vector),
    )


def test_restore_source_chunk_rows_reinstates_prior_vectors(tmp_path: Path) -> None:
    db = connect_lancedb(tmp_path / "lance")
    table = open_chunk_table(db, vector_size=3)
    source = "Guides/a.md"
    original = [_chunk("c1", source, "original text", [1.0, 0.0, 0.0])]
    replace_source_chunks(table, source, original)

    snapshot = load_source_chunk_rows(table, source)
    assert len(snapshot) == 1
    assert snapshot[0]["chunk_id"] == "c1"

    replacement = [_chunk("c2", source, "new text", [0.0, 1.0, 0.0])]
    replace_source_chunks(table, source, replacement)
    after_replace = load_source_chunk_rows(table, source)
    assert len(after_replace) == 1
    assert after_replace[0]["chunk_id"] == "c2"

    restore_source_chunk_rows(table, source, snapshot)
    restored = load_source_chunk_rows(table, source)
    assert len(restored) == 1
    assert restored[0]["chunk_id"] == "c1"
    assert restored[0]["text"] == "original text"


def test_restore_empty_snapshot_clears_source(tmp_path: Path) -> None:
    db = connect_lancedb(tmp_path / "lance")
    table = open_chunk_table(db, vector_size=3)
    source = "Guides/b.md"
    replace_source_chunks(table, source, [_chunk("c1", source, "text", [1.0, 0.0, 0.0])])
    restore_source_chunk_rows(table, source, [])
    assert load_source_chunk_rows(table, source) == []
