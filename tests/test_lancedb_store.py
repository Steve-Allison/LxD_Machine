from __future__ import annotations

from lxd.stores.lancedb import (
    connect_lancedb,
    replace_source_chunks,
    reset_chunk_table,
    search_chunks,
)
from lxd.stores.models import ChunkRecord


def test_lancedb_search_and_domain_filter(tmp_path) -> None:
    database = connect_lancedb(tmp_path / "lancedb")
    table = reset_chunk_table(database, vector_size=3)
    replace_source_chunks(
        table,
        "Guides/example.md",
        [
            ChunkRecord(
                chunk_id="chunk-guides",
                document_id="doc-guides",
                source_rel_path="Guides/example.md",
                source_path="/tmp/Guides/example.md",
                source_filename="example.md",
                source_type="markdown",
                source_domain="guides",
                source_hash="hash-guides-source",
                citation_label="Guides/example.md",
                chunk_index=0,
                chunk_occurrence=0,
                token_count=2,
                text="Guide text",
                chunk_hash="hash-guides",
                score_hint="Guide text",
                metadata_json="{}",
                vector=[1.0, 0.0, 0.0],
                embedding_model="test-embed",
                embedding_dims=3,
            ),
            ChunkRecord(
                chunk_id="chunk-theories",
                document_id="doc-theories",
                source_rel_path="Theories/example.md",
                source_path="/tmp/Theories/example.md",
                source_filename="example.md",
                source_type="markdown",
                source_domain="theories",
                source_hash="hash-theories-source",
                citation_label="Theories/example.md",
                chunk_index=0,
                chunk_occurrence=0,
                token_count=2,
                text="Theory text",
                chunk_hash="hash-theories",
                score_hint="Theory text",
                metadata_json="{}",
                vector=[0.0, 1.0, 0.0],
                embedding_model="test-embed",
                embedding_dims=3,
            ),
        ],
    )

    guides_hits = search_chunks(table, query_vector=[1.0, 0.0, 0.0], domain="guides", limit=5)
    theory_hits = search_chunks(table, query_vector=[1.0, 0.0, 0.0], domain="theories", limit=5)

    assert [item.chunk_id for item in guides_hits] == ["chunk-guides"]
    assert [item.chunk_id for item in theory_hits] == ["chunk-theories"]


def test_reset_chunk_table_ignores_missing_table() -> None:
    class FakeDatabase:
        def __init__(self) -> None:
            self.drop_calls: list[str] = []
            self.create_calls: list[tuple[str, str]] = []

        def drop_table(self, table_name: str) -> None:
            self.drop_calls.append(table_name)
            raise ValueError(f"Table '{table_name}' was not found")

        def create_table(self, table_name: str, *, schema, mode: str):
            self.create_calls.append((table_name, mode))
            return {"name": table_name, "mode": mode, "schema": schema}

    fake_database = FakeDatabase()

    table = reset_chunk_table(fake_database, vector_size=3)

    assert fake_database.drop_calls == ["chunk_vectors"]
    assert fake_database.create_calls == [("chunk_vectors", "create")]
    assert table["name"] == "chunk_vectors"
