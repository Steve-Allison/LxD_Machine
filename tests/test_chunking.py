from __future__ import annotations

import tiktoken

from lxd.ingest.chunking import TextChunk, chunk_document, split_chunk_for_context
from lxd.ingest.markdown import ExtractedDocument


def test_chunk_document_splits_oversized_single_block_by_tokens() -> None:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    oversized_text = " ".join(f"token{i}" for i in range(2500))
    document = ExtractedDocument(
        source_rel_path="Guides/oversized.md",
        source_type="markdown",
        citation_label="Guides/oversized.md",
        text_blocks=[oversized_text],
    )

    chunks = chunk_document(
        document,
        document_id="doc-1",
        chunk_size=800,
        chunk_overlap=150,
        min_tokens=20,
        tokenizer_backend="tiktoken",
        tokenizer_name="cl100k_base",
    )

    assert len(chunks) >= 4
    assert all(chunk.text for chunk in chunks)
    token_lengths = [len(tokenizer.encode(chunk.text)) for chunk in chunks]
    assert max(token_lengths) <= 800
    assert token_lengths[-1] >= 20


def test_split_chunk_for_context_uses_text_boundaries() -> None:
    chunk = TextChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        source_rel_path="Guides/example.md",
        source_type="markdown",
        citation_label="Guides/example.md",
        chunk_index=0,
        chunk_occurrence=0,
        token_count=12,
        text="First paragraph.\n\nSecond paragraph has more content.\n\nThird paragraph finishes the example.",
        chunk_hash="hash-1",
        score_hint="First paragraph.",
        metadata_json="{}",
    )

    split_chunks = split_chunk_for_context(chunk)

    assert len(split_chunks) == 2
    assert split_chunks[0].text == "First paragraph.\n\nSecond paragraph has more content."
    assert split_chunks[1].text == "Third paragraph finishes the example."


def test_split_chunk_for_context_uses_supplied_token_counter() -> None:
    chunk = TextChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        source_rel_path="Guides/example.md",
        source_type="markdown",
        citation_label="Guides/example.md",
        chunk_index=0,
        chunk_occurrence=0,
        token_count=12,
        text="Alpha beta.\n\nGamma delta epsilon.",
        chunk_hash="hash-1",
        score_hint="Alpha beta.",
        metadata_json="{}",
    )

    split_chunks = split_chunk_for_context(chunk, token_counter=lambda text: len(text))

    assert [item.token_count for item in split_chunks] == [11, 20]
