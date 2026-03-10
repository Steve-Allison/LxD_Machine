from __future__ import annotations

from lxd.ingest.chunking import chunk_document
from lxd.ingest.markdown import extract_text_blocks, load_markdown_document


def test_extract_text_blocks_collapses_padding_whitespace() -> None:
    text = """
    | Benefits            | Constraints                    |
    | ------------------- | ------------------------------ |
    | Faster authoring    | Requires human review          |


    A paragraph    with      irregular spacing.
    """.strip()

    blocks = extract_text_blocks(text)

    assert blocks == [
        "| Benefits | Constraints |\n| ------------------- | ------------------------------ |\n| Faster authoring | Requires human review |",
        "A paragraph with irregular spacing.",
    ]


def test_load_markdown_document_uses_docling_and_native_chunking(tmp_path) -> None:
    source_path = tmp_path / "example.md"
    source_path.write_text(
        "# Title\n\nParagraph one.\n\n## Section\n\n- Item A\n- Item B\n",
        encoding="utf-8",
    )

    document = load_markdown_document(source_path, "Guides/example.md")

    assert document.docling_document is not None
    chunks = chunk_document(
        document,
        document_id="doc-1",
        chunk_size=50,
        chunk_overlap=10,
        min_tokens=1,
        tokenizer_backend="tiktoken",
        tokenizer_name="cl100k_base",
        strategy="hybrid_docling",
    )

    assert chunks
    assert chunks[0].metadata_json
    assert "Title" in chunks[0].text
