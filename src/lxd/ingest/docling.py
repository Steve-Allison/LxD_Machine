"""Load and normalize rich documents through the Docling pipeline."""

from __future__ import annotations

from pathlib import Path

from docling_core.types.doc.document import DoclingDocument

from lxd.domain.citations import make_citation_label
from lxd.ingest.markdown import ExtractedDocument


def load_docling_document(path: Path, source_rel_path: str) -> ExtractedDocument:
    """Load a Docling JSON document as an extracted document.

    Args:
        path: Path to the source file or storage location.
        source_rel_path: Corpus-relative source path.

    Returns:
        Extracted document wrapper for the Docling source.
    """
    document = DoclingDocument.load_from_json(path)
    return ExtractedDocument(
        source_rel_path=source_rel_path,
        source_type="docling_json",
        citation_label=make_citation_label(source_rel_path),
        text_blocks=[],
        docling_document=document,
    )
