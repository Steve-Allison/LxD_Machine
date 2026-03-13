from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from lxd.domain.citations import make_citation_label


@dataclass(frozen=True)
class ExtractedDocument:
    source_rel_path: str
    source_type: str
    citation_label: str
    text_blocks: list[str]
    docling_document: Any | None = None


def load_markdown_document(
    path: Path,
    source_rel_path: str,
    *,
    source_type: str = "markdown",
) -> ExtractedDocument:
    text = path.read_text(encoding="utf-8")
    converter = DocumentConverter()
    document = converter.convert_string(
        content=text, format=InputFormat.MD, name=source_rel_path
    ).document
    return ExtractedDocument(
        source_rel_path=source_rel_path,
        source_type=source_type,
        citation_label=make_citation_label(source_rel_path),
        text_blocks=[],
        docling_document=document,
    )


def extract_text_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalize_line(line) for line in normalized.split("\n")]
    blocks: list[str] = []
    current_block: list[str] = []

    for line in lines:
        if not line:
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            continue
        current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block))
    return blocks


def _normalize_line(line: str) -> str:
    collapsed = re.sub(r"[ \t]+", " ", line).strip()
    return collapsed
