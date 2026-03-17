from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import tiktoken
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from lxd.domain.citations import make_citation_label
from lxd.domain.ids import blake3_hex, make_chunk_id
from lxd.ingest.markdown import ExtractedDocument


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    source_rel_path: str
    source_type: str
    citation_label: str
    chunk_index: int
    chunk_occurrence: int
    token_count: int
    text: str
    chunk_hash: str
    score_hint: str
    metadata_json: str


def chunk_document(
    document: ExtractedDocument,
    *,
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
    min_tokens: int,
    tokenizer_backend: str,
    tokenizer_name: str,
    strategy: str = "hybrid_docling",
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if document.docling_document is not None:
        return _chunk_docling_document(
            document=document,
            document_id=document_id,
            chunk_size=chunk_size,
            min_tokens=min_tokens,
            tokenizer_backend=tokenizer_backend,
            tokenizer_name=tokenizer_name,
            strategy=strategy,
        )

    tokenizer = build_tokenizer(tokenizer_backend, tokenizer_name)
    full_text = "\n\n".join(
        block.strip() for block in document.text_blocks if block.strip()
    ).strip()
    if not full_text:
        return []

    tokens = tokenizer.encode(full_text)
    if not tokens:
        return []

    chunks: list[TextChunk] = []
    stride = chunk_size - chunk_overlap
    chunk_index = 0
    occurrences: dict[str, int] = {}

    for start in range(0, len(tokens), stride):
        end = min(start + chunk_size, len(tokens))
        window_tokens = tokens[start:end]
        if len(window_tokens) < min_tokens and chunks:
            break
        text = tokenizer.decode(window_tokens).strip()
        if not text:
            if end == len(tokens):
                break
            continue
        chunk_hash = blake3_hex(text)
        chunk_occurrence = occurrences.get(chunk_hash, 0)
        occurrences[chunk_hash] = chunk_occurrence + 1
        chunks.append(
            _build_chunk(
                document=document,
                document_id=document_id,
                chunk_index=chunk_index,
                chunk_occurrence=chunk_occurrence,
                token_count=len(window_tokens),
                text=text,
                metadata_json="{}",
            )
        )
        chunk_index += 1
        if end == len(tokens):
            break
    return chunks


def split_chunk_for_context(
    chunk: TextChunk,
    *,
    token_counter: Callable[[str], int] | None = None,
) -> list[TextChunk]:
    normalized_text = chunk.text.strip()
    if not normalized_text:
        return [chunk]

    left_text, right_text = _split_text_at_best_boundary(normalized_text)
    if not left_text or not right_text:
        return [chunk]

    split_texts = [candidate for candidate in [left_text, right_text] if candidate]
    return [
        _build_chunk_from_fields(
            chunk=chunk,
            document_id=chunk.document_id,
            chunk_index=index,
            chunk_occurrence=index,
            token_count=(token_counter or _token_count_for_text)(text),
            text=text,
            metadata_json=chunk.metadata_json,
        )
        for index, text in enumerate(split_texts)
    ]


def _chunk_docling_document(
    *,
    document: ExtractedDocument,
    document_id: str,
    chunk_size: int,
    min_tokens: int,
    tokenizer_backend: str,
    tokenizer_name: str,
    strategy: str,
) -> list[TextChunk]:
    if document.docling_document is None:
        return []
    if strategy == "hybrid_docling":
        chunker = HybridChunker(
            tokenizer=_build_docling_tokenizer(tokenizer_backend, tokenizer_name, chunk_size),
            merge_peers=True,
        )
    elif strategy == "hierarchical_docling":
        chunker = HierarchicalChunker()
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    configured_tokenizer = build_tokenizer(tokenizer_backend, tokenizer_name)
    chunks: list[TextChunk] = []
    occurrences: dict[str, int] = {}
    for native_chunk in chunker.chunk(document.docling_document):
        raw_text = chunker.contextualize(native_chunk)
        normalized_text = _normalize_chunk_text(raw_text)
        if not normalized_text:
            continue
        token_count = len(configured_tokenizer.encode(normalized_text))
        if token_count < min_tokens and chunks:
            continue
        chunk_hash = blake3_hex(normalized_text)
        chunk_occurrence = occurrences.get(chunk_hash, 0)
        occurrences[chunk_hash] = chunk_occurrence + 1
        metadata_json = json.dumps(
            native_chunk.meta.export_json_dict(),
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        )
        chunks.append(
            _build_chunk(
                document=ExtractedDocument(
                    source_rel_path=document.source_rel_path,
                    source_type=document.source_type,
                    citation_label=_citation_label_for_chunk(
                        document.source_rel_path, metadata_json
                    ),
                    text_blocks=[],
                    docling_document=document.docling_document,
                ),
                document_id=document_id,
                chunk_index=len(chunks),
                chunk_occurrence=chunk_occurrence,
                token_count=token_count,
                text=normalized_text,
                metadata_json=metadata_json,
            )
        )
    return chunks


def _build_chunk(
    *,
    document: ExtractedDocument,
    document_id: str,
    chunk_index: int,
    chunk_occurrence: int,
    token_count: int,
    text: str,
    metadata_json: str,
) -> TextChunk:
    return _build_chunk_from_fields(
        chunk=TextChunk(
            chunk_id="",
            document_id=document_id,
            source_rel_path=document.source_rel_path,
            source_type=document.source_type,
            citation_label=document.citation_label,
            chunk_index=chunk_index,
            chunk_occurrence=chunk_occurrence,
            token_count=token_count,
            text=text,
            chunk_hash="",
            score_hint="",
            metadata_json=metadata_json,
        ),
        document_id=document_id,
        chunk_index=chunk_index,
        chunk_occurrence=chunk_occurrence,
        token_count=token_count,
        text=text,
        metadata_json=metadata_json,
    )


def _build_chunk_from_fields(
    *,
    chunk: TextChunk,
    document_id: str,
    chunk_index: int,
    chunk_occurrence: int,
    token_count: int,
    text: str,
    metadata_json: str,
) -> TextChunk:
    chunk_hash = blake3_hex(text)
    return TextChunk(
        chunk_id=make_chunk_id(document_id, chunk_hash, chunk_occurrence),
        document_id=document_id,
        source_rel_path=chunk.source_rel_path,
        source_type=chunk.source_type,
        citation_label=chunk.citation_label,
        chunk_index=chunk_index,
        chunk_occurrence=chunk_occurrence,
        token_count=token_count,
        text=text,
        chunk_hash=chunk_hash,
        score_hint=text[:160],
        metadata_json=metadata_json,
    )


@dataclass(frozen=True)
class _Tokenizer:
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]


def build_tokenizer(tokenizer_backend: str, tokenizer_name: str) -> _Tokenizer:
    if tokenizer_backend != "tiktoken":
        raise ValueError(f"Unsupported tokenizer backend: {tokenizer_backend}")
    encoding = tiktoken.get_encoding(tokenizer_name)
    return _Tokenizer(encode=encoding.encode, decode=encoding.decode)


def _build_docling_tokenizer(tokenizer_backend: str, tokenizer_name: str, max_tokens: int) -> Any:
    if tokenizer_backend != "tiktoken":
        raise ValueError(f"Unsupported docling tokenizer backend: {tokenizer_backend}")
    return OpenAITokenizer(tokenizer=tiktoken.get_encoding(tokenizer_name), max_tokens=max_tokens)


def _normalize_chunk_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _citation_label_for_chunk(source_rel_path: str, metadata_json: str) -> str:
    try:
        metadata = json.loads(metadata_json)
    except ValueError:
        return source_rel_path
    page_no = _find_page_no(metadata)
    if page_no is None:
        return source_rel_path
    return make_citation_label(source_rel_path, page_no)


def _find_page_no(value: Any) -> int | None:
    if isinstance(value, dict):
        if isinstance(value.get("page_no"), int):
            return int(value["page_no"])
        for child in value.values():
            page_no = _find_page_no(child)
            if page_no is not None:
                return page_no
        return None
    if isinstance(value, list):
        for child in value:
            page_no = _find_page_no(child)
            if page_no is not None:
                return page_no
    return None


def _token_count_for_text(text: str) -> int:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return 0
    return len(normalized.split(" "))


def token_count_with_tokenizer(
    tokenizer: _Tokenizer,
) -> Callable[[str], int]:
    def count(text: str) -> int:
        return len(tokenizer.encode(text))

    return count


def _split_text_at_best_boundary(text: str) -> tuple[str, str]:
    midpoint = len(text) // 2
    for pattern in (
        _PARAGRAPH_BOUNDARY,
        _LINE_BOUNDARY,
        _SENTENCE_BOUNDARY,
        _CLAUSE_BOUNDARY,
        _WORD_BOUNDARY,
    ):
        split_index = _find_boundary_near_midpoint(text, pattern, midpoint)
        if split_index is None:
            continue
        left = text[:split_index].strip()
        right = text[split_index:].strip()
        if left and right:
            return left, right
    fallback = midpoint
    left = text[:fallback].strip()
    right = text[fallback:].strip()
    return left, right


def _find_boundary_near_midpoint(text: str, pattern: re.Pattern[str], midpoint: int) -> int | None:
    matches = [match for match in pattern.finditer(text)]
    if not matches:
        return None
    best_match = min(matches, key=lambda match: abs(match.end() - midpoint))
    return best_match.end()


_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")
_LINE_BOUNDARY = re.compile(r"\n")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[;:])\s+|,\s+")
_WORD_BOUNDARY = re.compile(r"\s+")
