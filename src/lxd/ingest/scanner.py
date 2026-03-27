"""Scan filesystem sources and derive metadata for ingest candidates."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScannedCorpusFile:
    """Metadata for a discovered corpus file candidate."""
    absolute_path: Path
    relative_path: str
    source_type: str
    file_size_bytes: int
    content_hash: str
    source_domain: str


def scan_corpus(
    corpus_root: Path,
    text_extensions: list[str],
    asset_extensions: list[str],
    ignore_names: list[str],
) -> list[ScannedCorpusFile]:
    """Scan corpus files and collect ingestion metadata.

    Args:
        corpus_root: Root directory of the corpus to scan.
        text_extensions: File suffixes treated as text sources.
        asset_extensions: File suffixes treated as asset sources.
        ignore_names: Filenames to skip while scanning.

    Returns:
        Scanned corpus files eligible for ingestion.
    """
    scanned: list[ScannedCorpusFile] = []
    for path in _iter_files(corpus_root, ignore_names):
        source_type = classify_source_type(path, text_extensions, asset_extensions)
        if source_type is None:
            continue
        rel_path = str(path.relative_to(corpus_root))
        scanned.append(
            ScannedCorpusFile(
                absolute_path=path,
                relative_path=rel_path,
                source_type=source_type,
                file_size_bytes=path.stat().st_size,
                content_hash=_hash_file(path),
                source_domain=derive_source_domain(rel_path),
            )
        )
    return scanned


def classify_source_type(
    path: Path, text_extensions: list[str], asset_extensions: list[str]
) -> str | None:
    """Classify a file as text, docling, asset, or ignored.

    Args:
        path: Path to the source file or storage location.
        text_extensions: File suffixes treated as text sources.
        asset_extensions: File suffixes treated as asset sources.

    Returns:
        Detected source type, or `None` when ignored.
    """
    path_text = path.name
    if any(path_text.endswith(ext) for ext in text_extensions):
        if path_text.endswith(".docling.json"):
            return "docling_json"
        if path_text.endswith(".docling.md"):
            return "docling_md"
        return "markdown"
    if any(path_text.endswith(ext) for ext in asset_extensions):
        return "image_png"
    return None


def derive_source_domain(relative_path: str) -> str:
    """Derive the normalized source domain from a relative path.

    Args:
        relative_path: Corpus-relative file path.

    Returns:
        Normalized domain key derived from the relative path.
    """
    first_segment = Path(relative_path).parts[0] if Path(relative_path).parts else "root"
    return first_segment.lower().replace("-", "_").replace(" ", "_")


def _iter_files(root: Path, ignore_names: list[str]) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in ignore_names:
            yield path


def _hash_file(path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=32)
    with path.open("rb") as handle:
        while chunk := handle.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
