"""Move-detection, unchanged-source skip, document-id resolution, and chunk cloning."""

from __future__ import annotations

import sqlite3
from typing import Any

from lxd.domain.citations import make_citation_label
from lxd.domain.ids import blake3_hex, make_chunk_id
from lxd.domain.status import RetrievalStatus
from lxd.ingest.scanner import ScannedCorpusFile
from lxd.stores.lancedb import load_vectors_by_chunk_ids
from lxd.stores.models import ChunkRecord, ManifestRecord, MentionRecord
from lxd.stores.sqlite.chunks import (
    load_chunk_records_for_source,
    load_mentions_for_source,
)


def find_move_source(
    scanned: ScannedCorpusFile,
    existing_by_hash: dict[str, list[ManifestRecord]],
    scanned_paths: set[str],
) -> ManifestRecord | None:
    candidates = existing_by_hash.get(scanned.content_hash, [])
    for candidate in candidates:
        if candidate.source_rel_path == scanned.relative_path:
            continue
        if candidate.source_rel_path in scanned_paths:
            continue
        if candidate.source_type != scanned.source_type:
            continue
        return candidate
    return None


def can_skip_unchanged_source(
    sqlite_connection: sqlite3.Connection,
    scanned: ScannedCorpusFile,
    manifest: ManifestRecord,
) -> bool:
    if scanned.source_type == "image_png":
        return True
    if manifest.retrieval_status != RetrievalStatus.SEARCHABLE or manifest.chunk_count <= 0:
        return False
    committed_chunks = load_chunk_records_for_source(sqlite_connection, manifest.source_rel_path)
    return len(committed_chunks) == manifest.chunk_count


def resolve_document_id(
    scanned: ScannedCorpusFile,
    existing_manifest: ManifestRecord | None,
    move_source: ManifestRecord | None,
) -> str:
    """Return a deterministic document_id for a scanned source.

    `document_id` is a pure function of content identity (relative path +
    content hash) so that repeated full rebuilds yield identical identifiers
    and downstream tables keyed on `document_id` (claims, relations,
    profiles, communities) remain stable across runs.
    """
    if existing_manifest is not None and existing_manifest.document_id is not None:
        return existing_manifest.document_id
    if move_source is not None and move_source.document_id is not None:
        return move_source.document_id
    return blake3_hex(scanned.relative_path, scanned.content_hash)


def clone_source_records(
    *,
    sqlite_connection: sqlite3.Connection,
    vector_table: Any,
    old_manifest: ManifestRecord,
    new_scanned: ScannedCorpusFile,
    document_id: str,
) -> tuple[list[ChunkRecord], list[MentionRecord]]:
    """Clone an existing source's chunks/mentions under new identity.

    Vectors are hydrated from LanceDB (the canonical vector store as of
    schema v2) keyed on the old chunk IDs; SQLite no longer carries
    ``vector_json``. Chunks whose vectors are missing from LanceDB inherit an
    empty vector and must be re-embedded by the caller.
    """
    old_chunks = load_chunk_records_for_source(sqlite_connection, old_manifest.source_rel_path)
    mentions_by_chunk = load_mentions_for_source(sqlite_connection, old_manifest.source_rel_path)
    vectors_by_old_id = load_vectors_by_chunk_ids(
        vector_table, [chunk.chunk_id for chunk in old_chunks]
    )
    chunk_id_map: dict[str, str] = {}
    cloned_chunks: list[ChunkRecord] = []
    for old_chunk in old_chunks:
        chunk_id = make_chunk_id(document_id, old_chunk.chunk_hash, old_chunk.chunk_occurrence)
        chunk_id_map[old_chunk.chunk_id] = chunk_id
        cloned_chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                source_rel_path=new_scanned.relative_path,
                source_filename=new_scanned.absolute_path.name,
                source_type=old_chunk.source_type,
                source_domain=new_scanned.source_domain,
                source_hash=new_scanned.content_hash,
                citation_label=_clone_citation_label(
                    old_chunk.citation_label,
                    old_manifest.source_rel_path,
                    new_scanned.relative_path,
                ),
                chunk_index=old_chunk.chunk_index,
                chunk_occurrence=old_chunk.chunk_occurrence,
                token_count=old_chunk.token_count,
                text=old_chunk.text,
                chunk_hash=old_chunk.chunk_hash,
                score_hint=old_chunk.score_hint,
                metadata_json=old_chunk.metadata_json,
                vector=vectors_by_old_id.get(old_chunk.chunk_id, []),
                embedding_model=old_chunk.embedding_model,
                embedding_dims=old_chunk.embedding_dims,
                cited_sources=old_chunk.cited_sources,
                wiki_links=old_chunk.wiki_links,
            )
        )
    cloned_mentions: list[MentionRecord] = []
    for old_chunk_id, mentions in mentions_by_chunk.items():
        new_chunk_id = chunk_id_map.get(old_chunk_id)
        if new_chunk_id is None:
            continue
        cloned_mentions.extend(
            MentionRecord(
                chunk_id=new_chunk_id,
                entity_id=mention.entity_id,
                term_source=mention.term_source,
                surface_form=mention.surface_form,
                start_char=mention.start_char,
                end_char=mention.end_char,
            )
            for mention in mentions
        )
    return cloned_chunks, cloned_mentions


def _clone_citation_label(
    old_label: str, old_source_rel_path: str, new_source_rel_path: str
) -> str:
    page_fragment = ""
    if old_label.startswith(old_source_rel_path) and "#page=" in old_label:
        page_fragment = old_label.split("#page=", 1)[1]
    if page_fragment:
        try:
            return make_citation_label(new_source_rel_path, int(page_fragment))
        except ValueError:
            return f"{new_source_rel_path}#page={page_fragment}"
    return make_citation_label(new_source_rel_path)
