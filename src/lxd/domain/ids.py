"""Generate stable identifiers and content keys used across ingestion."""

from __future__ import annotations

from blake3 import blake3

_SEP = b"\x00"


def blake3_hex(*parts: str) -> str:
    """Hash ordered string parts into a deterministic BLAKE3 hex digest.

    Args:
        *parts: Ordered components of the logical key.

    Returns:
        Hex-encoded BLAKE3 digest.

    Constraints:
        Separates parts with a null-byte delimiter to avoid ambiguous concatenations.
    """
    hasher = blake3()
    for index, part in enumerate(parts):
        if index:
            hasher.update(_SEP)
        hasher.update(part.encode("utf-8"))
    return hasher.hexdigest()


def make_chunk_id(document_id: str, chunk_hash: str, chunk_occurrence: int) -> str:
    """Derive a stable chunk identifier from document and chunk identity.

    Args:
        document_id: Stable identifier for the source document.
        chunk_hash: Content hash of the chunk text.
        chunk_occurrence: Ordinal for repeated identical chunk hashes in one document.

    Returns:
        Deterministic chunk identifier hash.
    """
    return blake3_hex(document_id, chunk_hash, str(chunk_occurrence))


def make_graph_edge_key(
    origin_kind: str,
    source_file_rel_path: str,
    source_node_id: str,
    relation_type: str,
    target_node_id: str,
) -> str:
    """Derive a stable key for ontology graph edges.

    Args:
        origin_kind: Source origin namespace (for example `ontology` or `extracted`).
        source_file_rel_path: Relative source file path for provenance.
        source_node_id: Subject node identifier.
        relation_type: Canonical relation/predicate label.
        target_node_id: Object node identifier.

    Returns:
        Deterministic graph edge key.
    """
    return blake3_hex(
        origin_kind,
        source_file_rel_path,
        source_node_id,
        relation_type,
        target_node_id,
    )
