from __future__ import annotations

from blake3 import blake3

_SEP = b"\x00"


def blake3_hex(*parts: str) -> str:
    hasher = blake3()
    for index, part in enumerate(parts):
        if index:
            hasher.update(_SEP)
        hasher.update(part.encode("utf-8"))
    return hasher.hexdigest()


def make_chunk_id(document_id: str, chunk_hash: str, chunk_occurrence: int) -> str:
    return blake3_hex(document_id, chunk_hash, str(chunk_occurrence))


def make_graph_edge_key(
    origin_kind: str,
    source_file_rel_path: str,
    source_node_id: str,
    relation_type: str,
    target_node_id: str,
) -> str:
    return blake3_hex(
        origin_kind,
        source_file_rel_path,
        source_node_id,
        relation_type,
        target_node_id,
    )
