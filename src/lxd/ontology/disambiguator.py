"""Embedding-aware mention disambiguator.

Given an ambiguous surface form (one that maps to >1 ontology entity)
and the surrounding chunk text, embed the local window and pick the
candidate whose entity embedding is most similar by cosine.

Build dependency: this lane requires the ``entity_embeddings`` LanceDB
table to exist (built by :mod:`lxd.cli.graph`). On a freshly ingested
corpus before ``pixi run build-graph`` has been run, the disambiguator
fails gracefully — :func:`make_disambiguator` returns ``None`` and the
caller falls back to first-match behaviour.
"""

from __future__ import annotations

from collections.abc import Callable

import structlog

from lxd.retrieval.dense import embed_query
from lxd.settings.models import RuntimeConfig
from lxd.stores.lancedb import (
    connect_lancedb,
    search_similar_entities,
)
from lxd.stores.models import StorePaths

_log = structlog.get_logger(__name__)


# A mention disambiguator picks one entity id given the local window text and
# the candidate set. Returns ``None`` to signal "couldn't decide" — the
# caller then falls back to whatever the upstream policy is (typically
# first-match by ontology priority).
Disambiguator = Callable[[str, list[str]], str | None]


def make_disambiguator(
    *,
    config: RuntimeConfig,
    store_paths: StorePaths,
    top_k: int = 8,
) -> Disambiguator | None:
    """Build a disambiguator bound to the live entity-embeddings table.

    Returns ``None`` when the LanceDB store, the entity table, or the
    embedder are not available — the caller MUST treat this as "no
    disambiguation possible" and not raise. This contract preserves the
    "always-on, no toggle" rule (`feedback_mandatory_features`) while
    allowing graceful degradation on a fresh ingest before
    `pixi run build-graph` has populated `entity_embeddings`.

    Args:
        config: Runtime config; needed for embedder dispatch and dims.
        store_paths: Resolved data-dir paths; LanceDB lives under here.
        top_k: How many entity candidates to fetch from the entity table
            before intersecting with the per-term candidate set. Larger
            values cope with rare entities that don't make the global
            top of the cosine ranking.

    Returns:
        A callable ``(window_text, candidates) -> entity_id | None`` or
        ``None`` when prerequisites are missing. The caller is
        responsible for slicing the per-mention context window via
        :func:`context_window` before invoking it.
    """
    if not store_paths.lancedb_path.exists():
        _log.debug("disambiguator_unavailable", reason="lancedb_missing")
        return None
    try:
        database = connect_lancedb(store_paths.lancedb_path)
        # IMPORTANT: open WITHOUT creating. The disambiguation lane is a
        # consumer of the entity_embeddings table, never a producer; the
        # table is built by `pixi run build-graph`. If we used
        # `open_entity_table` here, we'd silently create an empty table
        # as a side effect of every ingest run, which (a) wastes disk,
        # (b) makes ingest's "no extra LanceDB writes" tests flake, and
        # (c) gives the disambiguator nothing to search anyway.
        entity_table = database.open_table("entity_embeddings")
    except (FileNotFoundError, ValueError) as exc:
        _log.debug("disambiguator_unavailable", reason=str(exc))
        return None
    try:
        if entity_table.count_rows() == 0:
            _log.debug("disambiguator_unavailable", reason="entity_table_empty")
            return None
    except (AttributeError, RuntimeError) as exc:
        _log.debug("disambiguator_unavailable", reason=f"count_rows_failed:{exc}")
        return None

    def disambiguate(window_text: str, candidates: list[str]) -> str | None:
        if not window_text.strip() or len(candidates) < 2:
            return None
        try:
            query_vector = embed_query(config, window_text)
        except Exception as exc:
            _log.debug("disambiguator_embed_failed", error=str(exc))
            return None
        try:
            nearest = search_similar_entities(
                entity_table,
                query_vector=query_vector,
                limit=top_k,
            )
        except (FileNotFoundError, ValueError) as exc:
            _log.debug("disambiguator_search_failed", error=str(exc))
            return None
        candidate_set = set(candidates)
        for row in nearest:
            entity_id = row.get("entity_id")
            if isinstance(entity_id, str) and entity_id in candidate_set:
                return entity_id
        return None

    return disambiguate


def context_window(text: str, *, start: int, end: int, radius: int = 200) -> str:
    """Slice a ±radius-character window around the mention span.

    Bounded by the text edges. The mention itself is included inside
    the window — embedding the surface form alongside its context lets
    the embedder disambiguate based on collocation, not just neighbour
    words.
    """
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right]
