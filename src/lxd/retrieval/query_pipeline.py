"""Run retrieval and answer synthesis orchestration pipelines."""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final

import structlog

from lxd.app.status import config_drift_warnings
from lxd.retrieval.dense import embed_query
from lxd.retrieval.expansion import ExpansionOutcome, expand_question
from lxd.retrieval.graph_routing import build_graph_context, format_graph_context_prompt
from lxd.retrieval.hyde import generate_hypothetical_answer
from lxd.retrieval.rerank import rerank_chunks
from lxd.retrieval.router import resolve_dense_top_k, route_query
from lxd.settings.models import RuntimeConfig
from lxd.stores.lancedb import (
    connect_lancedb,
    open_chunk_table,
    open_entity_table,
    search_chunks_hybrid,
    search_similar_entities,
)
from lxd.stores.models import StorePaths
from lxd.stores.sqlite.chunks import (
    load_chunk_centrality_signals,
    load_relation_chunk_ids,
)
from lxd.stores.sqlite.connection import (
    build_store_paths,
    connect_sqlite,
    initialize_schema,
)
from lxd.stores.sqlite.ontology import list_allowed_domains
from lxd.stores.sqlite.summary import summarize_store
from lxd.synthesis.answering import (
    AnswerEnvelope,
    EvidenceChunk,
    insufficient_evidence_answer,
    no_results_answer,
    no_retrieval_needed_answer,
    synthesize_answer,
)
from lxd.synthesis.sampler import Sampler

PhaseCallback = Callable[[int, str], None]

_log = structlog.get_logger(__name__)

_MAX_LIMIT: Final = 50
_MIN_EVIDENCE_CHUNKS: Final = 2
_MIN_EVIDENCE_CHARS: Final = 400
_RRF_K: Final = 20


@dataclass(frozen=True, slots=True)
class RankedChunk:
    """Retrieval chunk with metadata and ranking score.

    ``cited_sources`` and ``wiki_links`` are page-level signals carried
    through from the underlying chunk row. Empty when the chunk's source
    is not a wiki-formatted markdown page.

    ``central_entity_score`` and ``community_ids`` carry knowledge-graph
    signals: the highest PageRank across the chunk's mentioned entities,
    and the set of communities those entities belong to. Both default to
    "no signal" so the pipeline degrades gracefully when the graph is
    not yet built (centrality lane contributes nothing; community-aware
    diversification is a no-op).
    """

    chunk_id: str
    document_id: str
    citation_label: str
    source_rel_path: str
    source_filename: str
    source_type: str
    source_domain: str
    source_hash: str
    chunk_index: int
    chunk_occurrence: int
    token_count: int
    text: str
    score_hint: str
    metadata_json: str
    score: float
    cited_sources: tuple[str, ...] = ()
    wiki_links: tuple[str, ...] = ()
    central_entity_score: float = 0.0
    community_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Search results plus expansion/rerank diagnostics."""

    ranked: list[RankedChunk]
    warnings: list[str]
    reranking_applied: bool
    expansion_applied: bool
    matched_entity_ids: list[str]
    expansion_terms: list[str]
    config_drift_warnings: list[str]


def search_chunks(
    question: str,
    config: RuntimeConfig,
    domain: str | None = None,
    limit: int | None = None,
) -> SearchOutcome:
    """Run dense retrieval, optional rerank, and fusion.

    Args:
        question: User question text.
        config: Runtime configuration object.
        domain: Optional source domain filter.
        limit: Maximum number of records to return.

    Returns:
        Vector search matches ordered by similarity.
    """
    _validate_question(question)
    requested_limit = config.retrieval.dense_top_k if limit is None else limit
    _validate_limit(requested_limit)

    store_paths = build_store_paths(config.paths.data_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        allowed_domains = list_allowed_domains(connection)
        _validate_domain(domain, allowed_domains)
        drift_warnings = config_drift_warnings(connection, config)
        store_summary = summarize_store(
            connection,
            ontology_file_count=0,
            matcher_term_count=0,
            matcher_termset_hash=None,
            ontology_snapshot_hash=None,
            config_drift_warnings=drift_warnings,
        )
    finally:
        connection.close()

    if store_summary.chunk_count == 0:
        return SearchOutcome(
            ranked=[],
            warnings=["The searchable store is empty. Run ingest first."],
            reranking_applied=False,
            expansion_applied=False,
            matched_entity_ids=[],
            expansion_terms=[],
            config_drift_warnings=drift_warnings,
        )

    expansion = expand_question(question.strip(), config)
    embed_target = expansion.expanded_question
    if config.retrieval.hyde_enabled:
        hyde_text = generate_hypothetical_answer(expansion.expanded_question, config)
        if hyde_text:
            embed_target = hyde_text
    query_vector = embed_query(config, embed_target)
    expansion = _augment_with_embedding_neighbours(
        expansion=expansion,
        query_vector=query_vector,
        store_paths=store_paths,
        config=config,
    )
    table = open_chunk_table(
        connect_lancedb(store_paths.lancedb_path), vector_size=config.models.embed_dims
    )
    target_source_count = max(
        requested_limit,
        config.retrieval.dense_top_k,
        config.retrieval.rerank_top_k,
    )
    ranked = _hybrid_ranked_candidates(
        table=table,
        query=question,
        query_vector=query_vector,
        domain=domain,
        requested_limit=requested_limit,
        target_source_count=target_source_count,
        rerank_top_k=config.retrieval.rerank_top_k,
    )
    ranked = _attach_centrality_signals(store_paths, ranked)
    representative_candidates = _unique_source_prefix(ranked, target_source_count)
    rerank_limit = min(len(representative_candidates), config.retrieval.rerank_top_k)
    rerank_inputs = representative_candidates[:rerank_limit]
    reranked = rerank_chunks(question, rerank_inputs, config)
    relation_chunk_ids = _load_relation_chunk_ids(store_paths, expansion.matched_entity_ids)
    fused_prefix = _fuse_ranked_prefix(
        dense_prefix=rerank_inputs,
        reranked_prefix=reranked.ranked,
        relation_fusion_weight=config.retrieval.relation_fusion_weight,
        relation_chunk_ids=relation_chunk_ids,
        centrality_fusion_weight=config.retrieval.centrality_fusion_weight,
    )
    if config.retrieval.community_diversity_enabled:
        fused_prefix = _diversify_by_community(fused_prefix, len(fused_prefix))
    merged_ranked = _merge_ranked_prefix(ranked, fused_prefix)[:requested_limit]
    return SearchOutcome(
        ranked=merged_ranked,
        warnings=reranked.warnings,
        reranking_applied=reranked.applied,
        expansion_applied=bool(expansion.added_terms),
        matched_entity_ids=expansion.matched_entity_ids,
        expansion_terms=expansion.added_terms,
        config_drift_warnings=drift_warnings,
    )


def answer_question(
    question: str,
    config: RuntimeConfig,
    domain: str | None = None,
    on_phase: PhaseCallback | None = None,
    sampler: Sampler | None = None,
) -> AnswerEnvelope:
    """Generate an answer envelope from retrieval evidence.

    Adaptive flow:
      1. Route the question via :func:`lxd.retrieval.router.route_query`.
         If ``retrieve=False``, short-circuit with a canned "no-retrieval-needed"
         envelope — saves cost and avoids stuffing meaningless evidence
         into synthesis.
      2. Otherwise translate the route's ``breadth`` into a dense_top_k
         override, run retrieval / rerank / expansion, build graph
         context, and synthesise.

    Args:
        question: User question text.
        config: Runtime configuration object.
        domain: Optional source domain filter.
        on_phase: Optional progress callback. Called with ``(phase, message)``
            at each internal boundary — ``1`` when retrieval + fusion are
            complete, ``2`` when graph context has been built and synthesis
            is about to run. Callers that report progress to an MCP client
            wrap this with :func:`anyio.from_thread.run` so the sync
            worker thread can post to the async :class:`fastmcp.Context`.
            Never invoked on the ``no_retrieval_needed`` short-circuit.

    Returns:
        Synthesized answer with citations and route metadata.
    """
    route = route_query(question=question, config=config.adaptive_retrieval)
    route_metadata: dict[str, object] = {
        "router_retrieve": route.retrieve,
        "router_breadth": route.breadth,
        "router_rationale": route.rationale,
        "router_routed": route.routed,
    }
    route_warnings: list[str] = []
    if not route.routed:
        route_warnings.append("Query router fell back to default route — see router_rationale.")

    if not route.retrieve:
        skipped = no_retrieval_needed_answer(route.rationale)
        return AnswerEnvelope(
            answer_status=skipped.answer_status,
            answer_text=skipped.answer_text,
            citations=skipped.citations,
            warnings=route_warnings,
            metadata=route_metadata,
        )

    dense_top_k = resolve_dense_top_k(
        breadth=route.breadth,
        config=config.adaptive_retrieval,
        default_top_k=config.retrieval.dense_top_k,
    )

    outcome = search_chunks(question=question, config=config, domain=domain, limit=dense_top_k)
    if on_phase is not None:
        on_phase(1, "evidence ranked")

    # Build graph context from matched entities (graceful degradation)
    graph_context_prompt = _build_graph_context_prompt(config, outcome.matched_entity_ids)
    if on_phase is not None:
        on_phase(2, "synthesising answer")

    metadata: dict[str, object] = {
        **route_metadata,
        "reranking_applied": outcome.reranking_applied,
        "expansion_applied": outcome.expansion_applied,
        "matched_entity_ids": outcome.matched_entity_ids,
        "expansion_terms": outcome.expansion_terms,
        "result_count": len(outcome.ranked),
        "graph_context_applied": bool(graph_context_prompt),
        "dense_top_k": dense_top_k,
    }
    if not outcome.ranked:
        answer = no_results_answer()
        return AnswerEnvelope(
            answer_status=answer.answer_status,
            answer_text=answer.answer_text,
            citations=answer.citations,
            warnings=[*outcome.warnings, *outcome.config_drift_warnings],
            metadata=metadata,
        )

    evidence = [
        EvidenceChunk(
            citation_label=item.citation_label,
            text=item.text,
            score=item.score,
            cited_sources=item.cited_sources,
        )
        for item in outcome.ranked[: config.synthesis.max_chunks]
    ]
    if _insufficient_evidence(evidence):
        answer = insufficient_evidence_answer()
        return AnswerEnvelope(
            answer_status=answer.answer_status,
            answer_text=answer.answer_text,
            citations=[],
            warnings=[*outcome.warnings, *outcome.config_drift_warnings],
            metadata=metadata,
        )
    answer = synthesize_answer(
        question,
        evidence,
        config,
        graph_context_prompt=graph_context_prompt,
        sampler=sampler,
    )
    warnings = [*outcome.warnings, *outcome.config_drift_warnings, *answer.warnings]
    return AnswerEnvelope(
        answer_status=answer.answer_status,
        answer_text=answer.answer_text,
        citations=answer.citations,
        warnings=warnings,
        metadata=metadata,
    )


def _build_graph_context_prompt(config: RuntimeConfig, matched_entity_ids: list[str]) -> str:
    """Build graph context prompt from matched entity IDs.

    Returns empty string if no entities matched, store unavailable, or on any error.
    """
    if not matched_entity_ids:
        return ""
    store_paths = build_store_paths(config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return ""
    try:
        connection = connect_sqlite(store_paths.sqlite_path)
        try:
            context = build_graph_context(connection, matched_entity_ids, config)
            prompt = format_graph_context_prompt(context)
            if prompt:
                _log.info(
                    "graph context added to synthesis",
                    level=context.level,
                    entity_profiles=len(context.entity_profiles),
                    community_reports=len(context.community_reports),
                    claims=len(context.claims),
                )
            return prompt
        finally:
            connection.close()
    except (sqlite3.DatabaseError, OSError) as exc:
        _log.warning("graph_context_building_failed", exc_info=True, error=str(exc))
        return ""


def _validate_question(question: str) -> None:
    if not question.strip():
        raise ValueError("Question must be non-empty.")


def _validate_limit(limit: int) -> None:
    if limit < 1 or limit > _MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}.")


def _validate_domain(domain: str | None, allowed_domains: set[str]) -> None:
    if domain is None:
        return
    if not domain.strip():
        raise ValueError("domain must be non-empty when provided.")
    if not allowed_domains:
        raise ValueError("domain filtering is unavailable until ingest has committed corpus rows.")
    if domain not in allowed_domains:
        raise ValueError(
            f"Unknown domain '{domain}'. Allowed domains: {', '.join(sorted(allowed_domains))}"
        )


def _hybrid_ranked_candidates(
    *,
    table: object,
    query: str,
    query_vector: list[float],
    domain: str | None,
    requested_limit: int,
    target_source_count: int,
    rerank_top_k: int,
) -> list[RankedChunk]:
    """Fuse dense + BM25 candidates via LanceDB native hybrid search.

    Uses ``Table.search(query_type="hybrid")`` with ``RRFReranker`` — the
    engine issues one query, runs dense k-NN and BM25 FTS in parallel,
    and fuses them internally on ``_relevance_score``. Result is a
    single ordered stream; per-lane ranks are not exposed. The dense
    ``score`` on the returned ``RankedChunk`` carries the fused
    relevance (higher is better; the sign convention matches the
    dense-only path's negated cosine).

    Same overfetch-until-enough-unique-sources loop as the previous
    dense-only variant, so the downstream unique-source prefix
    guarantee is preserved.
    """
    raw_limit = min(_MAX_LIMIT, max(requested_limit, rerank_top_k))
    ranked: list[RankedChunk] = []
    while True:
        hits = search_chunks_hybrid(
            table,
            query=query,
            query_vector=query_vector,
            domain=domain,
            limit=raw_limit,
        )
        ranked = [
            RankedChunk(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                citation_label=item.citation_label,
                source_rel_path=item.source_rel_path,
                source_filename=item.source_filename,
                source_type=item.source_type,
                source_domain=item.source_domain,
                source_hash=item.source_hash,
                chunk_index=item.chunk_index,
                chunk_occurrence=item.chunk_occurrence,
                token_count=item.token_count,
                text=item.text,
                score_hint=item.score_hint,
                metadata_json=item.metadata_json,
                score=item.score,
                cited_sources=item.cited_sources,
                wiki_links=item.wiki_links,
            )
            for item in hits
        ]
        if len(_unique_source_prefix(ranked, target_source_count)) >= target_source_count:
            return ranked
        if raw_limit >= _MAX_LIMIT or len(ranked) < raw_limit:
            return ranked
        raw_limit = min(_MAX_LIMIT, raw_limit + max(1, rerank_top_k))


def _load_relation_chunk_ids(store_paths: object, entity_ids: list[str]) -> set[str]:
    """Load chunk IDs that have extracted relations for any of the queried entities.

    Returns an empty set when the store file does not exist, when the caller
    passed a non-``StorePaths`` value, or when SQLite reports a database-level
    error while loading (schema missing, corruption, etc.). I/O errors are
    logged with ``exc_info`` so the silent path is observable.
    """
    if not entity_ids or not isinstance(store_paths, StorePaths):
        return set()
    sqlite_path = store_paths.sqlite_path
    if not sqlite_path.exists():
        return set()
    try:
        connection = connect_sqlite(sqlite_path)
        try:
            return load_relation_chunk_ids(connection, entity_ids)
        finally:
            connection.close()
    except sqlite3.DatabaseError, OSError:
        _log.warning("load_relation_chunk_ids_failed", exc_info=True)
        return set()


def _augment_with_embedding_neighbours(
    *,
    expansion: ExpansionOutcome,
    query_vector: list[float],
    store_paths: StorePaths,
    config: RuntimeConfig,
) -> ExpansionOutcome:
    """Widen ``expansion.matched_entity_ids`` with the entities nearest to the query vector.

    The ``entity_embeddings`` LanceDB table holds per-entity vectors built
    during the knowledge-graph build. This helper looks up the nearest
    entities by cosine similarity and merges them into the expansion's
    matched-entity set so the relation lane in
    :func:`_fuse_ranked_prefix` benefits from semantic neighbourhoods, not
    just surface-form mention detection.

    Silently no-ops when the LanceDB store, the entity table, or the
    underlying KG build are not yet present; this keeps the retrieval
    path usable on a freshly ingested corpus before
    ``pixi run build-graph`` has run.
    """
    limit = max(1, config.expansion.max_terms)
    try:
        database = connect_lancedb(store_paths.lancedb_path)
        entity_table = open_entity_table(database, vector_size=config.models.embed_dims)
    except (FileNotFoundError, ValueError) as exc:
        _log.debug("embedding_entity_expansion_skipped", error=str(exc))
        return expansion
    try:
        nearest = search_similar_entities(entity_table, query_vector=query_vector, limit=limit)
    except (FileNotFoundError, ValueError) as exc:
        _log.debug("embedding_entity_search_failed", error=str(exc))
        return expansion

    if not nearest:
        return expansion

    existing = set(expansion.matched_entity_ids)
    additions = [
        str(row["entity_id"])
        for row in nearest
        if isinstance(row.get("entity_id"), str) and row["entity_id"] not in existing
    ]
    if not additions:
        return expansion
    return replace(
        expansion,
        matched_entity_ids=[*expansion.matched_entity_ids, *additions],
    )


def _merge_ranked_prefix(
    ranked: list[RankedChunk],
    ranked_prefix: list[RankedChunk],
) -> list[RankedChunk]:
    prefix_ids = {item.chunk_id for item in ranked_prefix}
    return [*ranked_prefix, *(item for item in ranked if item.chunk_id not in prefix_ids)]


def _unique_source_prefix(ranked: list[RankedChunk], limit: int) -> list[RankedChunk]:
    if limit <= 0:
        return []
    unique: list[RankedChunk] = []
    seen_sources: set[str] = set()
    for item in ranked:
        if item.source_rel_path in seen_sources:
            continue
        seen_sources.add(item.source_rel_path)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def _fuse_ranked_prefix(
    *,
    dense_prefix: list[RankedChunk],
    reranked_prefix: list[RankedChunk],
    relation_fusion_weight: float = 0.0,
    relation_chunk_ids: set[str] | None = None,
    centrality_fusion_weight: float = 0.0,
) -> list[RankedChunk]:
    """Fuse the hybrid-ranked prefix with rerank, relation, and centrality lanes.

    The dense+lexical fuse now lives inside LanceDB's native hybrid search
    (see :func:`_hybrid_ranked_candidates`), so the incoming
    ``dense_prefix`` order already carries the fused dense+BM25 signal on
    ``item.score``. This fuser layers three remaining Python-side lanes on
    top: cross-encoder rerank, matched-relation membership, and per-entity
    PageRank. The lexical lane and its ``lexical_fusion_weight`` knob are
    gone by construction.
    """
    if not dense_prefix:
        return []
    _relation_chunk_ids = relation_chunk_ids or set()
    dense_rank = {item.chunk_id: index for index, item in enumerate(dense_prefix, start=1)}
    rerank_rank = {item.chunk_id: index for index, item in enumerate(reranked_prefix, start=1)}
    relation_ranked = sorted(
        dense_prefix,
        key=lambda c: (0 if c.chunk_id in _relation_chunk_ids else 1, dense_rank[c.chunk_id]),
    )
    relation_rank = {item.chunk_id: index for index, item in enumerate(relation_ranked, start=1)}
    centrality_ranked = sorted(
        dense_prefix,
        key=lambda c: (-c.central_entity_score, dense_rank[c.chunk_id]),
    )
    centrality_rank = {
        item.chunk_id: index for index, item in enumerate(centrality_ranked, start=1)
    }
    return sorted(
        dense_prefix,
        key=lambda item: (
            -(
                _rrf_score(dense_rank[item.chunk_id])
                + _rrf_score(rerank_rank.get(item.chunk_id, len(dense_prefix) + 1))
                + (relation_fusion_weight * _rrf_score(relation_rank[item.chunk_id]))
                + (centrality_fusion_weight * _rrf_score(centrality_rank[item.chunk_id]))
            ),
            dense_rank[item.chunk_id],
        ),
    )


def _attach_centrality_signals(store_paths: object, ranked: list[RankedChunk]) -> list[RankedChunk]:
    """Populate ``central_entity_score`` and ``community_ids`` on each chunk.

    A chunk that mentions no profiled entity, or whose entities have no
    community assignment yet (graph not built), keeps its default
    ``(0.0, ())`` — the centrality lane and community-aware
    diversification then become no-ops, so retrieval degrades gracefully.
    """
    if not ranked or not isinstance(store_paths, StorePaths):
        return ranked
    if not store_paths.sqlite_path.exists():
        return ranked
    chunk_ids = [item.chunk_id for item in ranked]
    try:
        connection = connect_sqlite(store_paths.sqlite_path)
        try:
            signals = load_chunk_centrality_signals(connection, chunk_ids)
        finally:
            connection.close()
    except sqlite3.DatabaseError, OSError:
        _log.warning("attach_centrality_signals_failed", exc_info=True)
        return ranked
    if not signals:
        return ranked
    return [
        replace(
            item,
            central_entity_score=signals[item.chunk_id].max_pagerank,
            community_ids=signals[item.chunk_id].community_ids,
        )
        if item.chunk_id in signals
        else item
        for item in ranked
    ]


def _diversify_by_community(ranked: list[RankedChunk], limit: int) -> list[RankedChunk]:
    """Reorder so the first hits cover distinct communities before doubling up.

    A simple round-robin: take the first chunk whose community has not
    yet appeared, then the next, until every community in the prefix is
    represented; only then revisit chunks from already-seen communities.
    Chunks with empty ``community_ids`` (graph not built or entity not
    yet profiled) are treated as "no community" and only deferred to
    after the community-tagged chunks of equal rank.
    """
    if limit <= 0:
        return []
    if not ranked:
        return ranked
    seen_communities: set[int] = set()
    untagged_chunks: list[RankedChunk] = []
    first_pass: list[RankedChunk] = []
    revisit: list[RankedChunk] = []
    for item in ranked:
        if not item.community_ids:
            untagged_chunks.append(item)
            continue
        novel_communities = set(item.community_ids) - seen_communities
        if novel_communities:
            seen_communities.update(item.community_ids)
            first_pass.append(item)
        else:
            revisit.append(item)
    return [*first_pass, *revisit, *untagged_chunks][:limit]


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


def _insufficient_evidence(evidence: list[EvidenceChunk]) -> bool:
    if len(evidence) < _MIN_EVIDENCE_CHUNKS:
        return True
    return sum(len(chunk.text.strip()) for chunk in evidence) < _MIN_EVIDENCE_CHARS
