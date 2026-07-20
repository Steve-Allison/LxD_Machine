"""Run retrieval and answer synthesis orchestration pipelines."""

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, Literal

import structlog

from lxd.app.status import config_drift_warnings
from lxd.domain.brief import LearnerBrief
from lxd.domain.ids import blake3_hex
from lxd.domain.limits import MAX_RETRIEVAL_LIMIT
from lxd.domain.time import utc_now
from lxd.retrieval.dense import embed_query
from lxd.retrieval.expansion import ExpansionOutcome, expand_question
from lxd.retrieval.graph_lane import GraphLaneHit, graph_lane_chunk_ids, load_graph_lane_hits
from lxd.retrieval.graph_routing import (
    GraphContext,
    build_graph_context,
    format_graph_context_prompt,
)
from lxd.retrieval.hyde import generate_hypothetical_answer
from lxd.retrieval.multi_query import generate_query_paraphrases
from lxd.retrieval.rerank import rerank_chunks
from lxd.retrieval.router import RouteBreadth, resolve_dense_top_k, route_query
from lxd.retrieval.stores import open_retrieval_stores
from lxd.settings.models import RuntimeConfig
from lxd.stores.lancedb import (
    connect_lancedb,
    open_entity_table,
    search_chunks_hybrid,
    search_similar_entities,
)
from lxd.stores.models import (
    ChunkRecord,
    SessionRecord,
    SessionTurnRecord,
    StorePaths,
    VectorSearchRecord,
)
from lxd.stores.sqlite.chunks import (
    load_chunk_centrality_signals,
    load_chunk_record_by_id,
    load_relation_chunk_ids,
)
from lxd.stores.sqlite.connection import build_store_paths, connect_sqlite
from lxd.stores.sqlite.ontology import list_allowed_domains
from lxd.stores.sqlite.sessions import (
    append_turn,
    load_session,
    update_last_artefact,
    upsert_session_brief,
)
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
NoticeLevel = Literal["info", "warning", "error"]
NoticeCallback = Callable[[NoticeLevel, str], None]

_log = structlog.get_logger(__name__)

_MAX_LIMIT: Final = MAX_RETRIEVAL_LIMIT
_MIN_EVIDENCE_CHUNKS: Final = 2
_MIN_EVIDENCE_CHARS: Final = 400
_RRF_K: Final = 20
# Multi-query fan-out issues one hybrid search per query variant (original +
# paraphrases); this caps the fused candidate pool so a large
# ``multi_query_count`` cannot balloon the downstream centrality lookup /
# rerank cost.
_MULTI_QUERY_POOL_CAP: Final = MAX_RETRIEVAL_LIMIT * 2
# Breadth widens narrow < standard < broad — used to gate HyDE against
# ``retrieval.hyde_min_breadth`` (narrow factual lookups already embed
# well as literal questions, so HyDE is skipped there by default).
_BREADTH_RANK: Final[dict[RouteBreadth, int]] = {"narrow": 0, "standard": 1, "broad": 2}


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

    @classmethod
    def from_vector_hit(
        cls,
        item: VectorSearchRecord,
        *,
        central_entity_score: float = 0.0,
        community_ids: tuple[int, ...] = (),
    ) -> RankedChunk:
        """Lift a LanceDB hit into a ranked chunk, optionally with KG signals."""
        return cls(
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
            central_entity_score=central_entity_score,
            community_ids=community_ids,
        )

    @classmethod
    def from_chunk_record(cls, record: ChunkRecord, *, score: float) -> RankedChunk:
        """Lift a SQLite chunk record into a ranked chunk for graph-lane appends.

        Used when a claim-linked chunk did not surface in the dense/rerank
        prefix at all — the graph lane appends it directly from
        ``chunk_rows`` so a strong claim can still pull its source chunk
        into the fused prefix (see ``_fuse_ranked_prefix``'s
        ``graph_lane_chunk_ids`` lane).
        """
        return cls(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            citation_label=record.citation_label,
            source_rel_path=record.source_rel_path,
            source_filename=record.source_filename,
            source_type=record.source_type,
            source_domain=record.source_domain,
            source_hash=record.source_hash,
            chunk_index=record.chunk_index,
            chunk_occurrence=record.chunk_occurrence,
            token_count=record.token_count,
            text=record.text,
            score_hint=record.score_hint,
            metadata_json=record.metadata_json,
            score=score,
            cited_sources=record.cited_sources,
            wiki_links=record.wiki_links,
        )


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
    multi_query_applied: bool = False
    hyde_applied: bool = False


def search_chunks(
    question: str,
    config: RuntimeConfig,
    domain: str | None = None,
    limit: int | None = None,
    route_breadth: RouteBreadth | None = None,
) -> SearchOutcome:
    """Run dense retrieval, optional rerank, and fusion.

    Args:
        question: User question text.
        config: Runtime configuration object.
        domain: Optional source domain filter.
        limit: Maximum number of records to return.
        route_breadth: Breadth decided by :func:`lxd.retrieval.router.route_query`,
            used to gate HyDE against ``retrieval.hyde_min_breadth``.
            Callers that bypass the router (the eval harness, direct MCP
            search) leave this ``None``, which is treated as ``"standard"``.

    Returns:
        Vector search matches ordered by similarity.
    """
    _validate_question(question)
    requested_limit = config.retrieval.dense_top_k if limit is None else limit
    _validate_limit(requested_limit)
    breadth = route_breadth or "standard"

    with open_retrieval_stores(config) as stores:
        allowed_domains = list_allowed_domains(stores.sqlite)
        _validate_domain(domain, allowed_domains)
        drift_warnings = config_drift_warnings(stores.sqlite, config)
        store_summary = summarize_store(
            stores.sqlite,
            ontology_file_count=0,
            matcher_term_count=0,
            matcher_termset_hash=None,
            ontology_snapshot_hash=None,
            config_drift_warnings=drift_warnings,
        )

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
        hyde_applied = False
        min_breadth_rank = _BREADTH_RANK[config.retrieval.hyde_min_breadth]
        hyde_breadth_reached = _BREADTH_RANK[breadth] >= min_breadth_rank
        if config.retrieval.hyde_enabled and hyde_breadth_reached:
            hyde_text = generate_hypothetical_answer(expansion.expanded_question, config)
            if hyde_text:
                embed_target = hyde_text
                hyde_applied = True
        query_vector = embed_query(config, embed_target)
        expansion = _augment_with_embedding_neighbours(
            expansion=expansion,
            query_vector=query_vector,
            store_paths=stores.paths,
            config=config,
        )
        target_source_count = max(
            requested_limit,
            config.retrieval.dense_top_k,
            config.retrieval.rerank_top_k,
        )
        paraphrases = (
            generate_query_paraphrases(question.strip(), config.retrieval)
            if config.retrieval.multi_query_enabled
            else []
        )
        multi_query_applied = bool(paraphrases)
        if multi_query_applied:
            query_variants: list[tuple[str, list[float]]] = [(question, query_vector)]
            query_variants.extend(
                (paraphrase, embed_query(config, paraphrase)) for paraphrase in paraphrases
            )
            candidate_lists = [
                _hybrid_ranked_candidates(
                    table=stores.chunk_table,
                    query=variant_query,
                    query_vector=variant_vector,
                    domain=domain,
                    requested_limit=requested_limit,
                    target_source_count=target_source_count,
                    rerank_top_k=config.retrieval.rerank_top_k,
                )
                for variant_query, variant_vector in query_variants
            ]
            ranked = _rrf_fuse_candidate_lists(candidate_lists, cap=_MULTI_QUERY_POOL_CAP)
        else:
            ranked = _hybrid_ranked_candidates(
                table=stores.chunk_table,
                query=question,
                query_vector=query_vector,
                domain=domain,
                requested_limit=requested_limit,
                target_source_count=target_source_count,
                rerank_top_k=config.retrieval.rerank_top_k,
            )
        ranked = _attach_centrality_signals(stores.sqlite, ranked)
        representative_candidates = _unique_source_prefix(ranked, target_source_count)
        rerank_limit = min(len(representative_candidates), config.retrieval.rerank_top_k)
        rerank_inputs = representative_candidates[:rerank_limit]
        reranked = rerank_chunks(question, rerank_inputs, config)
        relation_chunk_ids = _relation_chunk_ids(stores.sqlite, expansion.matched_entity_ids)
        graph_lane_hits = _graph_lane_hits(stores.sqlite, expansion.matched_entity_ids, config)
        fusion_dense_prefix = _append_missing_graph_lane_chunks(
            stores.sqlite,
            dense_prefix=rerank_inputs,
            hits=graph_lane_hits,
            requested_limit=requested_limit,
        )
        fused_prefix = _fuse_ranked_prefix(
            dense_prefix=fusion_dense_prefix,
            reranked_prefix=reranked.ranked,
            relation_fusion_weight=config.retrieval.relation_fusion_weight,
            relation_chunk_ids=relation_chunk_ids,
            centrality_fusion_weight=config.retrieval.centrality_fusion_weight,
            graph_lane_fusion_weight=config.retrieval.graph_lane_fusion_weight,
            graph_lane_chunk_ids=set(graph_lane_chunk_ids(graph_lane_hits)),
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
            multi_query_applied=multi_query_applied,
            hyde_applied=hyde_applied,
        )


def answer_question(
    question: str,
    config: RuntimeConfig,
    domain: str | None = None,
    on_phase: PhaseCallback | None = None,
    on_notice: NoticeCallback | None = None,
    sampler: Sampler | None = None,
    *,
    audience: str | None = None,
    modality: str | None = None,
    bloom_target: str | None = None,
    constraints: str | None = None,
    session_id: str | None = None,
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
        audience: Optional target-audience brief field (see
            :class:`lxd.domain.brief.LearnerBrief`).
        modality: Optional delivery-modality brief field.
        bloom_target: Optional Bloom's-taxonomy-level brief field.
        constraints: Optional free-text constraints brief field.
        session_id: Optional session ID. When set, the request's brief
            fields are merged over any brief already on file for the
            session (request fields win), the merged brief is persisted,
            and a user/assistant turn pair is appended once the answer is
            finalised. Session persistence degrades gracefully — any
            SQLite failure logs a warning and falls back to treating the
            request as brief-only / stateless.

    Returns:
        Synthesized answer with citations and route metadata.
    """
    brief = _resolve_learner_brief(
        config,
        LearnerBrief(
            audience=audience,
            modality=modality,
            bloom_target=bloom_target,
            constraints=constraints,
            session_id=session_id,
        ),
    )

    def _finalize(envelope: AnswerEnvelope) -> AnswerEnvelope:
        _persist_session_turn(config, brief.session_id, question=question, answer=envelope)
        return envelope

    def _notice(level: NoticeLevel, message: str) -> None:
        """Fan out one degradation notice to both the streaming callback (if
        wired) and the envelope's buffered warnings list. Callers receive
        both: the client sees the notice live during the tool call, and the
        returned envelope carries the full list for post-hoc inspection.
        """
        if on_notice is not None and level in ("warning", "error"):
            on_notice(level, message)

    route = route_query(question=question, config=config.adaptive_retrieval)
    route_metadata: dict[str, object] = {
        "router_retrieve": route.retrieve,
        "router_breadth": route.breadth,
        "router_rationale": route.rationale,
        "router_routed": route.routed,
        "router_path": route.router_path,
    }
    route_warnings: list[str] = []
    if not route.routed:
        message = "Query router fell back to default route — see router_rationale."
        route_warnings.append(message)
        _notice("warning", message)

    if not route.retrieve:
        skipped = no_retrieval_needed_answer(route.rationale)
        return _finalize(
            AnswerEnvelope(
                answer_status=skipped.answer_status,
                answer_text=skipped.answer_text,
                citations=skipped.citations,
                warnings=route_warnings,
                metadata=route_metadata,
            )
        )

    dense_top_k = resolve_dense_top_k(
        breadth=route.breadth,
        config=config.adaptive_retrieval,
        default_top_k=config.retrieval.dense_top_k,
    )

    outcome = search_chunks(
        question=question,
        config=config,
        domain=domain,
        limit=dense_top_k,
        route_breadth=route.breadth,
    )
    for warning in outcome.warnings:
        _notice("warning", warning)
    for warning in outcome.config_drift_warnings:
        _notice("warning", warning)
    if on_phase is not None:
        on_phase(1, "evidence ranked")

    # Build graph context once — synthesis uses the prompt; deep MCP tools
    # reuse the structured object without a second SQLite round-trip.
    graph_context = _load_graph_context(config, outcome.matched_entity_ids)
    graph_context_prompt = format_graph_context_prompt(graph_context) if graph_context else ""
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
        "hyde_applied": outcome.hyde_applied,
        "multi_query_applied": outcome.multi_query_applied,
    }
    if not outcome.ranked:
        answer = no_results_answer()
        return _finalize(
            AnswerEnvelope(
                answer_status=answer.answer_status,
                answer_text=answer.answer_text,
                citations=answer.citations,
                warnings=[*outcome.warnings, *outcome.config_drift_warnings],
                metadata=metadata,
                graph_context=graph_context,
            )
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
        return _finalize(
            AnswerEnvelope(
                answer_status=answer.answer_status,
                answer_text=answer.answer_text,
                citations=[],
                warnings=[*outcome.warnings, *outcome.config_drift_warnings],
                metadata=metadata,
                graph_context=graph_context,
            )
        )
    answer = synthesize_answer(
        question,
        evidence,
        config,
        graph_context_prompt=graph_context_prompt,
        brief=brief,
        sampler=sampler,
    )
    for warning in answer.warnings:
        _notice("warning", warning)
    warnings = [*outcome.warnings, *outcome.config_drift_warnings, *answer.warnings]
    return _finalize(
        AnswerEnvelope(
            answer_status=answer.answer_status,
            answer_text=answer.answer_text,
            citations=answer.citations,
            warnings=warnings,
            metadata=metadata,
            sentence_citations=answer.sentence_citations,
            graph_context=graph_context,
        )
    )


def _load_graph_context(
    config: RuntimeConfig, matched_entity_ids: list[str]
) -> GraphContext | None:
    """Load structured graph context for matched entities.

    Returns ``None`` if no entities matched, the store is unavailable, or
    any error occurs (graceful degradation).
    """
    if not matched_entity_ids:
        return None
    store_paths = build_store_paths(config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return None
    try:
        connection = connect_sqlite(store_paths.sqlite_path)
        try:
            context = build_graph_context(connection, matched_entity_ids, config)
            if context.level != "none":
                _log.info(
                    "graph context added to synthesis",
                    level=context.level,
                    entity_profiles=len(context.entity_profiles),
                    community_reports=len(context.community_reports),
                    claims=len(context.claims),
                )
            return context
        finally:
            connection.close()
    except (sqlite3.DatabaseError, OSError) as exc:
        _log.warning("graph_context_building_failed", exc_info=True, error=str(exc))
        return None


def _resolve_learner_brief(config: RuntimeConfig, request_brief: LearnerBrief) -> LearnerBrief:
    """Merge the request's brief fields over any brief already on file.

    Request fields always win (``LearnerBrief.merge_over``); the merged
    brief is written back so a later turn in the same session sees the
    latest values. A missing store, missing session row, or any SQLite
    error degrades to "use the request brief as-is" rather than failing
    the answer.
    """
    if request_brief.session_id is None:
        return request_brief
    store_paths = build_store_paths(config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return request_brief
    try:
        connection = connect_sqlite(store_paths.sqlite_path)
        try:
            stored = load_session(connection, request_brief.session_id)
            stored_brief = (
                LearnerBrief(
                    audience=stored.audience,
                    modality=stored.modality,
                    bloom_target=stored.bloom_target,
                    constraints=stored.constraints_text,
                    session_id=stored.session_id,
                )
                if stored is not None
                else LearnerBrief(session_id=request_brief.session_id)
            )
            merged = request_brief.merge_over(stored_brief)
            now = utc_now()
            upsert_session_brief(
                connection,
                SessionRecord(
                    session_id=merged.session_id or request_brief.session_id,
                    audience=merged.audience,
                    modality=merged.modality,
                    bloom_target=merged.bloom_target,
                    constraints_text=merged.constraints,
                    created_at=stored.created_at if stored is not None else now,
                    updated_at=now,
                    last_artefact_json=stored.last_artefact_json if stored is not None else "{}",
                ),
            )
            return merged
        finally:
            connection.close()
    except (sqlite3.DatabaseError, OSError) as exc:
        _log.warning("session_brief_resolution_failed", exc_info=True, error=str(exc))
        return request_brief


def _persist_session_turn(
    config: RuntimeConfig, session_id: str | None, *, question: str, answer: AnswerEnvelope
) -> None:
    """Append a user/assistant turn pair and update the last-artefact reference.

    No-op when ``session_id`` is unset. Any SQLite failure is logged and
    swallowed — turn history is best-effort product state, never a reason
    to fail an already-computed answer.
    """
    if session_id is None:
        return
    store_paths = build_store_paths(config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return
    now = utc_now()
    artefact_json = json.dumps(
        {
            "answer_status": answer.answer_status.value,
            "citations": answer.citations,
        }
    )
    try:
        connection = connect_sqlite(store_paths.sqlite_path)
        try:
            user_turn_id = blake3_hex(session_id, "user", question, now)
            append_turn(
                connection,
                SessionTurnRecord(
                    turn_id=user_turn_id,
                    session_id=session_id,
                    role="user",
                    content_json=json.dumps({"question": question}),
                    created_at=now,
                ),
            )
            assistant_turn_id = blake3_hex(session_id, "assistant", answer.answer_text, now)
            append_turn(
                connection,
                SessionTurnRecord(
                    turn_id=assistant_turn_id,
                    session_id=session_id,
                    role="assistant",
                    content_json=artefact_json,
                    created_at=now,
                ),
            )
            update_last_artefact(
                connection,
                session_id=session_id,
                last_artefact_json=artefact_json,
                updated_at=now,
            )
        finally:
            connection.close()
    except (sqlite3.DatabaseError, OSError) as exc:
        _log.warning("session_turn_persistence_failed", exc_info=True, error=str(exc))


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
        ranked = [RankedChunk.from_vector_hit(item) for item in hits]
        if len(_unique_source_prefix(ranked, target_source_count)) >= target_source_count:
            return ranked
        if raw_limit >= _MAX_LIMIT or len(ranked) < raw_limit:
            return ranked
        raw_limit = min(_MAX_LIMIT, raw_limit + max(1, rerank_top_k))


def _relation_chunk_ids(connection: sqlite3.Connection, entity_ids: list[str]) -> set[str]:
    """Load chunk IDs that have extracted relations for any of the queried entities."""
    if not entity_ids:
        return set()
    try:
        return load_relation_chunk_ids(connection, entity_ids)
    except sqlite3.DatabaseError, OSError:
        _log.warning("load_relation_chunk_ids_failed", exc_info=True)
        return set()


def _graph_lane_hits(
    connection: sqlite3.Connection,
    entity_ids: list[str],
    config: RuntimeConfig,
) -> list[GraphLaneHit]:
    """Load graph-lane hits, degrading to no signal on any store error."""
    if not entity_ids or not config.retrieval.graph_lane_enabled:
        return []
    try:
        return load_graph_lane_hits(connection, entity_ids, config)
    except sqlite3.DatabaseError, OSError:
        _log.warning("load_graph_lane_hits_failed", exc_info=True)
        return []


def _append_missing_graph_lane_chunks(
    connection: sqlite3.Connection,
    *,
    dense_prefix: list[RankedChunk],
    hits: list[GraphLaneHit],
    requested_limit: int,
) -> list[RankedChunk]:
    """Append claim-linked chunks missing from ``dense_prefix`` so they can be fused.

    A claim can point at a chunk that never surfaced in the dense/BM25
    candidate pool (e.g. a short chunk that scored low lexically but
    carries a high-confidence claim). Without this, the graph lane could
    only reorder chunks that were already present — it could never pull
    a claim-backed chunk into view. Appends are capped at
    ``requested_limit`` total chunks so the graph lane cannot silently
    balloon the fused prefix past what the caller asked for.
    """
    claim_chunk_ids = graph_lane_chunk_ids(hits)
    if not claim_chunk_ids:
        return dense_prefix
    existing_ids = {item.chunk_id for item in dense_prefix}
    claim_scores = {hit.chunk_id: hit.score for hit in hits if hit.lane_kind == "claim"}
    appended: list[RankedChunk] = []
    for chunk_id in claim_chunk_ids:
        if chunk_id in existing_ids:
            continue
        if len(dense_prefix) + len(appended) >= requested_limit:
            break
        try:
            record = load_chunk_record_by_id(connection, chunk_id)
        except sqlite3.DatabaseError, OSError:
            _log.warning("load_chunk_record_by_id_failed", exc_info=True)
            continue
        if record is None:
            continue
        appended.append(
            RankedChunk.from_chunk_record(record, score=claim_scores.get(chunk_id, 0.0))
        )
        existing_ids.add(chunk_id)
    if not appended:
        return dense_prefix
    return [*dense_prefix, *appended]


def _load_relation_chunk_ids(store_paths: object, entity_ids: list[str]) -> set[str]:
    """Load relation chunk IDs when only store paths are available (legacy callers)."""
    if not entity_ids or not isinstance(store_paths, StorePaths):
        return set()
    sqlite_path = store_paths.sqlite_path
    if not sqlite_path.exists():
        return set()
    try:
        connection = connect_sqlite(sqlite_path)
        try:
            return _relation_chunk_ids(connection, entity_ids)
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
    graph_lane_fusion_weight: float = 0.0,
    graph_lane_chunk_ids: set[str] | None = None,
) -> list[RankedChunk]:
    """Fuse the hybrid-ranked prefix with rerank, relation, centrality, and graph lanes.

    The dense+lexical fuse now lives inside LanceDB's native hybrid search
    (see :func:`_hybrid_ranked_candidates`), so the incoming
    ``dense_prefix`` order already carries the fused dense+BM25 signal on
    ``item.score``. This fuser layers four remaining Python-side lanes on
    top: cross-encoder rerank, matched-relation membership, per-entity
    PageRank, and claim-linked chunks from the graph-as-retrieval-lane
    path (:mod:`lxd.retrieval.graph_lane`). The lexical lane and its
    ``lexical_fusion_weight`` knob are gone by construction.
    """
    if not dense_prefix:
        return []
    _relation_chunk_ids = relation_chunk_ids or set()
    _graph_lane_chunk_ids = graph_lane_chunk_ids or set()
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
    graph_lane_ranked = sorted(
        dense_prefix,
        key=lambda c: (0 if c.chunk_id in _graph_lane_chunk_ids else 1, dense_rank[c.chunk_id]),
    )
    graph_lane_rank = {
        item.chunk_id: index for index, item in enumerate(graph_lane_ranked, start=1)
    }
    return sorted(
        dense_prefix,
        key=lambda item: (
            -(
                _rrf_score(dense_rank[item.chunk_id])
                + _rrf_score(rerank_rank.get(item.chunk_id, len(dense_prefix) + 1))
                + (relation_fusion_weight * _rrf_score(relation_rank[item.chunk_id]))
                + (centrality_fusion_weight * _rrf_score(centrality_rank[item.chunk_id]))
                + (graph_lane_fusion_weight * _rrf_score(graph_lane_rank[item.chunk_id]))
            ),
            dense_rank[item.chunk_id],
        ),
    )


def _attach_centrality_signals(
    connection_or_paths: sqlite3.Connection | object, ranked: list[RankedChunk]
) -> list[RankedChunk]:
    """Populate ``central_entity_score`` and ``community_ids`` on each chunk.

    Accepts an open SQLite connection (preferred — no reconnect) or a
    :class:`StorePaths` for callers that only have paths. A chunk that
    mentions no profiled entity, or whose entities have no community
    assignment yet (graph not built), keeps its default ``(0.0, ())``.
    """
    if not ranked:
        return ranked
    connection: sqlite3.Connection | None
    owns_connection = False
    if isinstance(connection_or_paths, sqlite3.Connection):
        connection = connection_or_paths
    elif isinstance(connection_or_paths, StorePaths):
        if not connection_or_paths.sqlite_path.exists():
            return ranked
        connection = connect_sqlite(connection_or_paths.sqlite_path)
        owns_connection = True
    else:
        return ranked
    chunk_ids = [item.chunk_id for item in ranked]
    try:
        try:
            signals = load_chunk_centrality_signals(connection, chunk_ids)
        except sqlite3.DatabaseError, OSError:
            _log.warning("attach_centrality_signals_failed", exc_info=True)
            return ranked
    finally:
        if owns_connection:
            connection.close()
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


def _rrf_fuse_candidate_lists(
    candidate_lists: list[list[RankedChunk]],
    *,
    cap: int,
) -> list[RankedChunk]:
    """Reciprocal-rank-fuse the hybrid-ranked candidate lists from multi-query fan-out.

    Each query variant (the primary embed target plus any generated
    paraphrases) produces its own independently ranked candidate list.
    A chunk that surfaces in more than one list accumulates RRF score
    from each occurrence, so chunks robust across phrasings outrank
    chunks only one phrasing happened to retrieve. The first-seen
    ``RankedChunk`` instance is kept for a given ``chunk_id`` (its
    fields carry no per-variant state); the fused pool is truncated to
    ``cap`` so a large ``multi_query_count`` cannot balloon the
    downstream rerank / centrality-lookup cost.
    """
    if not candidate_lists:
        return []
    if len(candidate_lists) == 1:
        return candidate_lists[0][:cap]
    scores: dict[str, float] = {}
    first_seen: dict[str, RankedChunk] = {}
    for candidate_list in candidate_lists:
        for rank, item in enumerate(candidate_list, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + _rrf_score(rank)
            first_seen.setdefault(item.chunk_id, item)
    fused = sorted(first_seen.values(), key=lambda item: -scores[item.chunk_id])
    return fused[:cap]


def _insufficient_evidence(evidence: list[EvidenceChunk]) -> bool:
    if len(evidence) < _MIN_EVIDENCE_CHUNKS:
        return True
    return sum(len(chunk.text.strip()) for chunk in evidence) < _MIN_EVIDENCE_CHARS
