from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from lxd.retrieval.dense import embed_query
from lxd.retrieval.expansion import expand_question
from lxd.retrieval.rerank import rerank_chunks
from lxd.settings.models import RuntimeConfig
from lxd.stores.lancedb import connect_lancedb, open_chunk_table
from lxd.stores.lancedb import search_chunks as search_vector_chunks
from lxd.stores.sqlite import (
    build_store_paths,
    connect_sqlite,
    initialize_schema,
    list_allowed_domains,
    load_ingest_config_snapshot,
    summarize_store,
)
from lxd.synthesis.answering import (
    AnswerEnvelope,
    EvidenceChunk,
    insufficient_evidence_answer,
    no_results_answer,
    synthesize_answer,
)

_MAX_LIMIT = 50
_MIN_EVIDENCE_CHUNKS = 2
_MIN_EVIDENCE_CHARS = 400
_RRF_K = 20
_QUERY_STOPWORDS = {
    "a",
    "an",
    "are",
    "about",
    "cover",
    "covers",
    "define",
    "describe",
    "do",
    "does",
    "explain",
    "is",
    "mean",
    "means",
    "of",
    "tell",
    "the",
    "what",
}
_GENERIC_QUERY_TERMS = {
    "framework",
    "instruction",
    "model",
    "principle",
    "principles",
    "theory",
}


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    document_id: str
    citation_label: str
    source_rel_path: str
    source_path: str
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


@dataclass(frozen=True)
class SearchOutcome:
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
    _validate_question(question)
    requested_limit = config.retrieval.dense_top_k if limit is None else limit
    _validate_limit(requested_limit)

    store_paths = build_store_paths(config.paths.data_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        allowed_domains = list_allowed_domains(connection)
        _validate_domain(domain, allowed_domains)
        config_drift_warnings = _config_drift_warnings(connection, config)
        store_summary = summarize_store(
            connection,
            ontology_file_count=0,
            matcher_term_count=0,
            matcher_termset_hash=None,
            ontology_snapshot_hash=None,
            config_drift_warnings=config_drift_warnings,
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
            config_drift_warnings=config_drift_warnings,
        )

    expansion = expand_question(question.strip(), config)
    query_vector = embed_query(config, expansion.expanded_question)
    table = open_chunk_table(
        connect_lancedb(store_paths.lancedb_path), vector_size=config.models.embed_dims
    )
    target_source_count = max(
        requested_limit,
        config.retrieval.dense_top_k,
        config.retrieval.rerank_top_k,
    )
    ranked = _dense_ranked_candidates(
        table=table,
        query_vector=query_vector,
        domain=domain,
        requested_limit=requested_limit,
        target_source_count=target_source_count,
        rerank_top_k=config.retrieval.rerank_top_k,
    )
    representative_candidates = _unique_source_prefix(ranked, target_source_count)
    rerank_limit = min(len(representative_candidates), config.retrieval.rerank_top_k)
    rerank_inputs = representative_candidates[:rerank_limit]
    reranked = rerank_chunks(question, rerank_inputs, config)
    fused_prefix = _fuse_ranked_prefix(
        question=question,
        dense_prefix=rerank_inputs,
        reranked_prefix=reranked.ranked,
        lexical_fusion_weight=config.retrieval.lexical_fusion_weight,
    )
    merged_ranked = _merge_ranked_prefix(ranked, fused_prefix)[:requested_limit]
    return SearchOutcome(
        ranked=merged_ranked,
        warnings=reranked.warnings,
        reranking_applied=reranked.applied,
        expansion_applied=bool(expansion.added_terms),
        matched_entity_ids=expansion.matched_entity_ids,
        expansion_terms=expansion.added_terms,
        config_drift_warnings=config_drift_warnings,
    )


def answer_question(
    question: str,
    config: RuntimeConfig,
    domain: str | None = None,
) -> AnswerEnvelope:
    outcome = search_chunks(question=question, config=config, domain=domain)
    metadata: dict[str, object] = {
        "reranking_applied": outcome.reranking_applied,
        "expansion_applied": outcome.expansion_applied,
        "matched_entity_ids": outcome.matched_entity_ids,
        "expansion_terms": outcome.expansion_terms,
        "result_count": len(outcome.ranked),
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
    answer = synthesize_answer(question, evidence, config)
    warnings = [*outcome.warnings, *outcome.config_drift_warnings, *answer.warnings]
    return AnswerEnvelope(
        answer_status=answer.answer_status,
        answer_text=answer.answer_text,
        citations=answer.citations,
        warnings=warnings,
        metadata=metadata,
    )


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


def _dense_ranked_candidates(
    *,
    table: object,
    query_vector: list[float],
    domain: str | None,
    requested_limit: int,
    target_source_count: int,
    rerank_top_k: int,
) -> list[RankedChunk]:
    raw_limit = min(_MAX_LIMIT, max(requested_limit, rerank_top_k))
    ranked: list[RankedChunk] = []
    while True:
        dense_hits = search_vector_chunks(
            table,
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
                source_path=item.source_path,
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
                score=-item.score,
            )
            for item in dense_hits
        ]
        if len(_unique_source_prefix(ranked, target_source_count)) >= target_source_count:
            return ranked
        if raw_limit >= _MAX_LIMIT or len(ranked) < raw_limit:
            return ranked
        raw_limit = min(_MAX_LIMIT, raw_limit + max(1, rerank_top_k))


def _merge_ranked_prefix(
    ranked: list[RankedChunk],
    ranked_prefix: list[RankedChunk],
) -> list[RankedChunk]:
    prefix_ids = {item.chunk_id for item in ranked_prefix}
    return [*ranked_prefix, *(item for item in ranked if item.chunk_id not in prefix_ids)]


def _unique_source_prefix(ranked: list[RankedChunk], limit: int) -> list[RankedChunk]:
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
    question: str,
    dense_prefix: list[RankedChunk],
    reranked_prefix: list[RankedChunk],
    lexical_fusion_weight: float,
) -> list[RankedChunk]:
    if not dense_prefix:
        return []
    dense_rank = {item.chunk_id: index for index, item in enumerate(dense_prefix, start=1)}
    rerank_rank = {item.chunk_id: index for index, item in enumerate(reranked_prefix, start=1)}
    lexical_rank = {
        item.chunk_id: index
        for index, item in enumerate(_lexically_ranked(question, dense_prefix), start=1)
    }
    return sorted(
        dense_prefix,
        key=lambda item: (
            -(
                _rrf_score(dense_rank[item.chunk_id])
                + (lexical_fusion_weight * _rrf_score(lexical_rank[item.chunk_id]))
                + _rrf_score(rerank_rank.get(item.chunk_id, len(dense_prefix) + 1))
            ),
            dense_rank[item.chunk_id],
        ),
    )


def _lexically_ranked(question: str, candidates: list[RankedChunk]) -> list[RankedChunk]:
    return sorted(
        candidates,
        key=lambda item: -_lexical_signal_score(question, item),
    )


def _lexical_signal_score(question: str, candidate: RankedChunk) -> float:
    significant_terms = _significant_query_terms(question)
    if not significant_terms:
        return 0.0
    candidate_text = _normalize_ranking_text(
        " ".join(
            [
                candidate.source_filename,
                candidate.source_rel_path,
                candidate.citation_label,
                candidate.score_hint[:240],
            ]
        )
    )
    specific_terms = [term for term in significant_terms if term not in _GENERIC_QUERY_TERMS]
    score = 0.0
    for term in significant_terms:
        if _contains_rank_term(candidate_text, term):
            score += 5.0 if term in specific_terms else 0.5
    if specific_terms and all(_contains_rank_term(candidate_text, term) for term in specific_terms):
        score += 10.0
    return score


def _significant_query_terms(question: str) -> list[str]:
    return [
        term
        for term in _normalize_ranking_text(question).split()
        if len(term) >= 3 and term not in _QUERY_STOPWORDS
    ]


def _normalize_ranking_text(value: str) -> str:
    normalized = value.casefold().replace("’", "'").replace("`", "'")
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _contains_rank_term(normalized_text: str, term: str) -> bool:
    padded_text = f" {normalized_text} "
    return f" {term} " in padded_text


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


def _config_drift_warnings(connection: sqlite3.Connection, config: RuntimeConfig) -> list[str]:
    stored = load_ingest_config_snapshot(connection)
    if not stored:
        return []
    current = _current_ingest_config(config)
    warnings: list[str] = []
    for key, current_value in current.items():
        stored_value = stored.get(key)
        if stored_value is None:
            warnings.append(f"Committed ingest config is missing '{key}'.")
            continue
        if stored_value != current_value:
            warnings.append(f"Config drift: {key} stored={stored_value} current={current_value}.")
    return warnings


def _current_ingest_config(config: RuntimeConfig) -> dict[str, str]:
    return {
        "paths.corpus_path": str(config.paths.corpus_path),
        "paths.ontology_path": str(config.paths.ontology_path),
        "paths.data_path": str(config.paths.data_path),
        "chunking.chunk_overlap": str(config.chunking.chunk_overlap),
        "chunking.chunk_size": str(config.chunking.chunk_size),
        "chunking.min_tokens": str(config.chunking.min_tokens),
        "chunking.strategy": config.chunking.strategy,
        "chunking.tokenizer_backend": config.chunking.tokenizer_backend,
        "chunking.tokenizer_name": config.chunking.tokenizer_name,
        "models.embed": config.models.embed,
        "models.embed_backend": config.models.embed_backend,
        "models.embed_dims": str(config.models.embed_dims),
    }


def _insufficient_evidence(evidence: list[EvidenceChunk]) -> bool:
    if len(evidence) < _MIN_EVIDENCE_CHUNKS:
        return True
    return sum(len(chunk.text.strip()) for chunk in evidence) < _MIN_EVIDENCE_CHARS
