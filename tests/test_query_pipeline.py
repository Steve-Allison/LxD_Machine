import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from lxd.app.status import current_ingest_config
from lxd.retrieval import query_pipeline as _query_pipeline
from lxd.retrieval.query_pipeline import RankedChunk
from lxd.retrieval.rerank import RerankOutcome
from lxd.settings.models import RuntimeConfig
from lxd.stores.schema import ensure_schema

# Private helpers exercised intentionally — same logical unit as
# query_pipeline; the tests below regress its retrieval shape.
_diversify_by_community = _query_pipeline._diversify_by_community  # pyright: ignore[reportPrivateUsage]
_fuse_ranked_prefix = _query_pipeline._fuse_ranked_prefix  # pyright: ignore[reportPrivateUsage]
_merge_ranked_prefix = _query_pipeline._merge_ranked_prefix  # pyright: ignore[reportPrivateUsage]
_rrf_fuse_candidate_lists = _query_pipeline._rrf_fuse_candidate_lists  # pyright: ignore[reportPrivateUsage]
_unique_source_prefix = _query_pipeline._unique_source_prefix  # pyright: ignore[reportPrivateUsage]


def _chunk(
    chunk_id: str,
    *,
    source_rel_path: str | None = None,
    score_hint: str | None = None,
    central_entity_score: float = 0.0,
    community_ids: tuple[int, ...] = (),
) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        citation_label=chunk_id,
        source_rel_path=source_rel_path or f"{chunk_id}.md",
        source_filename=Path(source_rel_path or f"{chunk_id}.md").name,
        source_type="markdown",
        source_domain="guides",
        source_hash=f"hash-{chunk_id}",
        chunk_index=0,
        chunk_occurrence=0,
        token_count=10,
        text=f"text-{chunk_id}",
        score_hint=score_hint or chunk_id,
        metadata_json="{}",
        score=0.0,
        central_entity_score=central_entity_score,
        community_ids=community_ids,
    )


def test_merge_ranked_prefix_preserves_dense_tail() -> None:
    ranked = [_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d"), _chunk("e")]
    ranked_prefix = [ranked[2], ranked[0], ranked[1]]

    merged = _merge_ranked_prefix(ranked, ranked_prefix)

    assert [item.chunk_id for item in merged] == ["c", "a", "b", "d", "e"]


def test_unique_source_prefix_deduplicates_sources() -> None:
    ranked = [
        _chunk("a1", source_rel_path="alpha.md"),
        _chunk("a2", source_rel_path="alpha.md"),
        _chunk("b1", source_rel_path="beta.md"),
        _chunk("c1", source_rel_path="gamma.md"),
    ]

    prefix = _unique_source_prefix(ranked, 3)

    assert [item.chunk_id for item in prefix] == ["a1", "b1", "c1"]


def test_diversify_by_community_picks_distinct_communities_first() -> None:
    """Top-N should cover distinct communities before any community
    appears twice."""
    ranked = [
        _chunk("a", community_ids=(1,)),
        _chunk("b", community_ids=(1,)),  # same community as a
        _chunk("c", community_ids=(2,)),
        _chunk("d", community_ids=(3,)),
        _chunk("e", community_ids=(2,)),  # same community as c
    ]
    diversified = _diversify_by_community(ranked, len(ranked))
    assert [item.chunk_id for item in diversified] == ["a", "c", "d", "b", "e"]


def test_diversify_by_community_defers_chunks_with_no_community() -> None:
    """Chunks without community signals (graph not yet built) come last,
    after every community-tagged chunk."""
    ranked = [
        _chunk("untagged-1", community_ids=()),
        _chunk("a", community_ids=(1,)),
        _chunk("untagged-2", community_ids=()),
        _chunk("b", community_ids=(2,)),
    ]
    diversified = _diversify_by_community(ranked, 4)
    assert [item.chunk_id for item in diversified] == ["a", "b", "untagged-1", "untagged-2"]


def test_diversify_by_community_is_noop_when_signals_absent() -> None:
    """If no chunk carries community ids, ordering is preserved."""
    ranked = [_chunk("a"), _chunk("b"), _chunk("c")]
    diversified = _diversify_by_community(ranked, len(ranked))
    assert [item.chunk_id for item in diversified] == ["a", "b", "c"]


def test_fuse_ranked_prefix_centrality_lane_promotes_high_pagerank_chunks() -> None:
    """A chunk with strictly higher central_entity_score should outrank
    its baseline-equal sibling once the centrality lane is enabled."""
    high_pr = _chunk("high", central_entity_score=0.9)
    low_pr = _chunk("low", central_entity_score=0.1)
    dense_prefix = [low_pr, high_pr]
    no_lane = _fuse_ranked_prefix(
        dense_prefix=dense_prefix,
        reranked_prefix=[],
        relation_fusion_weight=0.0,
        relation_chunk_ids=set(),
        centrality_fusion_weight=0.0,
    )
    with_lane = _fuse_ranked_prefix(
        dense_prefix=dense_prefix,
        reranked_prefix=[],
        relation_fusion_weight=0.0,
        relation_chunk_ids=set(),
        centrality_fusion_weight=10.0,
    )
    # Without the lane: dense order preserved (low first).
    assert [item.chunk_id for item in no_lane] == ["low", "high"]
    # With a strong lane: high-PR chunk surfaces.
    assert [item.chunk_id for item in with_lane] == ["high", "low"]


def test_rrf_fuse_candidate_lists_empty_input_yields_empty_output() -> None:
    assert _rrf_fuse_candidate_lists([], cap=10) == []


def test_rrf_fuse_candidate_lists_single_list_passes_through_capped() -> None:
    ranked = [_chunk("a"), _chunk("b"), _chunk("c")]
    fused = _rrf_fuse_candidate_lists([ranked], cap=2)
    assert [item.chunk_id for item in fused] == ["a", "b"]


def test_rrf_fuse_candidate_lists_promotes_chunk_seen_in_multiple_variants() -> None:
    """A chunk ranked lower in every list but present in every list should
    outrank a chunk that only tops a single variant's list once RRF scores
    accumulate across variants."""
    primary = [_chunk("only-in-primary"), _chunk("shared")]
    paraphrase_one = [_chunk("shared"), _chunk("only-in-paraphrase-1")]
    paraphrase_two = [_chunk("shared"), _chunk("only-in-paraphrase-2")]
    fused = _rrf_fuse_candidate_lists([primary, paraphrase_one, paraphrase_two], cap=10)
    assert fused[0].chunk_id == "shared"


def test_rrf_fuse_candidate_lists_deduplicates_chunk_ids() -> None:
    primary = [_chunk("a"), _chunk("b")]
    paraphrase = [_chunk("b"), _chunk("a")]
    fused = _rrf_fuse_candidate_lists([primary, paraphrase], cap=10)
    chunk_ids = [item.chunk_id for item in fused]
    assert sorted(chunk_ids) == ["a", "b"]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_rrf_fuse_candidate_lists_respects_cap() -> None:
    primary = [_chunk("a"), _chunk("b"), _chunk("c")]
    paraphrase = [_chunk("d"), _chunk("e"), _chunk("f")]
    fused = _rrf_fuse_candidate_lists([primary, paraphrase], cap=3)
    assert len(fused) == 3


def test_current_ingest_config_excludes_query_time_reranker_settings() -> None:
    config = SimpleNamespace(
        paths=SimpleNamespace(
            corpus_path=Path("/tmp/corpus"),
            ontology_path=Path("/tmp/ontology"),
            data_path=Path("/tmp/data"),
        ),
        chunking=SimpleNamespace(
            chunk_overlap=60,
            chunk_size=300,
            min_tokens=80,
            strategy="hybrid",
            tokenizer_backend="tiktoken",
            tokenizer_name="cl100k_base",
        ),
        models=SimpleNamespace(
            embed="nomic-embed-text",
            embed_dims=768,
            embed_backend="mlx",
            rerank="dengcao/Qwen3-Reranker-4B:Q4_K_M",
        ),
        reranker=SimpleNamespace(
            backend="llama_cpp",
            url="http://127.0.0.1:8012",
            endpoint="/v1/rerank",
        ),
    )

    snapshot = current_ingest_config(cast("RuntimeConfig", config))

    assert "models.rerank" not in snapshot
    assert "reranker.backend" not in snapshot
    assert "reranker.url" not in snapshot
    assert "reranker.endpoint" not in snapshot


def test_multi_query_batches_paraphrase_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    embed_texts_calls: list[list[str]] = []

    def fake_embed_texts(cfg: object, texts: list[str]) -> list[list[float]]:
        del cfg
        embed_texts_calls.append(list(texts))
        return [[0.1] * 8 for _ in texts]

    def fake_embed_query(cfg: object, text: str) -> list[float]:
        del cfg, text
        return [0.1] * 8

    monkeypatch.setattr(_query_pipeline, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(_query_pipeline, "embed_query", fake_embed_query)
    monkeypatch.setattr(
        _query_pipeline, "generate_query_paraphrases", lambda q, r: ["alt 1", "alt 2"]
    )
    monkeypatch.setattr(_query_pipeline, "_hybrid_ranked_candidates", lambda **kw: [_chunk("c1")])
    monkeypatch.setattr(
        _query_pipeline,
        "rerank_chunks",
        lambda q, c, cfg: RerankOutcome(ranked=c, warnings=[], applied=False),
    )
    monkeypatch.setattr(_query_pipeline, "list_allowed_domains", lambda conn: {"guides"})
    monkeypatch.setattr(_query_pipeline, "_attach_centrality_signals", lambda conn, r: r)
    monkeypatch.setattr(_query_pipeline, "_diversify_by_community", lambda r, limit_count: r)
    monkeypatch.setattr(
        _query_pipeline,
        "expand_question",
        lambda q, o: SimpleNamespace(
            expanded_question=q,
            matched_entities=[],
            matched_entity_ids=set(),
            expansion_terms=[],
            added_terms=[],
            original_question=q,
        ),
    )

    sqlite_conn = sqlite3.connect(":memory:")
    sqlite_conn.row_factory = sqlite3.Row
    ensure_schema(sqlite_conn)
    sqlite_conn.execute(
        "INSERT INTO corpus_manifest ("
        "source_rel_path, absolute_path, source_type, source_domain, blake3_hash, "
        "file_size_bytes, lifecycle_status, retrieval_status, last_seen_at) "
        "VALUES ('a.md', '/tmp/a.md', 'markdown', 'guides', 'hash', 10, 'current', 'searchable', '2026-10-18T00:00:00Z');"
    )
    sqlite_conn.execute(
        "INSERT INTO chunk_rows ("
        "chunk_id, document_id, source_rel_path, source_filename, source_type, "
        "source_domain, source_hash, citation_label, chunk_index, chunk_occurrence, "
        "token_count, text, chunk_hash, score_hint, metadata_json, embedding_model, embedding_dims) "
        "VALUES ('c1', 'd1', 'a.md', 'a.md', 'markdown', 'guides', 'h1', '[c1]', 0, 0, 10, 'text', 'h1', 'c1', '{}', 'm', 8);"
    )

    stores_mock = SimpleNamespace(
        paths=SimpleNamespace(
            sqlite_path=Path("/tmp/fake.sqlite"), lancedb_path=Path("/tmp/fake.lance")
        ),
        sqlite=sqlite_conn,
        chunk_table=None,
    )

    from contextlib import contextmanager

    @contextmanager
    def fake_open_retrieval_stores(cfg: object) -> object:
        del cfg
        yield stores_mock

    monkeypatch.setattr(_query_pipeline, "open_retrieval_stores", fake_open_retrieval_stores)

    from lxd.settings.models import EmbeddingConfig, RetrievalConfig

    cfg = SimpleNamespace(
        embedding=EmbeddingConfig(timeout_secs=30.0, retry_attempts=3),
        retrieval=RetrievalConfig(
            multi_query_enabled=True, hyde_enabled=False, dense_top_k=5, rerank_top_k=5
        ),
        expansion=SimpleNamespace(max_terms=5),
        knowledge_graph=SimpleNamespace(enabled=False),
        models=SimpleNamespace(embed_dims=8),
        paths=SimpleNamespace(data_path=Path("/tmp/data")),
        chunking=SimpleNamespace(contextual_summary_enabled=False),
    )

    outcome = _query_pipeline.search_chunks("test query", cast("RuntimeConfig", cfg))
    assert outcome.ranked
    assert len(embed_texts_calls) == 1
    assert embed_texts_calls[0] == ["alt 1", "alt 2"]
