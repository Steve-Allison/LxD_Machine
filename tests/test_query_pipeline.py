from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from lxd.app.status import current_ingest_config
from lxd.retrieval import query_pipeline as _query_pipeline
from lxd.retrieval.query_pipeline import RankedChunk
from lxd.settings.models import RuntimeConfig

# Private helpers exercised intentionally — same logical unit as
# query_pipeline; the tests below regress its retrieval shape.
_diversify_by_community = _query_pipeline._diversify_by_community  # pyright: ignore[reportPrivateUsage]
_fuse_ranked_prefix = _query_pipeline._fuse_ranked_prefix  # pyright: ignore[reportPrivateUsage]
_merge_ranked_prefix = _query_pipeline._merge_ranked_prefix  # pyright: ignore[reportPrivateUsage]
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
        lexical_rank={},
        lexical_fusion_weight=0.0,
        relation_fusion_weight=0.0,
        relation_chunk_ids=set(),
        centrality_fusion_weight=0.0,
    )
    with_lane = _fuse_ranked_prefix(
        dense_prefix=dense_prefix,
        reranked_prefix=[],
        lexical_rank={},
        lexical_fusion_weight=0.0,
        relation_fusion_weight=0.0,
        relation_chunk_ids=set(),
        centrality_fusion_weight=10.0,
    )
    # Without the lane: dense order preserved (low first).
    assert [item.chunk_id for item in no_lane] == ["low", "high"]
    # With a strong lane: high-PR chunk surfaces.
    assert [item.chunk_id for item in with_lane] == ["high", "low"]


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
            embed_backend="ollama",
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
