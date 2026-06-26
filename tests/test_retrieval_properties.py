"""Property-based tests for retrieval-fusion and diversification invariants.

`hypothesis` is already a project dep. These tests target invariants
that are easy to break inadvertently when the fusion code is touched
(e.g. monotonicity of `_rrf_score`, idempotence of
`_unique_source_prefix`, completeness of `_diversify_by_community`).
"""

import math
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from lxd.retrieval import query_pipeline as _query_pipeline
from lxd.retrieval.query_pipeline import RankedChunk

# Private helpers exercised by property tests for the same module's
# fusion / diversification invariants.
_diversify_by_community = _query_pipeline._diversify_by_community  # pyright: ignore[reportPrivateUsage]
_fuse_ranked_prefix = _query_pipeline._fuse_ranked_prefix  # pyright: ignore[reportPrivateUsage]
_merge_ranked_prefix = _query_pipeline._merge_ranked_prefix  # pyright: ignore[reportPrivateUsage]
_rrf_score = _query_pipeline._rrf_score  # pyright: ignore[reportPrivateUsage]
_unique_source_prefix = _query_pipeline._unique_source_prefix  # pyright: ignore[reportPrivateUsage]


def _chunk(
    chunk_id: str,
    *,
    source_rel_path: str | None = None,
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
        score_hint=chunk_id,
        metadata_json="{}",
        score=0.0,
        central_entity_score=central_entity_score,
        community_ids=community_ids,
    )


# ---------------------------------------------------------------------------
# _rrf_score: monotonic-decreasing in rank, strictly positive
# ---------------------------------------------------------------------------


@given(rank=st.integers(min_value=1, max_value=10_000))
def test_rrf_score_is_strictly_positive(rank: int) -> None:
    assert _rrf_score(rank) > 0.0


@given(
    rank_a=st.integers(min_value=1, max_value=10_000),
    rank_b=st.integers(min_value=1, max_value=10_000),
)
def test_rrf_score_is_monotonic_decreasing(rank_a: int, rank_b: int) -> None:
    """A better (lower) rank yields a higher RRF score. Equal ranks tie."""
    if rank_a < rank_b:
        assert _rrf_score(rank_a) > _rrf_score(rank_b)
    elif rank_a > rank_b:
        assert _rrf_score(rank_a) < _rrf_score(rank_b)
    else:
        assert _rrf_score(rank_a) == _rrf_score(rank_b)


@given(rank=st.integers(min_value=1, max_value=10_000))
def test_rrf_score_is_finite(rank: int) -> None:
    """No infinity, no NaN — fusion sums must not break on extreme ranks."""
    score = _rrf_score(rank)
    assert math.isfinite(score)


# ---------------------------------------------------------------------------
# _unique_source_prefix: at most ``limit`` items, all sources distinct,
# stable order, no fabricated chunks
# ---------------------------------------------------------------------------


@st.composite
def _ranked_with_sources(draw: st.DrawFn) -> list[RankedChunk]:
    n = draw(st.integers(min_value=0, max_value=20))
    return [
        _chunk(f"c{i}", source_rel_path=f"src{draw(st.integers(min_value=0, max_value=5))}.md")
        for i in range(n)
    ]


@given(ranked=_ranked_with_sources(), limit=st.integers(min_value=0, max_value=20))
def test_unique_source_prefix_yields_distinct_sources(
    ranked: list[RankedChunk], limit: int
) -> None:
    prefix = _unique_source_prefix(ranked, limit)
    paths = [item.source_rel_path for item in prefix]
    assert len(paths) == len(set(paths)), "Duplicate sources in prefix."
    assert len(prefix) <= limit, "Prefix exceeded the requested limit."


@given(ranked=_ranked_with_sources(), limit=st.integers(min_value=0, max_value=20))
def test_unique_source_prefix_preserves_input_order(ranked: list[RankedChunk], limit: int) -> None:
    """The first occurrence of each source wins, in input order."""
    prefix = _unique_source_prefix(ranked, limit)
    if limit <= 0:
        assert prefix == []
        return
    seen: set[str] = set()
    expected: list[str] = []
    for item in ranked:
        if item.source_rel_path in seen:
            continue
        seen.add(item.source_rel_path)
        expected.append(item.chunk_id)
        if len(expected) >= limit:
            break
    assert [item.chunk_id for item in prefix] == expected


# ---------------------------------------------------------------------------
# _merge_ranked_prefix: prefix on the front, dense tail follows, no duplicates
# ---------------------------------------------------------------------------


@st.composite
def _ranked_then_prefix(draw: st.DrawFn) -> tuple[list[RankedChunk], list[RankedChunk]]:
    ranked = [_chunk(f"c{i}") for i in range(draw(st.integers(min_value=0, max_value=10)))]
    prefix_indices = draw(
        st.lists(st.integers(min_value=0, max_value=max(0, len(ranked) - 1)), unique=True)
    )
    prefix = [ranked[i] for i in prefix_indices] if ranked else []
    return ranked, prefix


@given(_ranked_then_prefix())
def test_merge_ranked_prefix_has_no_duplicate_chunk_ids(
    pair: tuple[list[RankedChunk], list[RankedChunk]],
) -> None:
    ranked, prefix = pair
    merged = _merge_ranked_prefix(ranked, prefix)
    chunk_ids = [item.chunk_id for item in merged]
    assert len(chunk_ids) == len(set(chunk_ids))


@given(_ranked_then_prefix())
def test_merge_ranked_prefix_starts_with_prefix(
    pair: tuple[list[RankedChunk], list[RankedChunk]],
) -> None:
    ranked, prefix = pair
    merged = _merge_ranked_prefix(ranked, prefix)
    assert [item.chunk_id for item in merged[: len(prefix)]] == [item.chunk_id for item in prefix]


# ---------------------------------------------------------------------------
# _fuse_ranked_prefix: never adds or drops chunks; output is a permutation
# ---------------------------------------------------------------------------


@st.composite
def _fusion_inputs(
    draw: st.DrawFn,
) -> tuple[list[RankedChunk], list[RankedChunk], dict[str, int], float]:
    n = draw(st.integers(min_value=0, max_value=10))
    dense = [
        _chunk(
            f"c{i}",
            central_entity_score=draw(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
            ),
        )
        for i in range(n)
    ]
    rerank_size = draw(st.integers(min_value=0, max_value=n))
    rerank_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=max(0, n - 1)),
            unique=True,
            min_size=rerank_size,
            max_size=rerank_size,
        )
    )
    reranked = [dense[i] for i in rerank_indices] if dense else []
    lexical_rank = {
        chunk.chunk_id: rank for rank, chunk in enumerate(dense, start=1) if draw(st.booleans())
    }
    weight = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    return dense, reranked, lexical_rank, weight


@given(_fusion_inputs())
@settings(max_examples=80, deadline=None)
def test_fuse_ranked_prefix_output_is_a_permutation_of_dense_input(
    inputs: tuple[list[RankedChunk], list[RankedChunk], dict[str, int], float],
) -> None:
    dense, reranked, lexical_rank, weight = inputs
    fused = _fuse_ranked_prefix(
        dense_prefix=dense,
        reranked_prefix=reranked,
        lexical_rank=lexical_rank,
        lexical_fusion_weight=weight,
        relation_fusion_weight=weight,
        relation_chunk_ids=set(),
        centrality_fusion_weight=weight,
    )
    assert sorted(item.chunk_id for item in fused) == sorted(item.chunk_id for item in dense)


# ---------------------------------------------------------------------------
# _diversify_by_community: total chunks preserved (per `limit`), distinct
# communities first, untagged chunks last
# ---------------------------------------------------------------------------


@st.composite
def _diversify_inputs(draw: st.DrawFn) -> tuple[list[RankedChunk], int]:
    n = draw(st.integers(min_value=0, max_value=12))
    chunks: list[RankedChunk] = []
    for i in range(n):
        community_ids = tuple(
            sorted(
                draw(
                    st.lists(
                        st.integers(min_value=0, max_value=4),
                        unique=True,
                        max_size=2,
                    )
                )
            )
        )
        chunks.append(_chunk(f"c{i}", community_ids=community_ids))
    limit = draw(st.integers(min_value=0, max_value=n))
    return chunks, limit


@given(_diversify_inputs())
def test_diversify_by_community_respects_limit(
    inputs: tuple[list[RankedChunk], int],
) -> None:
    ranked, limit = inputs
    diversified = _diversify_by_community(ranked, limit)
    assert len(diversified) <= limit


@given(_diversify_inputs())
def test_diversify_by_community_preserves_chunk_set_under_full_limit(
    inputs: tuple[list[RankedChunk], int],
) -> None:
    """Asking for the full length yields every chunk back (no fabrication, no drop)."""
    ranked, _limit = inputs
    diversified = _diversify_by_community(ranked, len(ranked))
    assert sorted(item.chunk_id for item in diversified) == sorted(item.chunk_id for item in ranked)


@given(_diversify_inputs())
def test_diversify_by_community_places_untagged_after_tagged(
    inputs: tuple[list[RankedChunk], int],
) -> None:
    """Once an untagged chunk appears in the output, every later chunk is
    also untagged (or the output is shorter than the tagged set)."""
    ranked, _ = inputs
    diversified = _diversify_by_community(ranked, len(ranked))
    seen_untagged = False
    for item in diversified:
        if not item.community_ids:
            seen_untagged = True
        elif seen_untagged:
            raise AssertionError(f"Tagged chunk {item.chunk_id} appeared after an untagged chunk.")
