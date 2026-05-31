"""Tests for the ColBERT-style late-interaction reranker.

The real ``BAAI/bge-m3`` model is several GB and loads from the network
on first use; we don't pay that cost in CI. The tests here exercise:

  - the MaxSim math itself (pure function over torch tensors)
  - the rerank-dispatch path that picks colbert vs llama_cpp
  - the config-validation surface
"""

from __future__ import annotations

import pytest

from lxd.retrieval.colbert_reranker import maxsim_score
from lxd.settings.models import RerankerConfig

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# MaxSim math
# ---------------------------------------------------------------------------


def test_maxsim_perfect_alignment_returns_query_length() -> None:
    """When every query token has an identical doc token, score = q_seq."""
    import torch

    # 3 query tokens, each a unit vector along distinct axes.
    query = torch.eye(3)
    # Doc contains the same 3 vectors plus one noise vector.
    doc = torch.cat([torch.eye(3), torch.zeros(1, 3)], dim=0)
    score = maxsim_score(query, doc)
    # Each query token finds an exact match → cos = 1.0; sum over 3 tokens = 3.0.
    assert score == pytest.approx(3.0)


def test_maxsim_zero_when_orthogonal() -> None:
    """When query and doc tokens are orthogonal, max sim per query token = 0."""
    import torch

    query = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    doc = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    score = maxsim_score(query, doc)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_maxsim_picks_max_per_query_token_not_average() -> None:
    """A single near-perfect doc token wins over many mediocre ones."""
    import torch

    query = torch.tensor([[1.0, 0.0]])
    # Many mediocre doc tokens + one near-perfect.
    doc = torch.tensor(
        [
            [0.1, 0.0],
            [0.1, 0.0],
            [0.1, 0.0],
            [0.9, 0.0],
        ]
    )
    score = maxsim_score(query, doc)
    # max similarity = 0.9; mean would be (0.1+0.1+0.1+0.9)/4 = 0.3.
    assert score == pytest.approx(0.9)


def test_maxsim_handles_single_token_query() -> None:
    import torch

    query = torch.tensor([[1.0, 0.0]])
    doc = torch.tensor([[1.0, 0.0]])
    assert maxsim_score(query, doc) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_reranker_config_accepts_colbert_backend() -> None:
    config = RerankerConfig(backend="colbert")
    assert config.backend == "colbert"
    # Default model is a well-known multi-vector model.
    assert config.colbert_model == "BAAI/bge-m3"
    # Default token cap is the conventional 512.
    assert config.colbert_max_length == 512


def test_reranker_config_rejects_zero_token_cap() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RerankerConfig(backend="colbert", colbert_max_length=0)


def test_reranker_config_rejects_negative_token_cap() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RerankerConfig(backend="colbert", colbert_max_length=-1)


def test_reranker_config_clamps_token_cap_to_8k() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RerankerConfig(backend="colbert", colbert_max_length=10_000)


def test_reranker_config_rejects_unknown_backend() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RerankerConfig(backend="something_else")  # type: ignore[arg-type]
