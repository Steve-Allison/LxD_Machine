"""Pin retrieval limit contracts across pipeline, config, and MCP."""

import pytest
from pydantic import ValidationError

from lxd.domain.limits import MAX_RETRIEVAL_LIMIT
from lxd.retrieval import query_pipeline as qp
from lxd.settings.models import AdaptiveRetrievalConfig, RetrievalConfig

pytestmark = [pytest.mark.unit]


def test_max_retrieval_limit_is_single_source_of_truth() -> None:
    assert MAX_RETRIEVAL_LIMIT == 50
    assert qp._MAX_LIMIT == MAX_RETRIEVAL_LIMIT  # pyright: ignore[reportPrivateUsage]


def test_adaptive_and_retrieval_config_cannot_exceed_pipeline_cap() -> None:
    RetrievalConfig(dense_top_k=MAX_RETRIEVAL_LIMIT, rerank_top_k=MAX_RETRIEVAL_LIMIT)
    AdaptiveRetrievalConfig(
        narrow_dense_top_k=MAX_RETRIEVAL_LIMIT,
        broad_dense_top_k=MAX_RETRIEVAL_LIMIT,
    )

    with pytest.raises(ValidationError):
        RetrievalConfig(dense_top_k=MAX_RETRIEVAL_LIMIT + 1, rerank_top_k=10)
    with pytest.raises(ValidationError):
        AdaptiveRetrievalConfig(broad_dense_top_k=MAX_RETRIEVAL_LIMIT + 1)
