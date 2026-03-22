from __future__ import annotations

import random

import pytest

from lxd.retrieval.eval import _normalize_expected, mrr_at_k, recall_at_k


def test_normalize_expected_rejects_ambiguous_basenames() -> None:
    with pytest.raises(ValueError, match="Ambiguous eval basename"):
        _normalize_expected(
            ["README.md"],
            [
                "Guides/README.md",
                "Research/README.md",
            ],
        )


def test_normalize_expected_resolves_unique_basenames() -> None:
    normalized = _normalize_expected(
        ["guide.md", "Research/item.md"],
        [
            "Guides/guide.md",
            "Research/item.md",
        ],
    )

    assert normalized == ["Guides/guide.md", "Research/item.md"]


def test_recall_at_k_is_bounded_and_monotonic() -> None:
    rng = random.Random(1337)
    universe = [f"doc-{index}" for index in range(20)]

    for _ in range(50):
        ranked = rng.sample(universe, k=10)
        expected = set(rng.sample(universe, k=5))
        scores = [recall_at_k(expected, ranked, k) for k in range(1, 11)]

        assert all(0.0 <= score <= 1.0 for score in scores)
        assert scores == sorted(scores)


def test_mrr_at_k_is_bounded_and_monotonic() -> None:
    rng = random.Random(9001)
    universe = [f"doc-{index}" for index in range(20)]

    for _ in range(50):
        ranked = rng.sample(universe, k=10)
        expected = set(rng.sample(universe, k=4))
        scores = [mrr_at_k(expected, ranked, k) for k in range(1, 11)]

        assert all(0.0 <= score <= 1.0 for score in scores)
        assert scores == sorted(scores)
