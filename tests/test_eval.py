from __future__ import annotations

import pytest

from lxd.retrieval.eval import _normalize_expected


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
