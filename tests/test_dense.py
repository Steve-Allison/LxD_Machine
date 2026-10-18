from types import SimpleNamespace
from typing import Any, cast

import pytest

from lxd.ingest import embedder as embedder_module
from lxd.retrieval import dense
from lxd.settings.models import RuntimeConfig


def test_embed_texts_sends_one_checked_request_per_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def _fake_embed(_cfg: Any, texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(len(calls))] * 3 for _ in texts]

    monkeypatch.setattr(embedder_module, "embed_texts_http", _fake_embed)

    config = cast(
        "RuntimeConfig",
        SimpleNamespace(
            models=SimpleNamespace(embed="test-model", embed_dims=3, embed_backend="mlx"),
            embedding=None,
        ),
    )

    vectors = dense.embed_texts(config, ["aaaa", "bbbb"])

    assert calls == [["aaaa", "bbbb"]]
    assert vectors == [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]


def test_embed_query_applies_instruction_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def _fake_embed(_cfg: Any, texts: list[str]) -> list[list[float]]:
        captured.append(list(texts))
        return [[0.0] * 3 for _ in texts]

    monkeypatch.setattr(embedder_module, "embed_texts_http", _fake_embed)

    config = cast(
        "RuntimeConfig",
        SimpleNamespace(
            models=SimpleNamespace(embed="test-model", embed_dims=3, embed_backend="mlx"),
            embedding=SimpleNamespace(
                query_instruction="Instruct: Test.\nQuery: ",
                retry_attempts=1,
                retry_backoff=[],
                timeout_secs=30,
            ),
        ),
    )

    dense.embed_query(config, "what is Bloom's taxonomy?")

    assert captured == [["Instruct: Test.\nQuery: what is Bloom's taxonomy?"]]


def test_embed_query_no_prefix_when_instruction_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def _fake_embed(_cfg: Any, texts: list[str]) -> list[list[float]]:
        captured.append(list(texts))
        return [[0.0] * 3 for _ in texts]

    monkeypatch.setattr(embedder_module, "embed_texts_http", _fake_embed)

    config = cast(
        "RuntimeConfig",
        SimpleNamespace(
            models=SimpleNamespace(
                embed="text-embedding-3-small", embed_dims=3, embed_backend="mlx"
            ),
            embedding=SimpleNamespace(
                query_instruction=None,
                retry_attempts=1,
                retry_backoff=[],
                timeout_secs=30,
            ),
        ),
    )

    dense.embed_query(config, "hello world")

    assert captured == [["hello world"]]
