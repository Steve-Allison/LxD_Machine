"""Regression tests for the batched embedding path introduced in Wave 5."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from lxd.ingest import embedder
from lxd.settings.models import RuntimeConfig


@dataclass
class _FakeEmbedding:
    embed_backend: str = "ollama"
    embed: str = "test-model"
    embed_dims: int = 3


@dataclass
class _FakeOllama:
    url: str = "http://localhost:11434/"


@dataclass
class _FakeEmbeddingConfig:
    timeout_secs: int = 30
    retry_attempts: int = 1
    retry_backoff: list[int] = field(default_factory=list)
    batch_size: int = 4
    max_workers: int = 1


@dataclass
class _FakeConfig:
    models: _FakeEmbedding = field(default_factory=_FakeEmbedding)
    ollama: _FakeOllama = field(default_factory=_FakeOllama)
    embedding: _FakeEmbeddingConfig = field(default_factory=_FakeEmbeddingConfig)
    openai: object | None = None


class _StubClient:
    """Mimics the tiny slice of ``ollama.Client`` the embedder uses."""

    def __init__(self, responses: list[list[list[float]]]) -> None:
        self._responses = list(responses)
        self.calls: list[object] = []

    def embed(
        self,
        *,
        model: str,
        input: object,
        truncate: bool,
        dimensions: int,
    ) -> dict[str, list[list[float]]]:
        self.calls.append(input)
        return {"embeddings": self._responses.pop(0)}


def test_embed_texts_batched_sends_single_ollama_call() -> None:
    """A batch smaller than ``batch_size`` is sent in one HTTP request."""

    config = _FakeConfig()
    texts = ["a", "b", "c"]
    expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    stub = _StubClient([expected])

    with patch.object(embedder, "_ollama_client", return_value=stub):
        vectors = embedder.embed_texts_batched(cast("RuntimeConfig", config), texts)

    assert vectors == expected
    assert len(stub.calls) == 1
    assert stub.calls[0] == texts


def test_embed_texts_batched_splits_by_batch_size() -> None:
    """Inputs longer than ``batch_size`` issue multiple Ollama requests."""

    config = _FakeConfig(embedding=_FakeEmbeddingConfig(batch_size=2))
    texts = ["a", "b", "c", "d", "e"]
    batches = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        [[1.0, 1.0, 1.0]],
    ]
    stub = _StubClient(batches)

    with patch.object(embedder, "_ollama_client", return_value=stub):
        vectors = embedder.embed_texts_batched(cast("RuntimeConfig", config), texts)

    assert vectors == [vec for batch in batches for vec in batch]
    assert [list(cast("list[str]", call)) for call in stub.calls] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_embed_texts_batched_falls_back_on_context_error() -> None:
    """A context-length failure forces per-text fallback for that batch."""

    config = _FakeConfig(embedding=_FakeEmbeddingConfig(batch_size=4))
    texts = ["short", "toolong"]

    call_log: list[object] = []

    def _fake_embed(**kwargs: object) -> dict[str, object]:
        call_log.append(kwargs["input"])
        payload = kwargs["input"]
        if isinstance(payload, list):
            raise embedder.ollama.ResponseError("input length exceeds the context length", 500)
        if payload == "toolong":
            raise embedder.ollama.ResponseError("input length exceeds the context length", 500)
        return {"embeddings": [[0.0, 0.0, 1.0]]}

    fake_client = SimpleNamespace(embed=_fake_embed)

    with (
        patch.object(embedder, "_ollama_client", return_value=fake_client),
        pytest.raises(embedder.EmbeddingContextError),
    ):
        embedder.embed_texts_batched(cast("RuntimeConfig", config), texts)

    assert call_log[0] == texts
    assert call_log[1] == "short"
    assert call_log[2] == "toolong"
