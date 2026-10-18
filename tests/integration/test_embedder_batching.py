"""Regression tests for the batched embedding path."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from lxd.ingest import embedder
from lxd.settings.models import RuntimeConfig


@dataclass
class _FakeEmbedding:
    embed_backend: str = "mlx"
    embed: str = "test-model"
    embed_dims: int = 3


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
    embedding: _FakeEmbeddingConfig = field(default_factory=_FakeEmbeddingConfig)
    openai: object | None = None
    mlx: object | None = None


def test_embed_texts_batched_empty() -> None:
    """Empty inputs return empty list without calling backends."""
    config = _FakeConfig()
    result = embedder.embed_texts_batched(cast("RuntimeConfig", config), [])
    assert result == []


def test_embed_texts_batched_delegates_to_mlx() -> None:
    """When embed_backend='mlx', embed_texts_http is called."""
    config = _FakeConfig(models=_FakeEmbedding(embed_backend="mlx"))
    texts = ["hello", "world"]
    expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    with patch("lxd.ingest.embedder.embed_texts_http", return_value=expected) as mock_http:
        vectors = embedder.embed_texts_batched(cast("RuntimeConfig", config), texts)

    assert vectors == expected
    mock_http.assert_called_once_with(config, texts)


def test_embed_texts_batched_delegates_to_openai() -> None:
    """When embed_backend='openai', _openai_embed_texts is called."""
    config = _FakeConfig(
        models=_FakeEmbedding(embed_backend="openai"),
        openai=SimpleNamespace(
            api_key_env="TEST_OPENAI_KEY",
            model="text-embedding-3-small",
            dims=3,
            batch_size=2,
            max_workers=1,
        ),
    )
    texts = ["a", "b", "c"]
    expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = [
        SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=[0.1, 0.2, 0.3]),
                SimpleNamespace(index=1, embedding=[0.4, 0.5, 0.6]),
            ]
        ),
        SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=[0.7, 0.8, 0.9]),
            ]
        ),
    ]

    with (
        patch.dict("os.environ", {"TEST_OPENAI_KEY": "test-key"}),
        patch("lxd.ingest.embedder.get_openai_client", return_value=mock_client),
    ):
        vectors = embedder.embed_texts_batched(cast("RuntimeConfig", config), texts)

    assert vectors == expected
    assert mock_client.embeddings.create.call_count == 2
