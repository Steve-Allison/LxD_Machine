from types import SimpleNamespace
from typing import cast

import pytest

from lxd.ingest import embedder as embedder_module
from lxd.retrieval import dense
from lxd.settings.models import RuntimeConfig


def _install_ollama_client(monkeypatch: pytest.MonkeyPatch, client: object) -> None:
    def _factory(_config: RuntimeConfig) -> object:
        return client

    monkeypatch.setattr(embedder_module, "_ollama_client", _factory)


def test_embed_texts_sends_one_checked_request_per_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool, int]] = []

    class FakeClient:
        def embed(
            self, *, model: str, input: str, truncate: bool, dimensions: int
        ) -> dict[str, list[list[float]]]:
            del model
            calls.append((input, truncate, dimensions))
            return {"embeddings": [[float(len(calls))] * dimensions]}

    _install_ollama_client(monkeypatch, FakeClient())

    config = cast(
        "RuntimeConfig",
        SimpleNamespace(
            ollama=SimpleNamespace(url="http://localhost:11434"),
            models=SimpleNamespace(embed="nomic-embed-text", embed_dims=3, embed_backend="ollama"),
            embedding=None,
        ),
    )

    vectors = dense.embed_texts(config, ["aaaa", "bbbb"])

    assert calls == [("aaaa", False, 3), ("bbbb", False, 3)]
    assert vectors == [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]


def test_embed_query_applies_instruction_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    class FakeClient:
        def embed(
            self, *, model: str, input: str, truncate: bool, dimensions: int
        ) -> dict[str, list[list[float]]]:
            del model, truncate
            captured.append(input)
            return {"embeddings": [[0.0] * dimensions]}

    _install_ollama_client(monkeypatch, FakeClient())

    config = cast(
        "RuntimeConfig",
        SimpleNamespace(
            ollama=SimpleNamespace(url="http://localhost:11434"),
            models=SimpleNamespace(
                embed="qwen3-embedding:4b", embed_dims=3, embed_backend="ollama"
            ),
            embedding=SimpleNamespace(
                query_instruction="Instruct: Test.\nQuery: ",
                retry_attempts=1,
                retry_backoff=[],
                timeout_secs=30,
            ),
        ),
    )

    dense.embed_query(config, "what is Bloom's taxonomy?")

    assert captured == ["Instruct: Test.\nQuery: what is Bloom's taxonomy?"]


def test_embed_query_no_prefix_when_instruction_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    class FakeClient:
        def embed(
            self, *, model: str, input: str, truncate: bool, dimensions: int
        ) -> dict[str, list[list[float]]]:
            del model, truncate
            captured.append(input)
            return {"embeddings": [[0.0] * dimensions]}

    _install_ollama_client(monkeypatch, FakeClient())

    config = cast(
        "RuntimeConfig",
        SimpleNamespace(
            ollama=SimpleNamespace(url="http://localhost:11434"),
            models=SimpleNamespace(
                embed="text-embedding-3-small", embed_dims=3, embed_backend="ollama"
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

    assert captured == ["hello world"]
