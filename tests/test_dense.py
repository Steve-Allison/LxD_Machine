from __future__ import annotations

from types import SimpleNamespace

from lxd.retrieval import dense


def test_embed_texts_sends_one_checked_request_per_text(monkeypatch) -> None:
    calls: list[tuple[str, bool, int]] = []

    class FakeClient:
        def embed(self, *, model: str, input: str, truncate: bool, dimensions: int):
            calls.append((input, truncate, dimensions))
            return {"embeddings": [[float(len(calls))] * dimensions]}

    monkeypatch.setattr(dense, "_client", lambda config: FakeClient())

    config = SimpleNamespace(
        ollama=SimpleNamespace(url="http://localhost:11434"),
        models=SimpleNamespace(embed="nomic-embed-text", embed_dims=3),
    )

    vectors = dense.embed_texts(config, ["aaaa", "bbbb"])

    assert calls == [("aaaa", False, 3), ("bbbb", False, 3)]
    assert vectors == [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]
