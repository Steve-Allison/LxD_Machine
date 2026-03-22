from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from lxd.retrieval import rerank


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    document_id: str
    citation_label: str
    source_rel_path: str
    source_path: str
    source_filename: str
    source_type: str
    source_domain: str
    source_hash: str
    chunk_index: int
    chunk_occurrence: int
    token_count: int
    text: str
    score_hint: str
    metadata_json: str
    score: float


def test_rerank_chunks_uses_llama_cpp_backend(monkeypatch) -> None:
    rerank._probe_cache.clear()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code
            self.text = ""

        def json(self) -> dict[str, object]:
            return self._payload

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, path: str, json: dict[str, object]) -> FakeResponse:
            calls.append((path, json))
            if len(calls) == 1:
                return FakeResponse({"results": [{"index": 0, "relevance_score": 0.9}]})
            return FakeResponse(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.8},
                        {"index": 0, "relevance_score": 0.2},
                    ]
                }
            )

    monkeypatch.setattr(rerank, "_client", lambda config: FakeClient())

    config = SimpleNamespace(
        models=SimpleNamespace(rerank="qwen3-reranker"),
        reranker=SimpleNamespace(
            enabled=True,
            backend="llama_cpp",
            url="http://127.0.0.1:8012",
            endpoint="/v1/rerank",
            timeout_secs=30,
            launch=SimpleNamespace(
                auto_start=False,
                executable="llama-server",
                model_source="ollama_blob",
                model_path=None,
                host="127.0.0.1",
                port=8012,
                startup_timeout_secs=30,
                extra_args=[],
            ),
        ),
        paths=SimpleNamespace(data_path=Path("/tmp")),
    )
    candidates = [
        Candidate(
            chunk_id="a",
            document_id="doc",
            citation_label="A",
            source_rel_path="a.md",
            source_path="/tmp/a.md",
            source_filename="a.md",
            source_type="markdown",
            source_domain="guides",
            source_hash="hash-a",
            chunk_index=0,
            chunk_occurrence=0,
            token_count=10,
            text="first text",
            score_hint="first",
            metadata_json="{}",
            score=0.1,
        ),
        Candidate(
            chunk_id="b",
            document_id="doc",
            citation_label="B",
            source_rel_path="b.md",
            source_path="/tmp/b.md",
            source_filename="b.md",
            source_type="markdown",
            source_domain="guides",
            source_hash="hash-b",
            chunk_index=1,
            chunk_occurrence=0,
            token_count=10,
            text="second text",
            score_hint="second",
            metadata_json="{}",
            score=0.05,
        ),
    ]

    outcome = rerank.rerank_chunks("question", candidates, config)

    assert outcome.applied is True
    assert [item.chunk_id for item in outcome.ranked] == ["b", "a"]
    assert calls[0][0] == "/v1/rerank"
    assert calls[1][0] == "/v1/rerank"


def test_rerank_chunks_falls_back_when_backend_disabled() -> None:
    rerank._probe_cache.clear()
    config = SimpleNamespace(
        models=SimpleNamespace(rerank="qwen3-reranker"),
        reranker=SimpleNamespace(
            enabled=False,
            backend="none",
            url=None,
            endpoint="/v1/rerank",
            timeout_secs=30,
            launch=None,
        ),
        paths=SimpleNamespace(data_path=Path("/tmp")),
    )
    candidates = [
        Candidate(
            chunk_id="a",
            document_id="doc",
            citation_label="A",
            source_rel_path="a.md",
            source_path="/tmp/a.md",
            source_filename="a.md",
            source_type="markdown",
            source_domain="guides",
            source_hash="hash-a",
            chunk_index=0,
            chunk_occurrence=0,
            token_count=10,
            text="first text",
            score_hint="first",
            metadata_json="{}",
            score=0.1,
        )
    ]

    outcome = rerank.rerank_chunks("question", candidates, config)

    assert outcome.applied is False
    assert outcome.ranked == candidates
    assert outcome.warnings == ["Configured reranker backend is disabled."]


def test_probe_reranker_autostarts_llama_server_from_ollama_blob(
    monkeypatch, tmp_path
) -> None:
    rerank._probe_cache.clear()
    commands: list[list[str]] = []
    probe_results = iter(
        [
            (False, "[Errno 61] Connection refused"),
            (False, "[Errno 61] Connection refused"),
            (True, None),
            (True, None),
        ]
    )

    class FakeProcess:
        pid = 43210

        def poll(self) -> None:
            return None

    def fake_probe(config) -> tuple[bool, str | None]:
        return next(probe_results)

    def fake_popen(command, **kwargs) -> FakeProcess:
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(rerank, "_probe_reranker_http", fake_probe)
    monkeypatch.setattr(rerank, "_resolve_llama_server_executable", lambda executable: executable)
    monkeypatch.setattr(
        rerank,
        "_resolve_ollama_blob_model_path",
        lambda model_name: Path("/tmp/local-reranker.gguf"),
    )
    monkeypatch.setattr(rerank.subprocess, "Popen", fake_popen)

    config = SimpleNamespace(
        models=SimpleNamespace(rerank="dengcao/Qwen3-Reranker-0.6B:F16"),
        reranker=SimpleNamespace(
            enabled=True,
            backend="llama_cpp",
            url="http://127.0.0.1:8012",
            endpoint="/v1/rerank",
            timeout_secs=30,
            launch=SimpleNamespace(
                auto_start=True,
                executable="llama-server",
                model_source="ollama_blob",
                model_path=None,
                host="127.0.0.1",
                port=8012,
                startup_timeout_secs=2,
                extra_args=["--threads", "4"],
            ),
        ),
        paths=SimpleNamespace(data_path=tmp_path),
    )

    supported, warning = rerank.probe_reranker(config)

    assert supported is True
    assert warning is None
    assert commands == [
        [
            "llama-server",
            "--model",
            "/tmp/local-reranker.gguf",
            "--embedding",
            "--pooling",
            "rank",
            "--alias",
            "dengcao/Qwen3-Reranker-0.6B:F16",
            "--host",
            "127.0.0.1",
            "--port",
            "8012",
            "--reranking",
            "--threads",
            "4",
        ]
    ]
    pid_files = list((tmp_path / "runtime").glob("reranker-llama-server-*.json"))
    assert len(pid_files) == 1


def test_apply_rerank_payload_orders_by_descending_relevance() -> None:
    candidates = [
        Candidate(
            chunk_id="a",
            document_id="doc",
            citation_label="A",
            source_rel_path="a.md",
            source_path="/tmp/a.md",
            source_filename="a.md",
            source_type="markdown",
            source_domain="guides",
            source_hash="hash-a",
            chunk_index=0,
            chunk_occurrence=0,
            token_count=10,
            text="alpha",
            score_hint="alpha",
            metadata_json="{}",
            score=0.1,
        ),
        Candidate(
            chunk_id="b",
            document_id="doc",
            citation_label="B",
            source_rel_path="b.md",
            source_path="/tmp/b.md",
            source_filename="b.md",
            source_type="markdown",
            source_domain="guides",
            source_hash="hash-b",
            chunk_index=1,
            chunk_occurrence=0,
            token_count=10,
            text="beta",
            score_hint="beta",
            metadata_json="{}",
            score=0.2,
        ),
        Candidate(
            chunk_id="c",
            document_id="doc",
            citation_label="C",
            source_rel_path="c.md",
            source_path="/tmp/c.md",
            source_filename="c.md",
            source_type="markdown",
            source_domain="guides",
            source_hash="hash-c",
            chunk_index=2,
            chunk_occurrence=0,
            token_count=10,
            text="gamma",
            score_hint="gamma",
            metadata_json="{}",
            score=0.3,
        ),
    ]

    reranked = rerank._apply_rerank_payload(
        candidates,
        {
            "results": [
                {"index": 2, "relevance_score": 0.1},
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.4},
            ]
        },
    )

    assert reranked is not None
    assert [item.chunk_id for item in reranked] == ["a", "b", "c"]
