from __future__ import annotations

import time
from dataclasses import dataclass

import ollama

from lxd.settings.models import RuntimeConfig


@dataclass(frozen=True)
class ModelProbeResult:
    ok: bool
    warning: str | None = None


class EmbeddingContextError(RuntimeError):
    pass


def probe_embedder(config: RuntimeConfig) -> ModelProbeResult:
    try:
        embeddings = embed_texts(config, ["lxd embed probe"])
    except (ollama.RequestError, ollama.ResponseError, EmbeddingContextError) as exc:
        return ModelProbeResult(ok=False, warning=str(exc))
    if not embeddings or len(embeddings[0]) != config.models.embed_dims:
        return ModelProbeResult(
            ok=False,
            warning=(
                f"Embedding probe returned {len(embeddings[0]) if embeddings else 0} dimensions; "
                f"expected {config.models.embed_dims}."
            ),
        )
    return ModelProbeResult(ok=True)


def embed_texts(config: RuntimeConfig, texts: list[str]) -> list[list[float]]:
    return [_embed_single_text(config, text) for text in texts]


def embed_query(config: RuntimeConfig, text: str) -> list[float]:
    return _embed_single_text(config, text)


def _embed_single_text(config: RuntimeConfig, text: str) -> list[float]:
    attempts = max(1, int(getattr(getattr(config, "embedding", None), "retry_attempts", 1)))
    backoff = list(getattr(getattr(config, "embedding", None), "retry_backoff", []))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = _client(config).embed(
                model=config.models.embed,
                input=text,
                truncate=False,
                dimensions=config.models.embed_dims,
            )
            if not response["embeddings"]:
                raise RuntimeError("Embedding response returned no vectors")
            return [float(value) for value in response["embeddings"][0]]
        except ollama.ResponseError as exc:
            if "input length exceeds the context length" in str(exc):
                raise EmbeddingContextError(str(exc)) from exc
            last_error = exc
        except ollama.RequestError as exc:
            last_error = exc
        if attempt < attempts - 1:
            time.sleep(float(backoff[min(attempt, len(backoff) - 1)]) if backoff else 0.0)
    assert last_error is not None
    raise last_error


def _client(config: RuntimeConfig) -> ollama.Client:
    timeout_secs = float(getattr(getattr(config, "embedding", None), "timeout_secs", 120))
    return ollama.Client(host=str(config.ollama.url), timeout=timeout_secs)
