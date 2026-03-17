from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import ollama

from lxd.settings.models import RuntimeConfig


@dataclass(frozen=True)
class ModelProbeResult:
    ok: bool
    warning: str | None = None


@dataclass(frozen=True)
class _EmbeddingRuntimeSettings:
    timeout_secs: int = 120
    retry_attempts: int = 1
    retry_backoff: tuple[int, ...] = ()


class EmbeddingContextError(RuntimeError):
    pass


def probe_embedder(config: RuntimeConfig) -> ModelProbeResult:
    try:
        embeddings = embed_texts(config, ["lxd ingest embed probe"])
    except (EmbeddingContextError, ImportError, OSError, RuntimeError, ValueError) as exc:
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
    if config.models.embed_backend == "openai":
        return _openai_embed_texts(config, texts)
    return _ollama_embed_texts(config, texts)


def embed_chunk_text(config: RuntimeConfig, text: str) -> list[float]:
    return embed_texts(config, [text])[0]


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _ollama_embed_texts(config: RuntimeConfig, texts: list[str]) -> list[list[float]]:
    return [_ollama_embed_single(config, text) for text in texts]


def _ollama_embed_single(config: RuntimeConfig, text: str) -> list[float]:
    runtime = _embedding_runtime_settings(config)
    attempts = runtime.retry_attempts
    backoff = runtime.retry_backoff
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = _ollama_client(config).embed(
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


def _ollama_client(config: RuntimeConfig) -> ollama.Client:
    runtime = _embedding_runtime_settings(config)
    return ollama.Client(host=str(config.ollama.url), timeout=float(runtime.timeout_secs))


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _openai_embed_texts(config: RuntimeConfig, texts: list[str]) -> list[list[float]]:
    import os

    import openai as _openai  # lazy import — only needed for openai backend

    cfg = config.openai
    if cfg is None:
        raise RuntimeError("openai config required when embed_backend=openai")
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {cfg.api_key_env!r} is not set. "
            "Set it before using the openai embedding backend."
        )
    client = _openai.OpenAI(api_key=api_key)
    batches = [texts[i : i + cfg.batch_size] for i in range(0, len(texts), cfg.batch_size)]
    results: list[list[list[float]] | None] = [None] * len(batches)

    def _embed_batch(idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
        response = client.embeddings.create(
            model=cfg.model,
            input=batch,
            dimensions=cfg.dims,
        )
        return idx, [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {
            executor.submit(_embed_batch, i, batch): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            idx, vectors = future.result()
            results[idx] = vectors

    flat: list[list[float]] = []
    for batch_result in results:
        assert batch_result is not None
        flat.extend(batch_result)
    return flat


def _embedding_runtime_settings(config: RuntimeConfig) -> _EmbeddingRuntimeSettings:
    embedding_config = getattr(config, "embedding", None)
    if embedding_config is None:
        return _EmbeddingRuntimeSettings()
    return _EmbeddingRuntimeSettings(
        timeout_secs=int(embedding_config.timeout_secs),
        retry_attempts=int(embedding_config.retry_attempts),
        retry_backoff=tuple(int(value) for value in embedding_config.retry_backoff),
    )
