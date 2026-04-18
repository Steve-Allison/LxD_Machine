"""Probe embedding backends and encode corpus or query text.

Responsibility:
    Front the Ollama and OpenAI embedding providers behind a uniform
    ``embed_texts`` / ``embed_chunk_text`` / ``embed_texts_batched`` API, and
    surface :class:`EmbeddingContextError` so the pipeline can split chunks
    that overflow the model's context window.

Design boundary:
    Ingest and retrieval both call into this module; they must remain
    ignorant of the underlying HTTP client, retry policy, and batching
    strategy.

Key constraints:
    * Ollama batches are sent in a single HTTP call using
      ``input=list[str]`` when possible, and fall back to per-text requests
      on context-window errors so a single oversized text cannot fail an
      entire batch silently.
    * OpenAI batches respect ``OpenAIEmbeddingConfig.batch_size`` and are
      issued concurrently via a thread pool bounded by ``max_workers``.
    * Retries are bounded by ``EmbeddingConfig.retry_attempts`` and
      ``retry_backoff`` (seconds, element-wise indexed by attempt).
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import ollama
import openai

from lxd.settings.models import RuntimeConfig


@dataclass(frozen=True)
class ModelProbeResult:
    """Embedding probe status and optional warning."""

    ok: bool
    warning: str | None = None


@dataclass(frozen=True)
class _EmbeddingRuntimeSettings:
    timeout_secs: int = 120
    retry_attempts: int = 1
    retry_backoff: tuple[int, ...] = ()
    batch_size: int = 32
    max_workers: int = 4


class EmbeddingContextError(RuntimeError):
    """Raised when input text exceeds embedding model context limits."""

    pass


def probe_embedder(config: RuntimeConfig) -> ModelProbeResult:
    """Probe backend availability and return probe metadata.

    Args:
        config: Runtime configuration object.

    Returns:
        Embedding probe status.
    """
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
    """Embed a list of input texts sequentially.

    Delegates to the provider-specific implementation. For large batches
    prefer :func:`embed_texts_batched`, which groups requests by
    ``batch_size`` and parallelises them where the backend supports it.

    Args:
        config: Runtime configuration object.
        texts: Texts to embed.

    Returns:
        One embedding vector per input text, in input order.
    """
    if config.models.embed_backend == "openai":
        return _openai_embed_texts(config, texts)
    return _ollama_embed_texts(config, texts)


def embed_chunk_text(config: RuntimeConfig, text: str) -> list[float]:
    """Embed a single text.

    Args:
        config: Runtime configuration object.
        text: Input text to embed.

    Returns:
        Embedding vector for the input text.

    Raises:
        EmbeddingContextError: If the text exceeds the model's context
            window and should be split before retrying.
    """
    return embed_texts(config, [text])[0]


def embed_texts_batched(
    config: RuntimeConfig,
    texts: Sequence[str],
) -> list[list[float]]:
    """Embed ``texts`` using the backend's native batch API.

    Args:
        config: Runtime configuration object.
        texts: Texts to embed, in order.

    Returns:
        One embedding vector per input text, in the same order as ``texts``.

    Raises:
        EmbeddingContextError: If any text exceeds the model's context
            window. The batch is retried per-text to isolate the offending
            entries; the first oversized text triggers the exception.
    """
    if not texts:
        return []
    if config.models.embed_backend == "openai":
        return _openai_embed_texts(config, list(texts))
    return _ollama_embed_batch(config, list(texts))


def _ollama_embed_texts(config: RuntimeConfig, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts via the Ollama backend sequentially."""
    return [_ollama_embed_single(config, text) for text in texts]


def _ollama_embed_batch(config: RuntimeConfig, texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` through Ollama in one batch, falling back per-text.

    Ollama's ``embed`` accepts ``input: Sequence[str]``; on a 200 response
    the returned ``embeddings`` list is ordered to match ``input``. If the
    call fails with a context-length error we cannot tell which text caused
    it, so we fall back to sequential single embeds and let the first
    offender surface :class:`EmbeddingContextError` to the caller.
    """
    runtime = _embedding_runtime_settings(config)
    batch_size = max(1, runtime.batch_size)
    max_workers = max(1, runtime.max_workers)

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    if not batches:
        return []

    if len(batches) == 1 or max_workers == 1:
        return [
            vector
            for batch in batches
            for vector in _ollama_embed_one_batch(config, batch, runtime)
        ]

    results: list[list[list[float]] | None] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_ollama_embed_one_batch, config, batch, runtime): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    flat: list[list[float]] = []
    for batch_result in results:
        if batch_result is None:
            raise RuntimeError("Ollama embedding batch returned no results")
        flat.extend(batch_result)
    return flat


def _ollama_embed_one_batch(
    config: RuntimeConfig,
    texts: list[str],
    runtime: _EmbeddingRuntimeSettings,
) -> list[list[float]]:
    """Send one Ollama ``embed`` call; fall back per-text on context errors."""
    attempts = max(1, runtime.retry_attempts)
    backoff = runtime.retry_backoff
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = _ollama_client(config).embed(
                model=config.models.embed,
                input=texts,
                truncate=False,
                dimensions=config.models.embed_dims,
            )
            embeddings = response["embeddings"]
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Ollama embed returned {len(embeddings)} vectors for {len(texts)} inputs"
                )
            return [[float(value) for value in vector] for vector in embeddings]
        except ollama.ResponseError as exc:
            if "input length exceeds the context length" in str(exc):
                return [_ollama_embed_single(config, text) for text in texts]
            last_error = exc
        except ollama.RequestError as exc:
            last_error = exc
        if attempt < attempts - 1:
            time.sleep(float(backoff[min(attempt, len(backoff) - 1)]) if backoff else 0.0)
    assert last_error is not None
    raise last_error


def _ollama_embed_single(config: RuntimeConfig, text: str) -> list[float]:
    """Embed one text via Ollama with retry on transient errors."""
    runtime = _embedding_runtime_settings(config)
    attempts = max(1, runtime.retry_attempts)
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


def _openai_embed_texts(config: RuntimeConfig, texts: list[str]) -> list[list[float]]:
    """Embed texts via OpenAI with per-batch concurrency."""
    cfg = config.openai
    if cfg is None:
        raise RuntimeError("openai config required when embed_backend=openai")
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {cfg.api_key_env!r} is not set. "
            "Set it before using the openai embedding backend."
        )
    client = openai.OpenAI(api_key=api_key)
    batches = [texts[i : i + cfg.batch_size] for i in range(0, len(texts), cfg.batch_size)]
    results: list[list[list[float]] | None] = [None] * len(batches)

    def _embed_batch(idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
        response = client.embeddings.create(
            model=cfg.model,
            input=batch,
            dimensions=cfg.dims,
        )
        return idx, [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {executor.submit(_embed_batch, i, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            idx, vectors = future.result()
            results[idx] = vectors

    if any(batch_result is None for batch_result in results):
        raise RuntimeError("OpenAI embedding results were incomplete.")
    flat: list[list[float]] = []
    for batch_result in results:
        if batch_result is not None:
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
        batch_size=int(getattr(embedding_config, "batch_size", 32)),
        max_workers=int(getattr(embedding_config, "max_workers", 4)),
    )
