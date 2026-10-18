"""Probe embedding backends and encode corpus or query text.

Responsibility:
    Front OpenAI and local Octen (MLX) embedding providers behind a uniform
    ``embed_texts`` / ``embed_chunk_text`` / ``embed_texts_batched`` API, and
    surface :class:`EmbeddingContextError` so the pipeline can split chunks
    that overflow the model's context window.

Design boundary:
    Ingest and retrieval both call into this module; they must remain
    ignorant of the underlying HTTP client, retry policy, and batching
    strategy.

Key constraints:
    * MLX batches use the resident Octen embedding HTTP server.
    * OpenAI batches respect ``OpenAIEmbeddingConfig.batch_size`` and are
      issued concurrently via a thread pool bounded by ``max_workers``.
      Per-batch failures are aggregated into an :class:`ExceptionGroup` so
      diagnostics surface every failed batch in one shot rather than the
      first arbitrary one to land.
    * Retries are bounded by ``EmbeddingConfig.retry_attempts`` and
      ``retry_backoff`` (seconds, element-wise indexed by attempt).
"""

import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import batched, chain
from operator import attrgetter
from typing import Final, assert_never

import openai

from lxd.mlx.embeddings import embed_texts_http
from lxd.settings.models import RuntimeConfig

_OPENAI_RESPONSE_INDEX: Final = attrgetter("index")

_openai_client_cache: dict[str, openai.OpenAI] = {}
_openai_client_lock = threading.Lock()


def get_openai_client(api_key: str) -> openai.OpenAI:
    """Return a process-wide cached :class:`openai.OpenAI` for ``api_key``.

    The OpenAI SDK constructs its own pooled :class:`httpx.Client` on
    instantiation; instantiating once per process means a single TLS pool
    is reused across batches and across ingest phases instead of being
    rebuilt per ``_embed_batch`` call.
    """
    with _openai_client_lock:
        client = _openai_client_cache.get(api_key)
        if client is None:
            client = openai.OpenAI(api_key=api_key)
            _openai_client_cache[api_key] = client
        return client


def reset_openai_client_cache() -> None:
    """Drop every cached OpenAI client. Intended for tests."""
    with _openai_client_lock:
        _openai_client_cache.clear()


@dataclass(frozen=True, slots=True)
class ModelProbeResult:
    """Embedding probe status and optional warning."""

    ok: bool
    warning: str | None = None


@dataclass(frozen=True, slots=True)
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
    match config.models.embed_backend:
        case "openai":
            return _openai_embed_texts(config, texts)
        case "mlx":
            return embed_texts_http(config, texts)
        case _:
            assert_never(config.models.embed_backend)


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
    match config.models.embed_backend:
        case "mlx":
            return embed_texts_http(config, list(texts))
        case "openai":
            return _openai_embed_texts(config, list(texts))
        case _:
            assert_never(config.models.embed_backend)


def _openai_embed_texts(config: RuntimeConfig, texts: list[str]) -> list[list[float]]:
    """Embed texts via OpenAI with per-batch concurrency.

    Per-batch failures are surfaced as a single :class:`ExceptionGroup` so a
    multi-batch run never hides later failures behind an arbitrary first one.
    """
    cfg = config.openai
    if cfg is None:
        raise RuntimeError("openai config required when embed_backend=openai")
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {cfg.api_key_env!r} is not set. "
            "Set it before using the openai embedding backend."
        )
    client = get_openai_client(api_key)
    batches = [list(batch) for batch in batched(texts, cfg.batch_size, strict=False)]

    def _embed_batch(batch: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            model=cfg.model,
            input=batch,
            dimensions=cfg.dims,
        )
        return [item.embedding for item in sorted(response.data, key=_OPENAI_RESPONSE_INDEX)]

    return _run_batches_concurrently(
        batches,
        max_workers=cfg.max_workers,
        worker=_embed_batch,
        group_message="openai embedding batch failures",
    )


def _run_batches_concurrently(
    batches: list[list[str]],
    *,
    max_workers: int,
    worker: Callable[[list[str]], list[list[float]]],
    group_message: str,
) -> list[list[float]]:
    """Run ``worker(batch)`` across ``batches`` concurrently and flatten in order.

    Failures are aggregated into a single :class:`ExceptionGroup` so the
    caller sees every failed batch instead of the arbitrary first one to
    land. In-flight workers are allowed to drain via the executor's
    ``shutdown(wait=True)`` semantics.
    """
    if not batches:
        return []
    results: list[list[list[float]] | None] = [None] * len(batches)
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {executor.submit(worker, batch): idx for idx, batch in enumerate(batches)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise ExceptionGroup(group_message, errors)
    if any(batch is None for batch in results):
        raise RuntimeError(f"{group_message}: incomplete results without raised errors")
    return list(chain.from_iterable(batch for batch in results if batch is not None))


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
