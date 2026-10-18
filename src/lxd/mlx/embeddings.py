"""OpenAI-compatible client for the local Octen embedding server."""

import os
import sys
import threading
from itertools import batched

import openai

from lxd.mlx.launch import ensure_subprocess, port_open
from lxd.settings.models import MlxConfig, RuntimeConfig

_clients: dict[str, openai.OpenAI] = {}
_client_lock = threading.Lock()

_MAX_BATCH: int = 32


def embed_texts_http(config: RuntimeConfig, texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` via the local Octen server, in input order."""
    if not texts:
        return []
    ensure_embed_server(config)
    mlx = _require_mlx(config)
    client = _client(config)
    timeout = float(config.embedding.timeout_secs)
    batch_size = min(_MAX_BATCH, max(1, config.embedding.batch_size))
    vectors: list[list[float]] = []
    for batch_items in batched(texts, batch_size, strict=False):
        batch = list(batch_items)
        try:
            response = client.embeddings.create(
                model=config.models.embed,
                input=batch,
                timeout=timeout,
            )
        except openai.OpenAIError as exc:
            raise RuntimeError(
                f"Octen embeddings request to {mlx.embed_url} failed: {exc}"
            ) from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(batch):
            raise RuntimeError(
                f"Octen embeddings returned {len(ordered)} vectors for {len(batch)} inputs."
            )
        vectors.extend(item.embedding for item in ordered)
    return vectors


def ensure_embed_server(config: RuntimeConfig) -> None:
    """Start the Octen embedding server when auto-start is on and the port is closed."""
    mlx = _require_mlx(config)
    if port_open(mlx.embed_host, mlx.embed_port):
        return
    if not mlx.auto_start_embed:
        raise RuntimeError(
            f"Octen embeddings server is not accepting connections at {mlx.embed_url}. "
            "Start it, or set mlx.auto_start_embed: true."
        )
    runtime_dir = config.paths.data_path / "runtime"
    child_env = os.environ.copy()
    child_env["LXD_EMBED_WEIGHTS"] = mlx.embed_weights
    child_env["LXD_EMBED_HOST"] = mlx.embed_host
    child_env["LXD_EMBED_PORT"] = str(mlx.embed_port)
    child_env["LXD_EMBED_MODEL_ID"] = config.models.embed
    ensure_subprocess(
        host=mlx.embed_host,
        port=mlx.embed_port,
        command=[sys.executable, "-m", "lxd.mlx.embed_server"],
        log_path=runtime_dir / "mlx-embed.log",
        pid_path=runtime_dir / "mlx-embed.json",
        startup_timeout_secs=mlx.startup_timeout_secs,
        label="Octen embeddings server",
        env=child_env,
    )


def _client(config: RuntimeConfig) -> openai.OpenAI:
    base_url = str(_require_mlx(config).embed_url).rstrip("/") + "/v1"
    with _client_lock:
        cached = _clients.get(base_url)
        if cached is not None:
            return cached
        client = openai.OpenAI(base_url=base_url, api_key="mlx", max_retries=0)
        _clients[base_url] = client
        return client


def _require_mlx(config: RuntimeConfig) -> MlxConfig:
    mlx = config.mlx
    if mlx is None:
        raise RuntimeError("models.embed_backend=mlx requires an mlx config section.")
    return mlx
