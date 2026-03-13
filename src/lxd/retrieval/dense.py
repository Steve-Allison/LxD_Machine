from __future__ import annotations

from lxd.ingest.embedder import (
    EmbeddingContextError,
    ModelProbeResult,
    embed_texts,
    probe_embedder,
)
from lxd.settings.models import RuntimeConfig

__all__ = [
    "EmbeddingContextError",
    "ModelProbeResult",
    "embed_query",
    "embed_texts",
    "probe_embedder",
]


def embed_query(config: RuntimeConfig, text: str) -> list[float]:
    instruction = getattr(config.embedding, "query_instruction", None)
    prefixed = f"{instruction}{text}" if instruction else text
    return embed_texts(config, [prefixed])[0]
