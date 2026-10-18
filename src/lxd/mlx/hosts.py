"""Resolve the local chat host without importing MLX."""

from lxd.settings.models import RuntimeConfig


def local_chat_host(config: RuntimeConfig) -> str:
    """Return the OpenAI-compatible base URL for local chat via MLX.

    The returned URL has no ``/v1`` suffix; callers that speak the OpenAI SDK append it.
    """
    mlx = getattr(config, "mlx", None)
    if mlx is None:
        raise RuntimeError("models.llm_backend=mlx requires an mlx config section.")
    return str(mlx.chat_url).rstrip("/")
