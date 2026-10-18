"""Chat completions against a local ``mlx_lm.server`` process."""

import json
import sys
from collections.abc import Iterator
from typing import Any

import httpx
import openai
import structlog

from lxd.mlx.launch import ensure_subprocess, port_open
from lxd.settings.models import MlxConfig, RuntimeConfig

_log = structlog.get_logger(__name__)

_clients: dict[str, openai.OpenAI] = {}

ChatError: tuple[type[BaseException], ...] = (
    openai.OpenAIError,
    httpx.HTTPError,
    OSError,
    RuntimeError,
    TimeoutError,
)


def complete_chat(
    config: RuntimeConfig,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout_secs: float,
    model: str | None = None,
) -> str:
    """Generate one completion from the resident MLX writer.

    A requested model other than ``models.llm`` is ignored. ``mlx_lm.server``
    drops the loaded model when a request names a different one, and this
    process keeps a single writer resident.
    """
    ensure_chat_server(config)
    client = _client(config)
    response = client.chat.completions.create(
        model=_resident_model(config, model),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_secs,
    )
    return response.choices[0].message.content or ""


def stream_chat(
    config: RuntimeConfig,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout_secs: float,
    model: str | None = None,
) -> Iterator[str]:
    """Stream completion text from the resident MLX writer.

    The server is contacted before the iterator is returned, so a refused
    connection surfaces to the caller instead of on the first ``next``.
    """
    ensure_chat_server(config)
    response = _client(config).chat.completions.create(
        model=_resident_model(config, model),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_secs,
        stream=True,
    )
    return _yield_content(response)


def _yield_content(response: Any) -> Iterator[str]:
    for chunk in response:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        text = choices[0].delta.content or ""
        if text:
            yield text


def ensure_chat_server(config: RuntimeConfig) -> None:
    """Start ``mlx_lm.server`` when chat auto-start is on and the port is closed."""
    mlx = _require_mlx(config)
    if port_open(mlx.chat_host, mlx.chat_port):
        return
    if not mlx.auto_start_chat:
        raise RuntimeError(
            f"MLX chat server is not accepting connections at {mlx.chat_url}. "
            "Start mlx_lm.server or set mlx.auto_start_chat: true."
        )
    runtime_dir = config.paths.data_path / "runtime"
    command = _server_command(config)
    ensure_subprocess(
        host=mlx.chat_host,
        port=mlx.chat_port,
        command=command,
        log_path=runtime_dir / "mlx-chat.log",
        pid_path=runtime_dir / "mlx-chat.json",
        startup_timeout_secs=mlx.startup_timeout_secs,
        label="MLX chat server",
    )


def _resident_model(config: RuntimeConfig, requested: str | None) -> str:
    resident = config.models.llm
    if requested is not None and requested != resident:
        _log.warning(
            "mlx_model_override_ignored",
            requested=requested,
            resident=resident,
        )
    return resident


def _server_command(config: RuntimeConfig) -> list[str]:
    mlx = _require_mlx(config)
    command = [
        sys.executable,
        "-m",
        "mlx_lm.server",
        "--model",
        config.models.llm,
        "--host",
        mlx.chat_host,
        "--port",
        str(mlx.chat_port),
    ]
    if config.models.llm_no_think:
        command.extend(
            ["--chat-template-args", json.dumps({"enable_thinking": False})],
        )
    return command


def _client(config: RuntimeConfig) -> openai.OpenAI:
    base_url = str(_require_mlx(config).chat_url).rstrip("/") + "/v1"
    cached = _clients.get(base_url)
    if cached is not None:
        return cached
    client = openai.OpenAI(base_url=base_url, api_key="mlx", max_retries=0)
    _clients[base_url] = client
    return client


def _require_mlx(config: RuntimeConfig) -> MlxConfig:
    mlx = config.mlx
    if mlx is None:
        raise RuntimeError("models.llm_backend=mlx requires an mlx config section.")
    return mlx
