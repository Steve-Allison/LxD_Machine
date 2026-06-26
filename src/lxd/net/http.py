"""Provide shared sync and async HTTPX clients keyed on (base_url, timeout).

Design boundary:
    Callers should never instantiate ``httpx.Client`` or ``httpx.AsyncClient``
    directly. Doing so forces a fresh TCP/TLS handshake on every request and
    disables HTTP/2 multiplexing. Route every HTTP call through the factories
    below so the process-wide pool is reused.

Key constraints:
    * Singletons are keyed on the tuple ``(base_url, timeout_secs)`` — two
      configurations with different hosts or timeouts each get their own pool.
    * HTTP/2 is enabled with a modest keep-alive pool; adjust here, never at
      call sites.
    * Clients are closed on interpreter shutdown via ``atexit`` so subprocesses
      do not leak file descriptors or sockets.
"""

import atexit
import threading
from dataclasses import dataclass

import httpx

_DEFAULT_LIMITS = httpx.Limits(
    max_keepalive_connections=20,
    max_connections=40,
    keepalive_expiry=30.0,
)


@dataclass(frozen=True, slots=True)
class _ClientKey:
    """Immutable cache key for a pooled HTTPX client."""

    base_url: str
    timeout_secs: float


_sync_clients: dict[_ClientKey, httpx.Client] = {}
_async_clients: dict[_ClientKey, httpx.AsyncClient] = {}
_lock = threading.Lock()


def get_sync_client(*, base_url: str, timeout_secs: float) -> httpx.Client:
    """Return a pooled ``httpx.Client`` for ``(base_url, timeout_secs)``.

    Args:
        base_url: Absolute URL used as the client base, e.g. ``"http://localhost:8012"``.
        timeout_secs: Per-request timeout in seconds.

    Returns:
        Long-lived ``httpx.Client`` with HTTP/2 enabled and keep-alive pooling.

    Side Effects:
        Caches the client in a module-global dict; releases on interpreter
        shutdown via ``atexit``.
    """
    key = _ClientKey(base_url=base_url, timeout_secs=float(timeout_secs))
    with _lock:
        client = _sync_clients.get(key)
        if client is None or client.is_closed:
            client = httpx.Client(
                base_url=base_url,
                timeout=float(timeout_secs),
                http2=True,
                limits=_DEFAULT_LIMITS,
            )
            _sync_clients[key] = client
        return client


def get_async_client(*, base_url: str, timeout_secs: float) -> httpx.AsyncClient:
    """Return a pooled ``httpx.AsyncClient`` for ``(base_url, timeout_secs)``.

    Args:
        base_url: Absolute URL used as the client base.
        timeout_secs: Per-request timeout in seconds.

    Returns:
        Long-lived ``httpx.AsyncClient`` with HTTP/2 and keep-alive pooling.

    Side Effects:
        Caches the client in a module-global dict; closes on interpreter
        shutdown via ``atexit`` (best-effort, synchronous ``aclose`` via a
        one-shot event loop).
    """
    key = _ClientKey(base_url=base_url, timeout_secs=float(timeout_secs))
    with _lock:
        client = _async_clients.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                base_url=base_url,
                timeout=float(timeout_secs),
                http2=True,
                limits=_DEFAULT_LIMITS,
            )
            _async_clients[key] = client
        return client


def reset_clients() -> None:
    """Close all pooled clients and drop the cache.

    Intended for tests; production code relies on the ``atexit`` hook.
    """
    with _lock:
        for client in list(_sync_clients.values()):
            if not client.is_closed:
                client.close()
        _sync_clients.clear()
        for async_client in list(_async_clients.values()):
            if not async_client.is_closed:
                _safe_close_async(async_client)
        _async_clients.clear()


def _safe_close_async(client: httpx.AsyncClient) -> None:
    """Close an async client from sync context without raising on shutdown."""
    import asyncio

    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(client.aclose())
        finally:
            loop.close()
    except Exception:
        pass


@atexit.register
def _cleanup() -> None:
    reset_clients()
