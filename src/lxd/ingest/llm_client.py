"""Shared async LLM client for OpenAI and Ollama chat completions.

Provides lazy client singletons, async call-with-fallback, concurrent extraction
with sub-batch commits, prompt caching expansion, and OpenAI Batch API helpers.
"""

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Final, Literal, assert_never

import ollama
import openai
import structlog

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Client singletons
# ---------------------------------------------------------------------------
#
# The AsyncOpenAI client is a per-api-key singleton (credentials never change
# mid-process). The Ollama client is keyed on ``(host, timeout)`` because two
# pipeline stages may legitimately target different hosts or request longer
# timeouts, and returning a stale client would silently misroute calls.

_async_openai_clients: dict[str, openai.AsyncOpenAI] = {}
_sync_openai_clients: dict[str, openai.OpenAI] = {}
_sync_ollama_clients: dict[tuple[str, float], ollama.Client] = {}

# Bump above the SDK default of 2 so 429/5xx bursts (which the SDK backs off
# through with Retry-After honouring) don't fail after two misses. Applies to
# both sync and async clients so batch polling and hot extraction share the
# same retry budget.
_OPENAI_MAX_RETRIES: Final = 3


def get_async_openai_client(api_key_env: str = "OPENAI_API_KEY") -> openai.AsyncOpenAI:
    """Return a lazily-initialised ``AsyncOpenAI`` client keyed on API key env."""
    cached = _async_openai_clients.get(api_key_env)
    if cached is not None:
        return cached
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env!r} is not set.")
    client = openai.AsyncOpenAI(api_key=api_key, max_retries=_OPENAI_MAX_RETRIES)
    _async_openai_clients[api_key_env] = client
    return client


def get_sync_openai_client(api_key_env: str = "OPENAI_API_KEY") -> openai.OpenAI:
    """Return a lazily-initialised sync ``OpenAI`` client keyed on API key env.

    Mirrors :func:`get_async_openai_client` for the code paths that must
    stay synchronous (Batch API helpers, extraction fallbacks). Reusing
    one client keeps the HTTP connection pool warm across the many small
    calls a long-running batch poll issues; the previous pattern of
    ``openai.OpenAI(api_key=...)`` per call paid a fresh TLS handshake
    every time.

    Per-call ``timeout`` overrides are still allowed at the request site
    (e.g. ``client.chat.completions.create(..., timeout=X)``); this
    constructor sets no default timeout so short polls are not throttled
    to the same ceiling as long completions.
    """
    cached = _sync_openai_clients.get(api_key_env)
    if cached is not None:
        return cached
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env!r} is not set.")
    client = openai.OpenAI(api_key=api_key, max_retries=_OPENAI_MAX_RETRIES)
    _sync_openai_clients[api_key_env] = client
    return client


def get_ollama_client(host: str, timeout: float) -> ollama.Client:
    """Return a lazily-initialised Ollama client keyed on ``(host, timeout)``.

    The key ensures that two callers with different hosts or different
    per-request timeouts do not share a single client: the Ollama client
    captures both values at construction time.
    """
    key = (host, float(timeout))
    cached = _sync_ollama_clients.get(key)
    if cached is not None:
        return cached
    client = ollama.Client(host=host, timeout=timeout)
    _sync_ollama_clients[key] = client
    return client


def reset_clients() -> None:
    """Reset client singletons (for testing)."""
    _async_openai_clients.clear()
    _sync_openai_clients.clear()
    _sync_ollama_clients.clear()


# ---------------------------------------------------------------------------
# Async call helpers
# ---------------------------------------------------------------------------


async def call_openai_async(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    timeout: float = 90.0,
    max_tokens: int = 2000,
    response_format: dict[str, str] | None = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> str:
    """Make a single async OpenAI chat completion call.

    Returns the response content string.
    """
    client = get_async_openai_client(api_key_env)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


async def call_ollama_sync_in_thread(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    host: str,
    temperature: float = 0.0,
    timeout: float = 90.0,
    format_: str | None = "json",
) -> str:
    """Call Ollama synchronously, wrapped in asyncio.to_thread for async compat.

    Returns the response content string.
    """

    def _sync_call() -> str:
        client = get_ollama_client(host, timeout)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": temperature},
        }
        if format_ is not None:
            kwargs["format"] = format_
        response = client.chat(**kwargs)
        content = (
            response["message"]["content"]
            if isinstance(response, dict)
            else response.message.content
        )
        return content or ""

    return await asyncio.to_thread(_sync_call)


async def call_with_fallback_async(
    *,
    system_prompt: str,
    user_prompt: str,
    primary_backend: Literal["openai", "ollama"],
    openai_model: str,
    ollama_model: str,
    fallback_backend: Literal["ollama", "none"] = "ollama",
    temperature: float = 0.0,
    openai_timeout: float = 90.0,
    ollama_timeout: float = 90.0,
    max_tokens: int = 2000,
    response_format: dict[str, str] | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    ollama_host: str = "http://localhost:11434",
    ollama_format: str | None = "json",
) -> str:
    """Call primary backend with fallback.

    Returns response content string, or empty string if all backends fail.
    """
    match primary_backend:
        case "openai":
            try:
                return await call_openai_async(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=openai_model,
                    temperature=temperature,
                    timeout=openai_timeout,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    api_key_env=api_key_env,
                )
            except Exception as exc:
                _log.warning("openai_call_failed", error=str(exc))
                if fallback_backend != "ollama":
                    return ""
                try:
                    return await call_ollama_sync_in_thread(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=ollama_model,
                        host=ollama_host,
                        temperature=temperature,
                        timeout=ollama_timeout,
                        format_=ollama_format,
                    )
                except Exception as fallback_exc:
                    _log.warning("ollama_fallback_failed", error=str(fallback_exc))
                    return ""
        case "ollama":
            try:
                return await call_ollama_sync_in_thread(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=ollama_model,
                    host=ollama_host,
                    temperature=temperature,
                    timeout=ollama_timeout,
                    format_=ollama_format,
                )
            except Exception as exc:
                _log.warning("ollama_call_failed", error=str(exc))
                return ""
        case _:
            assert_never(primary_backend)


# ---------------------------------------------------------------------------
# Concurrent extraction orchestrator
# ---------------------------------------------------------------------------


async def run_concurrent_extraction[T, R](
    items: Sequence[T],
    extract_fn: Callable[[T], Awaitable[R]],
    *,
    max_concurrent: int = 50,
    sub_batch_size: int = 500,
    commit_fn: Callable[[list[R]], None] | None = None,
    label: str = "extraction",
) -> list[R]:
    """Run extract_fn concurrently over items with semaphore gating and sub-batch commits.

    Args:
        items: Sequence of input items to process.
        extract_fn: Async function that processes one item and returns a result.
        max_concurrent: Maximum concurrent in-flight calls.
        sub_batch_size: Commit results to SQLite every N items.
        commit_fn: Called with each sub-batch of results for persistence.
        label: Label for progress logging.

    Returns:
        All results flattened.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    all_results: list[R] = []

    async def _guarded(item: T) -> R | Exception:
        # Catch per-item failures inside the task so ``asyncio.TaskGroup``
        # does not abort the whole batch via its first-error semantics.
        # ``CancelledError`` is *not* caught — cancellation propagates so
        # ``Ctrl-C`` works correctly.
        async with semaphore:
            try:
                return await extract_fn(item)
            except Exception as exc:
                return exc

    total = len(items)
    for batch_start in range(0, total, sub_batch_size):
        batch_end = min(batch_start + sub_batch_size, total)
        batch_items = items[batch_start:batch_end]

        start_time = time.monotonic()
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(_guarded(item)) for item in batch_items]
        # All tasks have completed by exit; pull results without awaiting.
        good_results: list[R] = []
        errors = 0
        for task in tasks:
            result = task.result()
            if isinstance(result, Exception):
                errors += 1
                _log.warning("extraction_item_failed", error=str(result), label=label)
            else:
                good_results.append(result)

        if commit_fn is not None and good_results:
            commit_fn(good_results)

        all_results.extend(good_results)
        elapsed = time.monotonic() - start_time

        _log.info(
            "sub_batch_complete",
            label=label,
            batch_start=batch_start,
            batch_end=batch_end,
            total=total,
            results=len(good_results),
            errors=errors,
            elapsed_secs=round(elapsed, 1),
        )

    return all_results


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------


def build_cached_system_prompt(
    base_prompt: str,
    entity_vocabulary: list[str] | None = None,
    predicate_vocabulary: list[str] | None = None,
) -> str:
    """Expand a system prompt with entity/predicate vocabulary for OpenAI prompt caching.

    OpenAI automatically caches prompt prefixes >= 1,024 tokens at 50% off.
    The vocabulary section is identical across all calls in a run, creating
    a stable, byte-identical prefix that triggers caching.
    """
    sections = [base_prompt]

    if entity_vocabulary:
        entity_list = "\n".join(f"  - {e}" for e in sorted(entity_vocabulary))
        sections.append(
            f"\n\n--- REFERENCE: Known entity vocabulary ({len(entity_vocabulary)} entities) ---\n"
            f"{entity_list}"
        )

    if predicate_vocabulary:
        pred_list = ", ".join(sorted(predicate_vocabulary))
        sections.append(
            f"\n\n--- REFERENCE: Valid predicate vocabulary ({len(predicate_vocabulary)} predicates) ---\n"
            f"{pred_list}"
        )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# OpenAI Batch API helpers
# ---------------------------------------------------------------------------


def prepare_batch_jsonl(
    items: Sequence[dict[str, Any]],
    *,
    build_messages_fn: Callable[[dict[str, Any]], list[dict[str, str]]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    response_format: dict[str, str] | None = None,
    output_path: Path,
) -> Path:
    """Write a JSONL file for the OpenAI Batch API.

    Each item produces one line with custom_id, method, url, and body.
    Returns the output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            messages = build_messages_fn(item)
            body: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                body["response_format"] = response_format
            line = {
                "custom_id": item.get("custom_id", f"item-{idx}"),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line) + "\n")

    _log.info("batch_jsonl_prepared", path=str(output_path), items=len(items))
    return output_path


def submit_batch(
    jsonl_path: Path,
    *,
    description: str = "lxd-machine extraction batch",
    api_key_env: str = "OPENAI_API_KEY",
    metadata: dict[str, str] | None = None,
) -> str:
    """Upload JSONL and submit an OpenAI batch job.

    Returns the batch ID.
    """
    client = get_sync_openai_client(api_key_env)

    with jsonl_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    create_kwargs: dict[str, Any] = {
        "input_file_id": uploaded.id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": metadata or {},
    }
    batch = client.batches.create(**create_kwargs)
    _log.info("batch_submitted", batch_id=batch.id, file_id=uploaded.id)

    # Write sidecar metadata
    sidecar_path = jsonl_path.with_suffix(".batch_meta.json")
    sidecar_path.write_text(
        json.dumps(
            {
                "batch_id": batch.id,
                "input_file_id": uploaded.id,
                "description": description,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "jsonl_path": str(jsonl_path),
            },
            indent=2,
        )
    )

    return batch.id


def poll_batch(
    batch_id: str,
    *,
    api_key_env: str = "OPENAI_API_KEY",
) -> dict[str, Any]:
    """Check batch status. Returns batch object as dict."""
    client = get_sync_openai_client(api_key_env)
    batch = client.batches.retrieve(batch_id)
    return {
        "id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": {
            "total": getattr(batch.request_counts, "total", 0) if batch.request_counts else 0,
            "completed": (
                getattr(batch.request_counts, "completed", 0) if batch.request_counts else 0
            ),
            "failed": getattr(batch.request_counts, "failed", 0) if batch.request_counts else 0,
        },
    }


def collect_batch_results(
    batch_id: str,
    *,
    parse_fn: Callable[[str, str], Any],
    api_key_env: str = "OPENAI_API_KEY",
) -> list[Any]:
    """Download and parse batch results.

    Args:
        batch_id: The OpenAI batch ID.
        parse_fn: Called with (custom_id, response_content) for each result line.

    Returns:
        List of parsed results from parse_fn.
    """
    client = get_sync_openai_client(api_key_env)
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch_id} status is {batch.status!r}, not 'completed'.")

    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch_id} has no output_file_id.")

    content = client.files.content(batch.output_file_id)
    results: list[Any] = []

    for line in content.text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row.get("custom_id", "")
        response_body = row.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if choices:
            message_content = choices[0].get("message", {}).get("content", "")
            parsed = parse_fn(custom_id, message_content)
            if parsed is not None:
                results.append(parsed)

    _log.info("batch_results_collected", batch_id=batch_id, results=len(results))
    return results
