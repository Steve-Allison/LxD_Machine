"""Configure structured logging for runtime components.

Responsibility:
    Wire up :mod:`structlog` with UTC timestamps, contextvar propagation,
    and a renderer that matches the ``LoggingConfig`` output format.
    Provide a ``log_duration`` context manager so callers can emit
    consistent ``*.started`` / ``*.completed`` events around expensive
    operations without re-implementing timing each time.

Design boundary:
    The application must call :func:`configure_logging` at start-up (see
    :mod:`lxd.app.bootstrap`). Everywhere else, modules should simply do
    ``_log = structlog.get_logger(__name__)`` and emit events via
    ``_log.info("event.name", **context)``.

Key constraints:
    * All timestamps emitted by the log renderer are UTC (``utc=True``);
      do not emit local-time strings in log payloads.
    * Contextual fields set via :func:`structlog.contextvars.bind_contextvars`
      are merged into every event without the caller having to thread them
      through manually.
    * ``log_duration`` is synchronous. Async callers should wrap it in a
      context manager appropriate for their loop or call it around
      ``anyio.run(...)`` boundaries.
"""

import logging
import sys
import threading
import time
from collections.abc import Generator, MutableMapping
from contextlib import contextmanager
from typing import Any

import structlog


def configure_logging(
    level: str,
    output_format: str = "json",
    *,
    sample_rate: int = 1,
    sampled_event_names: frozenset[str] = frozenset(),
) -> None:
    """Validate configuration and apply runtime settings.

    Args:
        level: Logging level name (for example, INFO or DEBUG).
        output_format: Log renderer format ("json" or "console").
        sample_rate: Emit one in every ``sample_rate`` copies of the events
            named in ``sampled_event_names``. ``1`` disables sampling.
        sampled_event_names: Event names eligible for sampling (e.g.
            ``embedding_cache_hit``). Anything outside this set is always
            emitted; errors and ``critical`` events are also always emitted.

    Side Effects:
        Mutates global ``logging`` and ``structlog`` configuration. Safe to
        call multiple times; the last call wins.
    """
    numeric_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    renderer: structlog.types.Processor
    if output_format == "console":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        scrub_secrets,
    ]
    if sample_rate > 1 and sampled_event_names:
        processors.append(
            make_sampled_processor(
                rate=sample_rate,
                high_volume_events=sampled_event_names,
            )
        )
    processors.append(renderer)

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        # Route structlog records to stderr. Required for MCP stdio servers
        # where stdout is reserved for JSON-RPC; otherwise log lines collide
        # with the protocol stream and Claude Desktop rejects them as malformed.
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


_SECRET_KEY_FRAGMENTS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "auth",
    "credential",
    "cookie",
)


def scrub_secrets(
    _logger: object,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Structlog processor that replaces sensitive values with ``"***"``.

    Applies a shallow scan of top-level keys plus a recursive pass over
    nested dicts and lists. A value is redacted when its key (case-folded)
    contains any fragment from :data:`_SECRET_KEY_FRAGMENTS`. Fragments are
    substring matches rather than exact equality so ``openai_api_key`` and
    ``aws_auth_token`` are both caught.

    Args:
        _logger: Structlog logger (unused; present to match the processor
            signature).
        _method_name: Level name passed by structlog (unused).
        event_dict: Event payload to mutate in place.

    Returns:
        The mutated event dict.
    """
    _scrub_mapping(event_dict)
    return event_dict


def _scrub_mapping(data: MutableMapping[str, Any]) -> None:
    for key, value in list(data.items()):
        if any(fragment in key.casefold() for fragment in _SECRET_KEY_FRAGMENTS):
            data[key] = "***"
            continue
        if isinstance(value, MutableMapping):
            _scrub_mapping(value)
        elif isinstance(value, list):
            _scrub_sequence(value)


def _scrub_sequence(items: list[Any]) -> None:
    for item in items:
        if isinstance(item, MutableMapping):
            _scrub_mapping(item)
        elif isinstance(item, list):
            _scrub_sequence(item)


def make_sampled_processor(
    *,
    rate: int,
    high_volume_events: frozenset[str],
) -> structlog.types.Processor:
    """Build a structlog processor that drops most copies of high-volume events.

    The processor lets through:

    * **Errors** (events whose level is ``error`` or ``critical``) — always.
    * **Non-sampled events** (event names not in ``high_volume_events``) —
      always; this is the escape hatch for run-summary lines, lifecycle
      transitions, and anything the caller has not opted into sampling.
    * **One-in-``rate`` copies** of each high-volume event — counted via a
      lock-guarded counter per event name so the sampling is exact rather
      than probabilistic.

    Suppression raises :class:`structlog.DropEvent`, which structlog uses to
    discard a log call without further processing.

    Args:
        rate: Emit one in every ``rate`` events; ``rate=1`` disables
            sampling. Values < 1 are treated as 1.
        high_volume_events: Event names eligible for sampling. Anything
            outside this set is always emitted.

    Returns:
        A structlog processor suitable for placement *before* the renderer.
    """
    sample_rate = max(1, int(rate))
    counters: dict[str, int] = {}
    lock = threading.Lock()

    def processor(
        _logger: object,
        _method_name: str,
        event_dict: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        if _method_name in {"error", "critical"}:
            return event_dict
        event = event_dict.get("event")
        if not isinstance(event, str) or event not in high_volume_events:
            return event_dict
        if sample_rate == 1:
            return event_dict
        with lock:
            counter = counters.get(event, 0)
            counters[event] = counter + 1
            keep = counter % sample_rate == 0
        if keep:
            return event_dict
        raise structlog.DropEvent

    return processor


@contextmanager
def log_duration(
    event: str,
    *,
    logger: structlog.stdlib.BoundLogger | None = None,
    level: str = "info",
    **fields: Any,
) -> Generator[dict[str, Any]]:
    """Emit ``<event>.started`` and ``<event>.completed`` log entries around a block.

    The completion event includes a ``duration_ms`` field measured on the
    monotonic clock. On exception the event becomes ``<event>.failed`` with
    the exception class attached and is re-raised; callers do not need to
    swallow exceptions just to emit the timing record.

    Args:
        event: Base event name. Must follow the ``<namespace>.<verb>`` convention
            (e.g. ``"ingest.run"`` or ``"mcp.search_corpus"``).
        logger: Optional pre-bound logger. Defaults to a structlog logger
            bound to this module.
        level: Log level for the lifecycle events (``"info"`` or ``"debug"``).
        **fields: Extra key/value pairs to attach to both the start and
            completion events. The yielded dict may be mutated inside the
            ``with`` block to add fields that are only known mid-execution;
            the completion event will include them.

    Yields:
        A mutable dict seeded with ``fields``; additions survive to the
        completion event.

    Raises:
        Exception: Re-raises whatever the wrapped block raised, after
            logging ``<event>.failed``.
    """
    bound_logger = logger or structlog.get_logger(__name__)
    log = getattr(bound_logger, level, bound_logger.info)

    extras: dict[str, Any] = dict(fields)
    start = time.monotonic()
    log(f"{event}.started", **extras)
    try:
        yield extras
    except BaseException as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        bound_logger.error(
            f"{event}.failed",
            duration_ms=duration_ms,
            error_class=type(exc).__name__,
            **extras,
        )
        raise
    else:
        duration_ms = int((time.monotonic() - start) * 1000)
        log(f"{event}.completed", duration_ms=duration_ms, **extras)
