"""Configure structured logging for runtime components."""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str, output_format: str = "json") -> None:
    """Validate configuration and apply runtime settings.

    Args:
        level: Logging level name (for example, INFO or DEBUG).
        output_format: Log renderer format ("json" or "console").
    """
    numeric_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    renderer: structlog.types.Processor
    if output_format == "console":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
