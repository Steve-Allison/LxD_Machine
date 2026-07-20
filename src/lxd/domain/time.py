"""UTC clock helpers shared across ingest and stores."""

from datetime import UTC, datetime


def utc_now() -> str:
    """Return an aware UTC ISO-8601 timestamp (``datetime.isoformat()``)."""
    return datetime.now(UTC).isoformat()
