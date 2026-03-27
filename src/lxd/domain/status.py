"""Define lifecycle status enums for ingest, retrieval, and synthesis."""

from __future__ import annotations

from enum import StrEnum


class LifecycleStatus(StrEnum):
    """Represent ingestion lifecycle states for tracked sources."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    DELETED = "deleted"


class RetrievalStatus(StrEnum):
    """Represent retrieval eligibility states for tracked sources."""
    SEARCHABLE = "searchable"
    ASSET_ONLY = "asset_only"
    NOT_SEARCHABLE = "not_searchable"


class QueryAnswerStatus(StrEnum):
    """Represent terminal answer outcomes returned by query synthesis."""
    ANSWERED = "answered"
    NO_RESULTS = "no_results"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SYNTHESIS_UNAVAILABLE = "synthesis_unavailable"
