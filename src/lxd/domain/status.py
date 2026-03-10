from __future__ import annotations

from enum import StrEnum


class LifecycleStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    DELETED = "deleted"


class RetrievalStatus(StrEnum):
    SEARCHABLE = "searchable"
    ASSET_ONLY = "asset_only"
    NOT_SEARCHABLE = "not_searchable"


class QueryAnswerStatus(StrEnum):
    ANSWERED = "answered"
    NO_RESULTS = "no_results"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SYNTHESIS_UNAVAILABLE = "synthesis_unavailable"
