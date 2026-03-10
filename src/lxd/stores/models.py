from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorePaths:
    sqlite_path: Path
    lancedb_path: Path


@dataclass(frozen=True)
class CorpusStatusSummary:
    corpus_file_count: int
    text_file_count: int
    asset_file_count: int
    retrieval_role_counts: dict[str, int]
    chunk_count: int
    mention_count: int
    ontology_file_count: int
    matcher_term_count: int
    matcher_termset_hash: str | None
    ontology_snapshot_hash: str | None
    ontology_coverage_path_count: int
    ontology_graph_relation_count: int
    ontology_validation_issue_count: int
    ontology_validation_issue_samples: list[str]
    config_drift_warnings: list[str]


@dataclass(frozen=True)
class ManifestRecord:
    source_rel_path: str
    absolute_path: str
    source_type: str
    source_domain: str
    document_id: str | None
    file_size_bytes: int
    content_hash: str
    parent_source_path: str | None
    chunk_count: int
    last_seen_at: str
    last_processed_at: str | None
    last_committed_at: str | None
    error_message: str | None
    lifecycle_status: str = "pending"
    retrieval_status: str = "not_searchable"


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    source_rel_path: str
    source_path: str
    source_filename: str
    source_type: str
    source_domain: str
    source_hash: str
    citation_label: str
    chunk_index: int
    chunk_occurrence: int
    token_count: int
    text: str
    chunk_hash: str
    score_hint: str
    metadata_json: str
    vector: list[float]
    embedding_model: str
    embedding_dims: int


@dataclass(frozen=True)
class MentionRecord:
    chunk_id: str
    entity_id: str
    term_source: str
    surface_form: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class AssetLinkRecord:
    asset_rel_path: str
    asset_filename: str
    source_domain: str
    parent_source_path: str | None
    parent_document_id: str | None
    link_method: str
    page_no: int | None
    asset_index: int | None
    blake3_hash: str
    last_committed_at: str


@dataclass(frozen=True)
class OntologySourceRecord:
    file_path: str
    file_rel_path: str
    blake3_hash: str
    last_seen_at: str


@dataclass(frozen=True)
class OntologySnapshotRecord:
    snapshot_id: str
    ontology_root: str
    snapshot_hash: str
    matcher_termset_hash: str
    matcher_term_count: int
    source_file_count: int
    entity_file_count: int
    entity_count: int
    coverage_path_count: int
    graph_relation_count: int
    validation_issue_count: int
    validation_issues_json: str
    last_loaded_at: str


@dataclass(frozen=True)
class IngestConfigSnapshotRecord:
    key: str
    value: str


@dataclass(frozen=True)
class VectorSearchRecord:
    chunk_id: str
    document_id: str
    source_rel_path: str
    source_path: str
    source_filename: str
    source_type: str
    source_domain: str
    source_hash: str
    citation_label: str
    chunk_index: int
    chunk_occurrence: int
    token_count: int
    text: str
    score_hint: str
    metadata_json: str
    score: float
