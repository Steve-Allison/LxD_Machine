"""Define typed store-layer records used across persistence boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorePaths:
    """Filesystem paths for SQLite and LanceDB stores."""

    sqlite_path: Path
    lancedb_path: Path


@dataclass(frozen=True)
class CorpusStatusSummary:
    """Aggregate corpus and ontology status counters."""

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
    """Manifest row describing ingest state for one source."""

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
    """Persisted chunk row with embedding and source metadata."""

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
    """Entity mention span detected in a chunk."""

    chunk_id: str
    entity_id: str
    term_source: str
    surface_form: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class AssetLinkRecord:
    """Resolved asset-to-parent link metadata."""

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
    """Persisted ontology source file metadata."""

    file_path: str
    file_rel_path: str
    blake3_hash: str
    last_seen_at: str


@dataclass(frozen=True)
class OntologySnapshotRecord:
    """Persisted ontology snapshot metadata and hashes."""

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
    """Persisted ingest config key-value entry."""

    key: str
    value: str


@dataclass(frozen=True)
class EntityMentionResult:
    """Chunk match summary for entity mention queries."""

    chunk_id: str
    document_id: str
    source_rel_path: str
    citation_label: str
    chunk_index: int
    text: str
    score_hint: str
    metadata_json: str
    entity_match_count: int
    total_entity_ids: int

    @property
    def score(self) -> float:
        """Return the computed result for this operation.

        Returns:
            Entity match ratio for this result.
        """
        return self.entity_match_count / self.total_entity_ids if self.total_entity_ids > 0 else 0.0


@dataclass(frozen=True)
class ExtractedRelationRecord:
    """Relation extracted from chunk text."""

    relation_id: str
    chunk_id: str
    document_id: str
    source_rel_path: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    confidence: float
    extraction_model: str
    extracted_at: str


@dataclass(frozen=True)
class VectorSearchRecord:
    """Vector search hit with source metadata."""

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


# ---------------------------------------------------------------------------
# Knowledge Graph records (Phase 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimRecord:
    """Claim extracted from a chunk by LLM."""

    claim_id: str
    chunk_id: str
    document_id: str
    source_rel_path: str
    claim_text: str
    subject_entity_id: str | None
    object_entity_id: str | None
    claim_type: str
    confidence: float
    extraction_model: str
    extracted_at: str


@dataclass(frozen=True)
class EntityProfileRecord:
    """Persisted entity profile with centrality and summaries."""

    entity_id: str
    label: str
    entity_type: str
    domain: str
    aliases_json: str
    deterministic_summary: str
    llm_summary: str | None
    chunk_count: int
    doc_count: int
    mention_count: int
    claim_count: int
    top_predicates_json: str
    top_claims_json: str
    pagerank: float
    betweenness: float
    closeness: float
    in_degree: int
    out_degree: int
    eigenvector: float
    community_id: int | None
    source_hash: str
    generated_at: str


@dataclass(frozen=True)
class EntityCommunityRecord:
    """Community assignment for one entity."""

    entity_id: str
    community_id: int
    community_level: int
    modularity_class: str | None
    assigned_at: str


@dataclass(frozen=True)
class CommunityReportRecord:
    """Summary report for one community."""

    community_id: int
    community_level: int
    member_count: int
    member_entity_ids_json: str
    deterministic_summary: str
    llm_summary: str | None
    top_entities_json: str
    top_claims_json: str
    intra_community_edge_count: int
    source_hash: str
    generated_at: str


@dataclass(frozen=True)
class CanonicalRelationRecord:
    """Canonical consolidated relation (one row per unique triple)."""

    relation_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    support_count: int
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    first_seen_at: str
    last_seen_at: str


@dataclass(frozen=True)
class RelationEvidenceRecord:
    """Per-chunk evidence for a canonical relation."""

    evidence_id: str
    relation_id: str
    chunk_id: str
    surface_subject: str
    surface_object: str
    evidence_text: str
    confidence: float
    extraction_model: str
    extracted_at: str


@dataclass(frozen=True)
class GraphBuildStateRecord:
    """State machine row for a graph build run."""

    run_id: str
    started_at: str
    finished_at: str | None
    status: str
    current_phase: str
    graph_version: int
    relations_consolidated: int
    evidence_rows_built: int
    claims_extracted: int
    entity_profiles_built: int
    communities_detected: int
    community_reports_built: int
    centrality_computed: int
    entity_embeddings_computed: int
    llm_enrichment_count: int
    notes_json: str


@dataclass(frozen=True)
class GraphMetadataRecord:
    """Key-value metadata for the knowledge graph."""

    key: str
    value: str
    updated_at: str
