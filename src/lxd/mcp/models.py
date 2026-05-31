"""Pydantic output models for every MCP tool, resource, and prompt.

FastMCP derives JSON output schemas from these models, so MCP clients receive
structured, schema-discoverable responses instead of opaque dicts. Every model
is frozen and forbids extra keys to catch silent shape drift at the boundary.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Base class for MCP output models. Frozen, strict, extra-keys forbidden."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)


class CorpusCounts(_Frozen):
    """File-count breakdown for the corpus."""

    total: int = Field(description="Total corpus files tracked (text + asset).")
    text: int = Field(description="Number of text source files (Markdown, Docling JSON).")
    asset: int = Field(description="Number of asset files (PNGs, etc.).")


class CorpusStatusResponse(_Frozen):
    """Health snapshot returned by ``corpus_status``."""

    corpus_counts: CorpusCounts
    retrieval_role_counts: dict[str, int] = Field(
        description="Counts keyed by retrieval role: searchable / asset_only / not_searchable."
    )
    chunk_count: int
    mention_count: int
    ontology_file_count: int
    entity_count: int
    matcher_term_count: int
    ontology_snapshot_hash: str | None
    matcher_termset_hash: str | None
    ontology_coverage_path_count: int
    ontology_graph_relation_count: int
    ontology_validation_issue_count: int
    ontology_validation_issue_samples: list[str]
    config_drift_warnings: list[str]


class EntityNeighbor(_Frozen):
    """A single edge from the ontology graph."""

    entity_id: str
    relation: str
    direction: str = Field(description="Edge direction: ``outgoing`` or ``incoming``.")


class ChunkSearchResult(_Frozen):
    """One ranked chunk returned by ``search_corpus``."""

    chunk_id: str
    document_id: str
    citation_label: str
    source_rel_path: str
    score: float
    text: str
    metadata_json: str = Field(description="Per-chunk metadata, JSON-encoded.")


class ConceptDocumentMatch(_Frozen):
    """One chunk matched against an expanded entity set."""

    chunk_id: str
    document_id: str
    citation_label: str
    source_rel_path: str
    score: float
    entity_match_count: int
    matched_from_total: int = Field(
        description="Number of entity IDs in the expansion set this chunk could match against."
    )
    text: str
    metadata_json: str


class CorpusRelation(_Frozen):
    """One extracted (subject, predicate, object) relation from the corpus."""

    subject: str
    predicate: str
    object: str
    confidence: float
    source_rel_path: str
    chunk_id: str


class EntitySummary(_Frozen):
    """Full entity profile returned by ``get_entity_summary``."""

    entity_id: str
    label: str
    entity_type: str
    domain: str | None = None
    aliases: str | None = Field(default=None, description="JSON-encoded list of aliases.")
    deterministic_summary: str | None = None
    llm_summary: str | None = None
    chunk_count: int
    doc_count: int
    mention_count: int
    claim_count: int
    top_predicates: str | None = Field(
        default=None, description="JSON-encoded top predicates with counts."
    )
    top_claims: str | None = Field(default=None, description="JSON-encoded top claim payloads.")
    pagerank: float
    betweenness: float
    closeness: float
    in_degree: int
    out_degree: int
    eigenvector: float
    community_id: int | None


class CommunityContext(_Frozen):
    """Community report returned by ``get_community_context``."""

    community_id: int
    member_count: int
    member_entity_ids: str = Field(description="JSON-encoded list of member entity IDs.")
    deterministic_summary: str | None = None
    llm_summary: str | None = None
    top_entities: str | None = Field(default=None, description="JSON-encoded top entities.")
    top_claims: str | None = Field(default=None, description="JSON-encoded top claims.")
    intra_community_edge_count: int


class SimilarEntity(_Frozen):
    """One entity returned by vector-similarity search."""

    entity_id: str
    label: str
    community_id: int | None
    score: float = Field(description="Cosine distance (lower = closer).")


class EntitySearchResult(_Frozen):
    """One entity returned by name/alias search."""

    entity_id: str
    label: str
    entity_type: str
    pagerank: float
    community_id: int | None
    mention_count: int


class RelationEvidence(_Frozen):
    """One evidence row for a canonical relation."""

    evidence_id: str
    relation_id: str
    chunk_id: str
    surface_subject: str
    surface_object: str
    evidence_text: str = Field(description="Trimmed to 500 chars at the tool boundary.")
    confidence: float
    extraction_model: str


class PathEdge(_Frozen):
    """One edge in an unweighted shortest-path result."""

    source: str
    target: str
    relation_type: str


class PathBetweenEntities(_Frozen):
    """Unweighted shortest-path response from ``find_path_between_entities``."""

    path: list[str]
    edges: list[PathEdge]
    hops: int
    note: str | None = None


class WeightedEdge(_Frozen):
    """One edge in a confidence-weighted path result."""

    source: str
    target: str
    weight: float = Field(description="1 - confidence (so Dijkstra minimises cost).")


class WeightedPath(_Frozen):
    """Confidence-weighted path response from ``find_weighted_path``."""

    path: list[str]
    edges: list[WeightedEdge]
    total_weight: float


class HubEntity(_Frozen):
    """One entity in the top-PageRank list."""

    entity_id: str
    label: str
    pagerank: float
    community_id: int | None


class BridgeEntity(_Frozen):
    """One entity in the top-betweenness list."""

    entity_id: str
    label: str
    betweenness: float
    community_id: int | None


class FoundationalEntity(_Frozen):
    """One entity in the top-closeness list."""

    entity_id: str
    label: str
    closeness: float
    community_id: int | None


class EntityGraphStats(_Frozen):
    """Counts-only graph statistics from ``get_entity_graph_stats``."""

    graph_version: int
    last_build_at: str = Field(default="never", description="ISO-8601 timestamp or 'never'.")
    entity_profiles: int
    communities: int
    community_reports: int
    canonical_relations: int
    relation_evidence: int
    claims: int


class GraphOverview(_Frozen):
    """High-level graph overview from ``get_graph_overview``."""

    graph_version: int
    last_build_at: str = Field(default="never")
    community_algorithm: str | None = None
    entity_profiles: int
    communities: int
    community_reports: int
    canonical_relations: int
    relation_evidence: int
    claims: int


class GraphContextEntityProfile(_Frozen):
    """Entity profile included in deep-search graph context."""

    entity_id: str
    label: str
    entity_type: str
    deterministic_summary: str | None = None
    llm_summary: str | None = None
    pagerank: float
    community_id: int | None


class GraphContextCommunityReport(_Frozen):
    """Community report included in deep-search graph context."""

    community_id: int
    member_count: int
    deterministic_summary: str | None = None
    llm_summary: str | None = None


class GraphContextClaim(_Frozen):
    """Claim included in deep-search graph context."""

    claim_text: str
    claim_type: str
    confidence: float
    subject_entity_id: str | None = None
    object_entity_id: str | None = None


class GraphContextData(_Frozen):
    """Structured graph context from ``search_knowledge_deep``."""

    level: str = Field(description="``none`` / ``minimal`` / ``standard`` / ``deep``.")
    entity_profiles: list[GraphContextEntityProfile] = Field(default_factory=list)
    community_reports: list[GraphContextCommunityReport] = Field(default_factory=list)
    claims: list[GraphContextClaim] = Field(default_factory=list)


class KnowledgeAnswer(_Frozen):
    """Full answer envelope from ``search_knowledge``."""

    answer_status: str = Field(
        description="``answered`` / ``no_results`` / ``insufficient_evidence`` / ``synthesis_unavailable``."
    )
    answer_text: str
    citations: list[str] = Field(description="Citation labels referenced by the answer.")
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Query-pipeline metadata: matched entities, expansion terms, fusion stats.",
    )


class KnowledgeAnswerDeep(_Frozen):
    """Deep-search answer envelope with structured graph context."""

    answer_status: str
    answer_text: str
    citations: list[str]
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    graph_context: GraphContextData
