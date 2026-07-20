"""Pydantic output models for every MCP tool, resource, and prompt.

FastMCP derives JSON output schemas from these models, so MCP clients receive
structured, schema-discoverable responses instead of opaque dicts. Every model
is frozen and forbids extra keys to catch silent shape drift at the boundary.
"""

from typing import Literal

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


class PredicateCount(_Frozen):
    """One row in an entity's top-predicates breakdown."""

    predicate: str
    count: int


class TopClaim(_Frozen):
    """One claim in an entity's or community's top-claims list.

    ``claim_type`` is populated for entity-level top claims (from
    :func:`lxd.ontology.profiles.build_entity_profile`); it is absent for
    community-level top claims — that origin currently emits only
    ``claim_text`` + ``confidence``, so the field remains ``None`` there.
    """

    claim_text: str
    confidence: float
    claim_type: str | None = None


class TopEntity(_Frozen):
    """One entity in a community's top-entities list, keyed by PageRank."""

    entity_id: str
    pagerank: float


class EntitySummary(_Frozen):
    """Full entity profile returned by ``get_entity_summary``."""

    entity_id: str
    label: str
    entity_type: str
    domain: str | None = None
    aliases: list[str] = Field(
        default_factory=list, description="Alias surface forms for the entity."
    )
    deterministic_summary: str | None = None
    llm_summary: str | None = None
    chunk_count: int
    doc_count: int
    mention_count: int
    claim_count: int
    top_predicates: list[PredicateCount] = Field(
        default_factory=list, description="Top canonical predicates by frequency."
    )
    top_claims: list[TopClaim] = Field(
        default_factory=list, description="Top LLM-extracted claims for the entity."
    )
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
    member_entity_ids: list[str] = Field(
        default_factory=list, description="Entity IDs that belong to this community."
    )
    deterministic_summary: str | None = None
    llm_summary: str | None = None
    top_entities: list[TopEntity] = Field(
        default_factory=list, description="Top entities in the community by PageRank."
    )
    top_claims: list[TopClaim] = Field(
        default_factory=list, description="Top LLM-extracted claims across community members."
    )
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


class SentenceCitationView(_Frozen):
    """One sentence + the citation labels supporting it (MCP-facing shape).

    Mirrors :class:`lxd.synthesis.citation_alignment.SentenceCitation` but
    lives in the MCP namespace so clients can import a typed model
    alongside the rest of the MCP surface. Sentences with empty
    ``citation_labels`` are unattributed claims — surface them in UIs as
    a hallucination risk signal.
    """

    text: str
    citation_labels: list[str] = Field(default_factory=list)


class KnowledgeAnswerMetadata(_Frozen):
    """Typed query-pipeline metadata returned alongside a knowledge answer.

    The ``router_*`` fields are always populated (the router runs first,
    on every call). The remaining fields are populated only when
    retrieval + synthesis actually ran — the "no retrieval needed" branch
    from the adaptive router returns just the four ``router_*`` values.
    Absent-when-not-applicable is expressed as ``None`` (booleans / ints)
    or an empty list (id / term collections) so the schema surface is
    always the same shape.
    """

    router_retrieve: bool = Field(
        description="Whether the router decided to run retrieval + synthesis."
    )
    router_breadth: Literal["narrow", "standard", "broad"] = Field(
        description="Retrieval breadth knob the router picked (drives dense_top_k)."
    )
    router_rationale: str = Field(
        description="One-line reason the router gave for its retrieve+breadth call."
    )
    router_routed: bool = Field(
        description="True when the router LLM ran; false when we fell back to the default route."
    )
    reranking_applied: bool | None = Field(
        default=None,
        description="Whether the cross-encoder reranker ran. Null when retrieval was skipped.",
    )
    expansion_applied: bool | None = Field(
        default=None,
        description="Whether ontology expansion added terms. Null when retrieval was skipped.",
    )
    matched_entity_ids: list[str] = Field(
        default_factory=list,
        description="Ontology entities matched to the question after expansion.",
    )
    expansion_terms: list[str] = Field(
        default_factory=list,
        description="Terms added by the expansion pass.",
    )
    result_count: int | None = Field(
        default=None,
        description="Ranked chunk count after fusion. Null when retrieval was skipped.",
    )
    graph_context_applied: bool | None = Field(
        default=None,
        description="Whether graph context was prepended to synthesis. Null when skipped.",
    )
    dense_top_k: int | None = Field(
        default=None,
        description="Dense top-k the router picked for this call. Null when retrieval was skipped.",
    )


class KnowledgeAnswer(_Frozen):
    """Full answer envelope from ``search_knowledge``."""

    answer_status: str = Field(
        description=(
            "One of: ``answered`` (synthesis ran), ``no_results`` (retrieval "
            "returned nothing), ``insufficient_evidence`` (results too weak "
            "for grounded answer), ``synthesis_unavailable`` (LLM unreachable), "
            "or ``no_retrieval_needed`` (adaptive router classified the query "
            "as meta / out-of-scope and skipped retrieval — see the "
            "``router_*`` fields in ``metadata`` for the route rationale)."
        )
    )
    answer_text: str
    citations: list[str] = Field(description="Citation labels referenced by the answer.")
    sentence_citations: list[SentenceCitationView] = Field(
        default_factory=list,
        description=(
            "Per-sentence attribution parsed from the inline ``[citation_label]`` "
            "markers the synthesis preamble required. Empty ``citation_labels`` "
            "on a sentence means the model could not attribute that claim."
        ),
    )
    warnings: list[str] = Field(default_factory=list)
    metadata: KnowledgeAnswerMetadata


class KnowledgeAnswerDeep(_Frozen):
    """Deep-search answer envelope with structured graph context."""

    answer_status: str
    answer_text: str
    citations: list[str]
    sentence_citations: list[SentenceCitationView] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: KnowledgeAnswerMetadata
    graph_context: GraphContextData


class EvalGapTicketView(_Frozen):
    """One retrieval-eval gap ticket returned by ``list_eval_gaps``.

    Mirrors :class:`lxd.eval.gaps.GapTicket` — a human-reviewed artefact
    derived from a failing retrieval-eval case, never an auto-applied fix.
    """

    ticket_id: str = Field(description="BLAKE3 hash of the question and expected sources.")
    question: str
    expected_sources: list[str] = Field(default_factory=list)
    ranked_top: list[str] = Field(
        default_factory=list, description="Top-10 ranked source paths retrieval actually returned."
    )
    recall_at_10: float
    mrr_at_10: float
    gap_kind: Literal["missed_source", "weak_rank", "empty_results", "eval_warning"]
    notes: str
    created_at: str
    status: Literal["open", "closed"]


class LearningObjectivesView(_Frozen):
    """Bloom's-aligned objectives (mirrors ``lxd.agents.artefacts.LearningObjectives``)."""

    items: list[str] = Field(default_factory=list, description="Learning objective statements.")
    citations: list[str] = Field(
        default_factory=list, description="Citation labels grounding these objectives."
    )


class ModalityPlanView(_Frozen):
    """Recommended delivery modality (mirrors ``lxd.agents.artefacts.ModalityPlan``)."""

    text: str = Field(default="", description="Recommended modality and rationale.")
    citations: list[str] = Field(
        default_factory=list, description="Citation labels grounding this recommendation."
    )


class OutlineView(_Frozen):
    """Ordered module headings (mirrors ``lxd.agents.artefacts.Outline``)."""

    items: list[str] = Field(default_factory=list, description="Ordered outline headings.")
    citations: list[str] = Field(
        default_factory=list, description="Citation labels grounding the sequencing."
    )


class AssessmentBlueprintView(_Frozen):
    """Assessment items (mirrors ``lxd.agents.artefacts.AssessmentBlueprint``)."""

    items: list[str] = Field(default_factory=list, description="Assessment items.")
    citations: list[str] = Field(
        default_factory=list, description="Citation labels grounding the assessment design."
    )


class DesignArtefactBundleView(_Frozen):
    """Full output of ``design_learning`` (mirrors ``lxd.agents.artefacts.DesignArtefactBundle``)."""

    topic: str
    objectives: LearningObjectivesView
    modality_plan: ModalityPlanView
    outline: OutlineView
    assessment: AssessmentBlueprintView
    steps_completed: int = Field(
        description="How many of the agent's bounded steps actually ran (<= max_steps)."
    )
    warnings: list[str] = Field(default_factory=list)


class CritiqueResultView(_Frozen):
    """Output of ``critique_design`` (mirrors ``lxd.agents.artefacts.CritiqueResult``)."""

    overall_score: float = Field(
        description="Holistic 0.0-1.0 score, not necessarily the mean of dimension_scores."
    )
    dimension_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension 0.0-1.0 scores, e.g. objective_alignment, evidence_grounding.",
    )
    feedback: list[str] = Field(default_factory=list, description="Concise, actionable bullets.")
    citations: list[str] = Field(
        default_factory=list, description="Citation labels grounding the critique."
    )
    warnings: list[str] = Field(default_factory=list)
