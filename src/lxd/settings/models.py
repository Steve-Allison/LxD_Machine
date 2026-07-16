"""Define strongly typed runtime configuration models."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)


def _normalize_query_instruction(value: str | None) -> str | None:
    """Treat a blank query instruction as ``None`` so config consumers see a single absent shape."""
    if value is None:
        return None
    return value if value.strip() else None


def _validate_corpus_id_shape(value: str) -> str:
    """Enforce the slug shape ``^[a-z0-9][a-z0-9_-]{0,62}$`` for ``tenancy.corpus_id``.

    The slug must be safe for filesystem paths, LanceDB filter clauses, and
    SQLite row keys, so we reject anything that would force downstream
    quoting or normalisation.
    """
    if not value or len(value) > 63:
        raise ValueError("tenancy.corpus_id must be 1..63 characters")
    if not value[0].isalnum():
        raise ValueError("tenancy.corpus_id must start with an alphanumeric character")
    for ch in value:
        if not (ch.isalnum() or ch in {"_", "-"}):
            raise ValueError("tenancy.corpus_id may only contain [a-z0-9_-] characters")
        if ch.isalpha() and not ch.islower():
            raise ValueError("tenancy.corpus_id must be lowercase")
    return value


class OllamaConfig(BaseModel):
    """Configuration for connecting to the Ollama API."""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl


class ModelsConfig(BaseModel):
    """Model identifiers and embedding/rerank model settings."""

    model_config = ConfigDict(extra="forbid")

    embed: str
    embed_dims: int = Field(gt=0)
    embed_backend: Literal["ollama", "openai"] = "ollama"
    llm: str
    rerank: str
    llm_no_think: bool = False


class ChunkingConfig(BaseModel):
    """Document chunking strategy and tokenizer settings.

    ``contextual_summary_*`` settings control optional contextual
    retrieval (Anthropic-style chunk-level context preamble): when
    enabled, the ingest pipeline asks the local Ollama LLM to
    generate a 1-sentence summary of *what each chunk is about in the
    context of its document*, prepends it to the chunk text **before
    embedding only** (the stored chunk text stays clean for citation
    rendering), and caches the summary keyed on
    ``(chunk_hash, model)``. Default off — opt in by setting
    ``contextual_summary_enabled: true`` and re-running ingest.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    min_tokens: int = Field(ge=0)
    tokenizer_backend: str
    tokenizer_name: str
    contextual_summary_enabled: bool = False
    contextual_summary_model: str = "qwen3:14b"
    contextual_summary_temperature: float = Field(default=0.0, ge=0.0)
    contextual_summary_timeout_secs: int = Field(default=60, gt=0)
    contextual_summary_max_tokens: int = Field(default=80, gt=0)


class EmbeddingConfig(BaseModel):
    """Embedding client timeout, retry, and instruction settings."""

    model_config = ConfigDict(extra="forbid")

    timeout_secs: int = Field(gt=0)
    retry_attempts: int = Field(gt=0)
    retry_backoff: list[int] = Field(default_factory=list)
    query_instruction: Annotated[str | None, BeforeValidator(_normalize_query_instruction)] = None
    batch_size: int = Field(default=32, gt=0)
    max_workers: int = Field(default=4, gt=0)


class OpenAIEmbeddingConfig(BaseModel):
    """OpenAI embedding backend credentials and model options."""

    model_config = ConfigDict(extra="forbid")

    api_key_env: str = "OPENAI_API_KEY"
    model: str = "text-embedding-3-small"
    dims: int = Field(default=1536, gt=0)
    batch_size: int = Field(default=512, gt=0)
    max_workers: int = Field(default=8, gt=0)


class CorpusConfig(BaseModel):
    """Corpus file extension and scanning filters."""

    model_config = ConfigDict(extra="forbid")

    text_extensions: list[str]
    asset_extensions: list[str]
    ignore_names: list[str]
    min_text_file_bytes: int = Field(ge=0)


class AssetsConfig(BaseModel):
    """Asset ingestion toggles for registration and parent inference."""

    model_config = ConfigDict(extra="forbid")

    register_png: bool
    infer_docling_parent: bool


class OntologyConfig(BaseModel):
    """Ontology file inclusion and ignore filters."""

    model_config = ConfigDict(extra="forbid")

    include_globs: list[str]
    ignore_names: list[str]


class IngestBudget(BaseModel):
    """Per-run ingest cost ceiling.

    A runaway ``--full`` ingest against a large corpus can burn LLM-API
    spend before any error-class circuit-breaker trips, because relation
    extraction makes one LLM call per qualifying chunk regardless of
    success. The budget tracker counts those calls and aborts the run
    when the configured ceiling is reached, so an over-large corpus or
    a misconfigured ``min_entity_mentions`` cannot keep spending.

    Currently tracks **only LLM call count** (relation extraction during
    ingest). Embedding-token tracking and per-call cost estimation are
    not yet wired through and remain a Tier 7 backlog item; for embedding
    spend, set conservative limits via your provider's account dashboard.
    """

    model_config = ConfigDict(extra="forbid")

    max_llm_calls_per_run: int | None = None


class RetrievalConfig(BaseModel):
    """Dense retrieval and fusion weighting parameters."""

    model_config = ConfigDict(extra="forbid")

    dense_top_k: int = Field(gt=0)
    rerank_top_k: int = Field(gt=0)
    lexical_fusion_weight: float = Field(default=2.0, ge=0.0)
    relation_fusion_weight: float = Field(default=1.0, ge=0.0)
    centrality_fusion_weight: float = Field(default=1.0, ge=0.0)
    community_diversity_enabled: bool = True
    hyde_enabled: bool = False
    hyde_model: str = "qwen3:14b"
    hyde_temperature: float = Field(default=0.0, ge=0.0)
    hyde_timeout_secs: int = Field(default=30, gt=0)
    hyde_max_tokens: int = Field(default=200, gt=0)


class RerankerLaunchConfig(BaseModel):
    """Auto-start settings for llama.cpp reranker service."""

    model_config = ConfigDict(extra="forbid")

    auto_start: bool = False
    executable: str = "llama-server"
    model_source: Literal["ollama_blob", "model_path"] = "ollama_blob"
    model_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8012, gt=0, le=65535)
    startup_timeout_secs: int = Field(default=120, gt=0)
    extra_args: list[str] = Field(default_factory=list)


class RerankerConfig(BaseModel):
    """Reranker backend connectivity and launch settings."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["llama_cpp", "colbert"] = "llama_cpp"
    url: HttpUrl | None = None
    endpoint: str = "/v1/rerank"
    timeout_secs: int = Field(default=30, gt=0)
    launch: RerankerLaunchConfig | None = None
    colbert_model: str = Field(
        default="BAAI/bge-m3",
        description=(
            "HuggingFace model id used by the ``colbert`` backend for "
            "late-interaction (multi-vector) reranking. ``BAAI/bge-m3`` "
            "produces token-level vectors that the reranker scores via "
            "MaxSim — the same mechanism ColBERT v2 uses."
        ),
    )
    colbert_max_length: int = Field(
        default=512,
        gt=0,
        le=8192,
        description="Token cap per document when encoding for late-interaction scoring.",
    )

    @model_validator(mode="after")
    def _validate_launch_contract(self) -> Self:
        if self.launch is None or not self.launch.auto_start:
            return self
        if self.url is None:
            raise ValueError("reranker.launch.auto_start requires reranker.url to be configured.")
        if self.url.host != self.launch.host:
            raise ValueError("reranker.launch.host must match reranker.url host.")
        if self.url.port != self.launch.port:
            raise ValueError("reranker.launch.port must match reranker.url port.")
        if self.launch.model_source == "model_path" and self.launch.model_path is None:
            raise ValueError(
                "reranker.launch.model_path must be set when reranker.launch.model_source=model_path."
            )
        return self


class ExpansionConfig(BaseModel):
    """Ontology-based query expansion behavior."""

    model_config = ConfigDict(extra="forbid")

    hops: int = Field(ge=0)
    max_terms: int = Field(gt=0)


class RelationExtractionConfig(BaseModel):
    """Relation extraction backend and generation controls."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["openai", "ollama"] = "openai"
    fallback_backend: Literal["ollama", "none"] = "ollama"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "qwen3:14b"
    min_entity_mentions: int = Field(default=2, ge=1)
    max_relations_per_chunk: int = Field(default=15, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    timeout_secs: int = Field(default=30, gt=0)
    max_concurrent: int = Field(default=50, gt=0)
    sub_batch_size: int = Field(default=500, gt=0)


class SynthesisConfig(BaseModel):
    """Answer synthesis limits and generation settings."""

    model_config = ConfigDict(extra="forbid")

    max_chunks: int = Field(gt=0)
    timeout_secs: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    max_tokens: int = Field(gt=0)


class AdaptiveRetrievalConfig(BaseModel):
    """Adaptive (Self-RAG / CRAG-style) retrieval router controls.

    The router runs a cheap LLM call before retrieval. It returns a
    :class:`lxd.retrieval.router.QueryRoute` that the pipeline uses to:

      - skip retrieval entirely for meta queries (e.g. "hello", "what
        can you do?", "how does this work?") — saves cost and returns
        a graceful answer
      - widen retrieval breadth for broad survey queries
      - tighten it for narrow factual lookups

    Mandatory feature; disabling the router is not a config knob. The
    router degrades gracefully — any failure of the LLM call leaves
    the pipeline on a sensible default route (``retrieve=True``,
    breadth=``standard``).
    """

    model_config = ConfigDict(extra="forbid")

    router_backend: Literal["openai", "ollama"] = Field(
        default="openai",
        description="LLM backend used for the router classification.",
    )
    router_model: str = Field(
        default="gpt-4o-mini",
        description="Chat model used for the router classification.",
    )
    router_timeout_secs: float = Field(
        default=15.0,
        gt=0.0,
        description="Hard timeout for the router LLM call (seconds).",
    )
    narrow_dense_top_k: int = Field(
        default=8,
        gt=0,
        le=200,
        description=(
            "Dense retrieval depth for narrow queries. Smaller than "
            "``retrieval.dense_top_k`` so synthesis sees a focused set."
        ),
    )
    broad_dense_top_k: int = Field(
        default=40,
        gt=0,
        le=200,
        description=(
            "Dense retrieval depth for broad / survey queries. Larger than "
            "``retrieval.dense_top_k`` so synthesis can cover more ground."
        ),
    )


class KnowledgeGraphConfig(BaseModel):
    """Knowledge graph build and query settings."""

    model_config = ConfigDict(extra="forbid")

    min_relation_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Community detection
    community_resolution: float = Field(default=1.0, gt=0.0)
    community_algorithm: Literal["leiden", "louvain"] = "louvain"
    community_seed: int = Field(default=42)

    # Entity profiles
    entity_summary_max_chunks: int = Field(default=20, gt=0)
    entity_embedding_min_mentions: int = Field(default=3, ge=1)

    # Claim extraction
    claim_extraction_backend: Literal["openai", "ollama"] = "openai"
    claim_extraction_model: str = "gpt-4o-mini"
    claim_extraction_fallback_model: str = "qwen3:14b"
    claim_extraction_min_mentions: int = Field(default=1, ge=1)
    claim_max_per_chunk: int = Field(default=10, gt=0)
    claim_extraction_timeout_secs: int = Field(default=90, gt=0)
    claim_extraction_temperature: float = Field(default=0.0, ge=0.0)
    claim_extraction_max_concurrent: int = Field(default=50, gt=0)
    claim_extraction_sub_batch_size: int = Field(default=500, gt=0)

    # LLM enrichment
    llm_enrichment_backend: Literal["openai", "ollama"] = "openai"
    llm_enrichment_model: str = "gpt-4o-mini"
    llm_enrichment_fallback_model: str = "qwen3:14b"
    llm_enrichment_temperature: float = Field(default=0.1, ge=0.0)
    llm_enrichment_timeout_secs: int = Field(default=30, gt=0)

    # Query routing
    multi_hop_max: int = Field(default=3, ge=1, le=5)
    max_entity_context: int = Field(default=5, gt=0)
    max_community_context: int = Field(default=3, gt=0)
    max_claim_context: int = Field(default=10, gt=0)
    max_graph_context_tokens: int = Field(default=1500, gt=0)


class MCPConfig(BaseModel):
    """MCP server identity and runtime behaviour."""

    model_config = ConfigDict(extra="forbid")

    server_name: str
    version: str
    async_tools_enabled: bool = True
    tool_timeout_secs: float = Field(default=60.0, ge=0.0)
    synthesis_backend: Literal["server_llm", "client_sampling"] = Field(
        default="server_llm",
        description=(
            "Which LLM answers the synthesis step for search_knowledge / "
            "search_knowledge_deep. ``server_llm`` (default) calls the "
            "server's own Ollama model — trust and cost boundary stays "
            "server-side. ``client_sampling`` delegates to MCP "
            "``Context.sample`` so the client's own LLM runs the "
            "synthesis (client picks model, client pays for tokens); the "
            "server falls back to ``server_llm`` automatically for any "
            "client that has not advertised the sampling capability."
        ),
    )


class LoggingConfig(BaseModel):
    """Runtime logging level and output format."""

    model_config = ConfigDict(extra="forbid")

    level: str
    format: Literal["json", "console"] = "json"
    sample_rate: int = Field(default=1, ge=1)
    sampled_event_names: list[str] = Field(
        default_factory=lambda: [
            "embedding_cache_hit",
            "embedding_cache_miss",
            "chunk_processed",
            "mention_detected",
        ]
    )


class PathsConfig(BaseModel):
    """Filesystem paths for corpus, ontology, and data."""

    model_config = ConfigDict(extra="forbid")

    corpus_path: Path
    ontology_path: Path
    data_path: Path


class TenancyConfig(BaseModel):
    """Multi-tenant corpus identity.

    The single-tenant default keeps the existing single-workspace shape: one
    SQLite + LanceDB store under ``paths.data_path``. Setting ``corpus_id``
    marks every persisted artefact (future migration) with a stable tenant
    tag that downstream tooling (CQRS replicas, cross-corpus reporting) can
    filter on.

    Attributes:
        corpus_id: Slug-style identifier, ``"default"`` when unspecified.
            Must match ``^[a-z0-9][a-z0-9_-]{0,62}$`` so it is safe to use
            in filesystem paths and LanceDB filter clauses.
    """

    model_config = ConfigDict(extra="forbid")

    corpus_id: Annotated[str, AfterValidator(_validate_corpus_id_shape)] = "default"


class RuntimeConfig(BaseModel):
    """Top-level runtime configuration for the application."""

    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig
    tenancy: TenancyConfig = Field(default_factory=TenancyConfig)
    ollama: OllamaConfig
    models: ModelsConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    corpus: CorpusConfig
    assets: AssetsConfig
    ontology: OntologyConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    expansion: ExpansionConfig
    relation_extraction: RelationExtractionConfig = Field(default_factory=RelationExtractionConfig)
    synthesis: SynthesisConfig
    adaptive_retrieval: AdaptiveRetrievalConfig = Field(default_factory=AdaptiveRetrievalConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    ingest_budget: IngestBudget = Field(default_factory=IngestBudget)
    mcp: MCPConfig
    logging: LoggingConfig
    openai: OpenAIEmbeddingConfig | None = None

    @model_validator(mode="after")
    def _validate_openai_backend(self) -> Self:
        if self.models.embed_backend == "openai" and self.openai is None:
            raise ValueError("models.embed_backend=openai requires an [openai] config section.")
        if (
            self.models.embed_backend == "openai"
            and self.openai is not None
            and self.models.embed_dims != self.openai.dims
        ):
            raise ValueError(
                "models.embed_dims must match openai.dims when models.embed_backend=openai."
            )
        return self
