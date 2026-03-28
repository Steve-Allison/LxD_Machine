"""Define strongly typed runtime configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


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
    """Document chunking strategy and tokenizer settings."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    min_tokens: int = Field(ge=0)
    tokenizer_backend: str
    tokenizer_name: str


class EmbeddingConfig(BaseModel):
    """Embedding client timeout, retry, and instruction settings."""

    model_config = ConfigDict(extra="forbid")

    timeout_secs: int = Field(gt=0)
    retry_attempts: int = Field(gt=0)
    retry_backoff: list[int] = Field(default_factory=list)
    query_instruction: str | None = None

    @model_validator(mode="after")
    def _normalize_query_instruction(self) -> EmbeddingConfig:
        if self.query_instruction is not None and not self.query_instruction.strip():
            self.query_instruction = None
        return self


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


class RetrievalConfig(BaseModel):
    """Dense retrieval and fusion weighting parameters."""

    model_config = ConfigDict(extra="forbid")

    dense_top_k: int = Field(gt=0)
    rerank_top_k: int = Field(gt=0)
    lexical_fusion_weight: float = Field(default=2.0, ge=0.0)
    relation_fusion_weight: float = Field(default=1.0, ge=0.0)


class RerankerConfig(BaseModel):
    """Reranker backend connectivity and launch settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    backend: Literal["llama_cpp", "none"]
    url: HttpUrl | None = None
    endpoint: str = "/v1/rerank"
    timeout_secs: int = Field(default=30, gt=0)
    launch: RerankerLaunchConfig | None = None

    @model_validator(mode="after")
    def _validate_launch_contract(self) -> RerankerConfig:
        if self.launch is None or not self.launch.auto_start:
            return self
        if self.backend != "llama_cpp":
            raise ValueError("reranker.launch.auto_start requires reranker.backend=llama_cpp.")
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


class ExpansionConfig(BaseModel):
    """Ontology-based query expansion behavior."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
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


class SynthesisConfig(BaseModel):
    """Answer synthesis limits and generation settings."""

    model_config = ConfigDict(extra="forbid")

    max_chunks: int = Field(gt=0)
    timeout_secs: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    max_tokens: int = Field(gt=0)


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


class MCPConfig(BaseModel):
    """MCP server identity configuration."""

    model_config = ConfigDict(extra="forbid")

    server_name: str
    version: str


class LoggingConfig(BaseModel):
    """Runtime logging level and output format."""

    model_config = ConfigDict(extra="forbid")

    level: str
    format: Literal["json", "console"] = "json"


class PathsConfig(BaseModel):
    """Filesystem paths for corpus, ontology, and data."""

    model_config = ConfigDict(extra="forbid")

    corpus_path: Path
    ontology_path: Path
    data_path: Path


class RuntimeConfig(BaseModel):
    """Top-level runtime configuration for the application."""

    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig
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
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    mcp: MCPConfig
    logging: LoggingConfig
    openai: OpenAIEmbeddingConfig | None = None

    @model_validator(mode="after")
    def _validate_openai_backend(self) -> RuntimeConfig:
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
