from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embed: str
    embed_dims: int = Field(gt=0)
    embed_backend: Literal["ollama", "openai"] = "ollama"
    llm: str
    rerank: str
    llm_no_think: bool = False


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    min_tokens: int = Field(ge=0)
    tokenizer_backend: str
    tokenizer_name: str


class EmbeddingConfig(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    api_key_env: str = "OPENAI_API_KEY"
    model: str = "text-embedding-3-small"
    dims: int = Field(default=1536, gt=0)
    batch_size: int = Field(default=512, gt=0)
    max_workers: int = Field(default=8, gt=0)


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_extensions: list[str]
    asset_extensions: list[str]
    ignore_names: list[str]
    min_text_file_bytes: int = Field(ge=0)


class AssetsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    register_png: bool
    infer_docling_parent: bool


class OntologyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_globs: list[str]
    ignore_names: list[str]


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense_top_k: int = Field(gt=0)
    rerank_top_k: int = Field(gt=0)
    lexical_fusion_weight: float = Field(default=2.0, ge=0.0)


class RerankerConfig(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    hops: int = Field(ge=0)
    max_terms: int = Field(gt=0)


class SynthesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chunks: int = Field(gt=0)
    timeout_secs: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    max_tokens: int = Field(gt=0)


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_name: str
    version: str


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str
    format: Literal["json", "console"] = "json"


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_path: Path
    ontology_path: Path
    data_path: Path


class RuntimeConfig(BaseModel):
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
    synthesis: SynthesisConfig
    mcp: MCPConfig
    logging: LoggingConfig
    openai: OpenAIEmbeddingConfig | None = None

    @model_validator(mode="after")
    def _validate_openai_backend(self) -> RuntimeConfig:
        if self.models.embed_backend == "openai" and self.openai is None:
            raise ValueError(
                "models.embed_backend=openai requires an [openai] config section."
            )
        if (
            self.models.embed_backend == "openai"
            and self.openai is not None
            and self.models.embed_dims != self.openai.dims
        ):
            raise ValueError(
                "models.embed_dims must match openai.dims when models.embed_backend=openai."
            )
        return self
