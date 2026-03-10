from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, HttpUrl, model_validator


class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embed: str
    embed_dims: int
    llm: str
    rerank: str
    llm_no_think: bool = False


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    chunk_size: int
    chunk_overlap: int
    min_tokens: int
    tokenizer_backend: str
    tokenizer_name: str


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_secs: int
    retry_attempts: int
    retry_backoff: list[int]


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_extensions: list[str]
    asset_extensions: list[str]
    ignore_names: list[str]
    min_text_file_bytes: int


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

    dense_top_k: int
    rerank_top_k: int
    lexical_fusion_weight: float = 2.0


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    backend: Literal["llama_cpp", "none"]
    url: HttpUrl | None = None
    endpoint: str = "/v1/rerank"
    timeout_secs: int = 30
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
    port: int = 8012
    startup_timeout_secs: int = 120
    extra_args: list[str] = []


class ExpansionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    hops: int
    max_terms: int


class SynthesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chunks: int
    timeout_secs: int
    temperature: float
    max_tokens: int


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_name: str
    version: str


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str
    format: str


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
