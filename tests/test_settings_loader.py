from __future__ import annotations

from pathlib import Path

from lxd.settings.loader import load_runtime_config, resolve_repo_root


def test_load_runtime_config_uses_default_config_yaml() -> None:
    repo_root = resolve_repo_root(Path.cwd())
    config, config_path = load_runtime_config(repo_root)

    assert config_path == repo_root / "config.yaml"
    assert config.paths.corpus_path == repo_root / "Knowledge_Base"
    assert config.paths.ontology_path == repo_root / "Yamls"
    assert config.paths.data_path == repo_root / "data"


def test_load_runtime_config_resolves_relative_paths_from_selected_file(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_file = repo_root / "portable.yaml"
    config_file.write_text(
        """
paths:
  corpus_path: corpus
  ontology_path: ontology
  data_path: data
ollama:
  url: http://localhost:11434
models:
  embed: nomic-embed-text
  embed_dims: 768
  llm: mistral-small3.1
  rerank: dengcao/Qwen3-Reranker-0.6B:F16
chunking:
  strategy: hybrid_docling
  chunk_size: 300
  chunk_overlap: 60
  min_tokens: 20
  tokenizer_backend: tiktoken
  tokenizer_name: cl100k_base
embedding:
  timeout_secs: 120
  retry_attempts: 3
  retry_backoff: [2, 4, 8]
corpus:
  text_extensions: [".md"]
  asset_extensions: [".png"]
  ignore_names: [".DS_Store"]
  min_text_file_bytes: 1
assets:
  register_png: true
  infer_docling_parent: true
ontology:
  include_globs: ["**/*.yaml"]
  ignore_names: []
retrieval:
  dense_top_k: 20
  rerank_top_k: 20
  lexical_fusion_weight: 2.0
reranker:
  enabled: true
  backend: llama_cpp
  url: http://127.0.0.1:8012
  endpoint: /v1/rerank
  timeout_secs: 30
  launch:
    auto_start: true
    executable: llama-server
    model_source: model_path
    model_path: models/reranker.gguf
    host: 127.0.0.1
    port: 8012
    startup_timeout_secs: 30
    extra_args: []
expansion:
  enabled: false
  hops: 1
  max_terms: 12
synthesis:
  max_chunks: 8
  timeout_secs: 60
  temperature: 0.1
  max_tokens: 1500
mcp:
  server_name: lxd-machine
  version: 0.1.0
logging:
  level: INFO
  format: json
""".strip(),
        encoding="utf-8",
    )

    config, resolved = load_runtime_config(repo_root, config_path=config_file)

    assert resolved == config_file
    assert config.paths.corpus_path == repo_root / "corpus"
    assert config.paths.ontology_path == repo_root / "ontology"
    assert config.paths.data_path == repo_root / "data"
    assert config.reranker.launch is not None
    assert config.reranker.launch.model_path == repo_root / "models" / "reranker.gguf"


def test_resolve_repo_root_finds_project_root() -> None:
    repo_root = resolve_repo_root(Path.cwd())
    nested = repo_root / "src" / "lxd"

    resolved = resolve_repo_root(nested)

    assert resolved == repo_root
