# LxD Machine - Configuration Design

## 1. Design Principle

Every tuneable parameter lives in configuration.

The system uses file-only runtime config.

- `config.yaml`: canonical portable default config
- `config.{profile}.yaml`: optional machine-specific variants selected explicitly via `--profile`

## 2. Supported Machines

- `m1max`: MacBook Pro M1 Max
- `m4mini`: Mac Mini M4

Both profiles share the same structure.

Model selection in this document assumes:

- the corpus is English-only
- dense retrieval is the primary retrieval path
- baseline reranking is enabled in shipped profiles through a dedicated `llama.cpp` server, but query must fall back to dense-only if the reranker is unavailable at runtime
- the answer model is optimized for grounded long-context synthesis rather than multilingual breadth
- sparse/FTS retrieval is not part of the V1 baseline

## 3. Canonical Runtime Config

The repo-local default runtime contract is `config.yaml`.

Profile variants such as `config.m1max.yaml` and `config.m4mini.yaml` are selected explicitly with `--profile`.

All runtime paths must live inside the YAML config under `paths`.

## 4. Runtime Config Shape

Both machine profiles must define the following sections:

- `paths`
- `ollama`
- `models`
- `chunking`
- `embedding`
- `corpus`
- `assets`
- `ontology`
- `retrieval`
- `reranker`
- `expansion`
- `synthesis`
- `mcp`
- `logging`

### Required `paths` settings

```yaml
paths:
  corpus_path: Knowledge_Base
  ontology_path: Yamls
  data_path: data
```

Relative paths are resolved from the directory containing the selected config file.

### Required `reranker` settings

```yaml
reranker:
  backend: llama_cpp
  url: http://127.0.0.1:8012
  endpoint: /v1/rerank
  timeout_secs: 30
  launch:
    auto_start: true
    executable: llama-server
    model_source: ollama_blob
    host: 127.0.0.1
    port: 8012
    startup_timeout_secs: 120
    extra_args: []
```

`llama.cpp` reranking requires `llama-server` with a reranker model and reranking enabled. The documented aliases include `/reranking`, `/rerank`, `/v1/rerank`, and `/v1/reranking`; the checked-in default is `/v1/rerank`.

Autostart contract:

- `reranker.launch.auto_start: true` means query and eval may launch `llama-server` automatically when the configured endpoint is not yet listening
- `reranker.launch.model_source: ollama_blob` means the runtime resolves the real local model blob via `ollama show --modelfile <models.rerank>` rather than hard-coding a home-directory path
- the launch command must include `--embedding --pooling rank --reranking` for reranker models served by `llama.cpp`
- `reranker.launch.host` and `reranker.launch.port` must match the configured `reranker.url`
- runtime PID and log files must be namespaced by host, port, and model alias so multiple reranker configs do not collide

### Required `corpus` settings

```yaml
corpus:
  text_extensions: [".md", ".docling.json"]
  asset_extensions: [".png"]
  ignore_names: [".DS_Store"]
  min_text_file_bytes: 1
```

### Required `assets` settings

```yaml
assets:
  register_png: true
  infer_docling_parent: true
```

### Required `ontology` settings

```yaml
ontology:
  include_globs: ["**/*.yaml"]
  ignore_names: []
```

### Required `chunking` settings

```yaml
chunking:
  strategy: hybrid_docling
  chunk_size: 300
  chunk_overlap: 60
  min_tokens: 20
  tokenizer_backend: tiktoken
  tokenizer_name: cl100k_base
```

Tokenization must be explicit in config. No hidden tokenizer defaults are allowed.

`chunk_size` and `chunk_overlap` are initial chunker targets, not a trusted embedder safety contract.

For embedding safety:

- the system must call the Ollama embed API with `truncate=false`
- the live embedder response is authoritative
- if the embedder rejects a chunk as oversize, ingest must split that chunk again on text boundaries and retry until accepted or until no further split is possible
- `tiktoken` counts are advisory for initial chunk construction only; they are not treated as proof that a chunk is safe for a non-OpenAI embedder

## 5. Example `config.m1max.yaml`

This profile is the **best-balance** configuration: fast ingest, modest vector size, and strong grounded answer quality on local hardware.

```yaml
paths:
  corpus_path: Knowledge_Base
  ontology_path: Yamls
  data_path: data

ollama:
  url: http://localhost:11434

models:
  embed: nomic-embed-text
  embed_dims: 768
  llm: mistral-small3.1
  rerank: dengcao/Qwen3-Reranker-4B:Q4_K_M
  llm_no_think: true

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
  text_extensions: [".md", ".docling.json"]
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
  backend: llama_cpp
  url: http://127.0.0.1:8012
  endpoint: /v1/rerank
  timeout_secs: 30
  launch:
    auto_start: true
    executable: llama-server
    model_source: ollama_blob
    host: 127.0.0.1
    port: 8012
    startup_timeout_secs: 120
    extra_args: []

expansion:
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
```

## 6. Example `config.m4mini.yaml`

This profile is the **best-quality** configuration: higher-quality reranking and stronger synthesis at the cost of more RAM and lower throughput.

```yaml
paths:
  corpus_path: Knowledge_Base
  ontology_path: Yamls
  data_path: data

ollama:
  url: http://localhost:11434

models:
  embed: nomic-embed-text
  embed_dims: 768
  llm: qwen3:30b-a3b
  rerank: dengcao/Qwen3-Reranker-4B:Q4_K_M
  llm_no_think: true

chunking:
  strategy: hierarchical_docling
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
  text_extensions: [".md", ".docling.json"]
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
  backend: llama_cpp
  url: http://127.0.0.1:8012
  endpoint: /v1/rerank
  timeout_secs: 30
  launch:
    auto_start: true
    executable: llama-server
    model_source: ollama_blob
    host: 127.0.0.1
    port: 8012
    startup_timeout_secs: 120
    extra_args: []

expansion:
  hops: 1
  max_terms: 12

synthesis:
  max_chunks: 12
  timeout_secs: 90
  temperature: 0.1
  max_tokens: 2000

mcp:
  server_name: lxd-machine
  version: 0.1.0

logging:
  level: INFO
  format: json
```

### Recommended model policy

- Default English embedder in the live local runtime: `nomic-embed-text`
- Default vector width: `768`
- Storage reduction option after benchmark confirmation: `512` or `256` dimensions using Matryoshka-capable embedders only
- Default reranker model alias for the `llama.cpp` server: `dengcao/Qwen3-Reranker-4B:Q4_K_M`
- Best-balance synthesis model in the current local runtime: `mistral-small3.1`
- Best-quality synthesis model in the current local runtime: `qwen3:30b-a3b`

If the `llama.cpp` server uses local aliases that differ from model-card names, the checked-in config must use the real runnable server alias, not the upstream registry identifier.

## 7. What Changing Config Requires

| Change | Re-ingest required? |
|---|---|
| Change `models.llm` | No |
| Change `models.rerank` | No |
| Change `models.embed` or `models.embed_dims` | Full rebuild of searchable chunk rows |
| Change `chunking.*` | Full rebuild of searchable chunk rows |
| Change `chunking.tokenizer_*` | Full rebuild of searchable chunk rows |
| Change `corpus.text_extensions` | Full rescan and rebuild |
| Change `corpus.asset_extensions` | Full rescan of asset rows |
| Change `assets.*` | Asset relink/re-registration |
| Change `ontology.include_globs` | Full ontology rebuild and mention rebuild |
| Change `paths.corpus_path` | Full rebuild |
| Change `paths.ontology_path` | Full ontology rebuild and mention rebuild |
| Change `paths.data_path` | No, if the data directory moves with it |

## 8. Startup Validation

Validation must fail fast when:

- `paths.corpus_path` does not exist
- `paths.ontology_path` does not exist
- `chunk_overlap >= chunk_size`
- `corpus.text_extensions` and `corpus.asset_extensions` overlap
- `models.embed_dims` is not a valid output width supported by the configured embedder
- the configured embedder cannot complete an embedding request with `truncate=false`
- `reranker.launch.auto_start = true`, but `llama-server` is not present on `PATH`
- `reranker.launch.auto_start = true` and `reranker.launch.model_source = ollama_blob`, but `ollama show --modelfile <models.rerank>` does not resolve to a local blob path

Validation must not assume that `chunking.chunk_size` is automatically safe for the configured embedder.

## 9. Config Snapshot Rule

At the end of a successful ingest, the system snapshots the settings that affect stored state:

- `models.embed`
- `models.embed_dims`
- `chunking`
- `corpus`
- `assets`
- `ontology`

`status` must warn when the current config no longer matches the committed snapshot.
