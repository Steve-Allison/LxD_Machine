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
- `tenancy` *(optional; defaults to single-tenant `"default"`)*

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
  backend: llama_cpp             # or: colbert
  url: http://127.0.0.1:8012
  endpoint: /v1/rerank
  timeout_secs: 30
  colbert_model: BAAI/bge-m3     # only consulted when backend=colbert
  colbert_max_length: 512
  launch:
    auto_start: true
    executable: llama-server
    model_source: ollama_blob
    host: 127.0.0.1
    port: 8012
    startup_timeout_secs: 120
    extra_args: []
```

Two backends are supported:

- `llama_cpp` — cross-encoder reranker served by an external `llama-server` process; the documented endpoint aliases include `/reranking`, `/rerank`, `/v1/rerank`, and `/v1/reranking`; the checked-in default is `/v1/rerank`.
- `colbert` — in-process late-interaction (multi-vector) reranker via `retrieval/colbert_reranker.py`; scores documents by MaxSim over token-level vectors, using the HuggingFace model named in `colbert_model` (default `BAAI/bge-m3`) with a per-document token cap of `colbert_max_length`. When `backend=colbert` the `url`, `endpoint`, and `launch` block are ignored.

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
  # Optional contextual-summary preamble (Anthropic-style chunk-level
  # context). When enabled the pipeline asks the local Ollama LLM for a
  # 1-sentence summary of what each chunk is about within its document
  # and prepends it to the chunk text before embedding only; the stored
  # chunk text stays clean for citation rendering. Cache is keyed on
  # (chunk_hash, model). Defaults off — opt in and re-ingest.
  contextual_summary_enabled: false
  contextual_summary_model: qwen3:14b
  contextual_summary_temperature: 0.0
  contextual_summary_timeout_secs: 60
  contextual_summary_max_tokens: 80
```

Tokenization must be explicit in config. No hidden tokenizer defaults are allowed.

### Required `embedding` settings

```yaml
embedding:
  timeout_secs: 120
  retry_attempts: 3
  retry_backoff: [2, 4, 8]
  batch_size: 32        # texts per backend request (Ollama batch / OpenAI batch)
  max_workers: 4        # concurrent batch workers for OpenAI; 1 for Ollama
  query_instruction: null
```

Batching rules:

- the ingest pipeline always calls `embed_texts_batched`, which uses the
  backend's native batch API
- on an Ollama `input length exceeds the context length` error the whole
  batch falls back to per-text embedding to isolate the oversize input
- OpenAI batches are dispatched concurrently via a thread pool of
  `max_workers`; Ollama defaults to a single worker to keep the local
  model warm

### Optional `mcp` settings

```yaml
mcp:
  server_name: lxd-machine
  version: 0.1.0
  async_tools_enabled: true
  tool_timeout_secs: 60.0            # 0 disables the hard timeout (not recommended)
  synthesis_backend: server_llm      # or: client_sampling
```

All MCP tools are `async def`; synchronous bodies run in a worker thread
under `lxd.mcp.async_runtime.run_tool`, which enforces `tool_timeout_secs`
via `anyio.fail_after`. Timeouts and exceptions emit structured
`mcp.tool.timeout` / `mcp.tool.error` log events.

`synthesis_backend` chooses where the `search_knowledge` /
`search_knowledge_deep` synthesis call runs:

- `server_llm` (default) — the server calls its own Ollama model. Trust
  and cost boundary stays server-side; the server owns the API keys.
- `client_sampling` — the server dispatches the synthesis prompt via
  `fastmcp.Context.sample` so the connected client's own LLM answers
  (client picks the model, client pays for tokens). The server
  transparently falls back to `server_llm` when the connected client
  has not advertised the sampling capability or the sampling call
  fails, and surfaces the reason as a single warning on the resulting
  envelope's `warnings` list plus a streamed `notifications/message`
  notice.

### Optional `tenancy` settings

```yaml
tenancy:
  corpus_id: default
```

`corpus_id` must match `^[a-z0-9][a-z0-9_-]{0,62}$`. It is stamped onto
persistent `llm_jobs` rows and is reserved as a future multi-tenancy
filter key across the store.

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

## 10. Config Digest And Lock File

On every bootstrap, the runtime computes `config_digest = blake3(json.dumps(config.model_dump(mode="json"), sort_keys=True))` and reconciles it against `<paths.data_path>/config.lock`:

- first run: seed the lock file with the current digest
- subsequent runs: on mismatch, emit a `config.lock.mismatch` warning with
  both digests; never overwrite automatically
- deleting `config.lock` is the supported way to reseed

The digest covers every field in `RuntimeConfig`, including `tenancy`,
so infrastructure changes (tenancy slug, etc.) surface as drift warnings.
