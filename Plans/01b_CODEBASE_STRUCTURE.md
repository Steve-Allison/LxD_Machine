# LxD Machine - Codebase Structure

## 1. Architectural Style

The implementation must use a **layered, pipeline-first architecture**.

Rules:

- domain rules must be separable from IO and framework code
- CLI and MCP are adapters, not business-logic owners
- ingest, retrieval, ontology, and synthesis orchestration live in application modules
- storage backends are infrastructure modules behind narrow interfaces
- do not use catch-all utility modules

The design goal is:

- deterministic ingest behavior
- explicit state transitions
- thin adapters
- testable pure logic where possible
- low coupling between runtime surfaces

## 2. Canonical Package Layout

The Python package root is:

```text
src/lxd/
```

The canonical module layout is:

```text
src/lxd/
  app/
    bootstrap.py              # AppContext, bootstrap_app, config-digest + config.lock reconciliation
    status.py                 # Committed status snapshot + config_drift_warnings
  settings/
    models.py                 # Pydantic v2 config models (incl. tenancy, observability)
    loader.py
  domain/
    ids.py
    citations.py
    status.py
  net/
    http.py                   # Shared httpx.Client / httpx.AsyncClient factories
  ontology/
    loader.py
    inventory.py
    graph.py
    entity_graph.py           # Combined ontology + corpus relation graph + centrality
    communities.py            # Louvain (NetworkX) / Leiden (optional graspologic)
    evidence.py               # Canonical relation dedup + evidence provenance
    profiles.py               # Entity profiles + community reports (+ LLM enrichment)
    matcher.py                # Aho-Corasick matcher + normalization
    normalization.py
    schema_models.py          # Pydantic ontology schema (opt-in structural validation)
  ingest/
    scanner.py
    diff.py
    markdown.py
    docling.py
    wiki_metadata.py          # Frontmatter (**Sources**:, [[slug]]) parser
    wiki_relations.py
    chunking.py
    contextual_chunker.py     # Optional Anthropic-style per-chunk context preamble
    assets.py
    mentions.py
    relations.py              # LLM-based relation extraction
    claims.py                 # LLM-based claim extraction
    embedder.py               # Batched Ollama / OpenAI embedding with context-aware retry
    embedding_cache.py        # Content-addressed (chunk_hash, model, dims) cache in LanceDB
    error_classification.py   # TRANSIENT / DATA / SYSTEMIC classifier + circuit breaker
    budget.py                 # Per-run LLM-spend ceiling
    llm_client.py             # Shared sync/async OpenAI + Ollama-via-OpenAI-compat clients + Batch API helpers
    pipeline/                 # Sequential per-source orchestrator subpackage — NO re-export façade
      __init__.py
      orchestrator.py         # run_ingest, build_ingest_plan, IngestPlan, persist+commit loop
      sources.py              # Per-source extract → chunk → embed → assemble records
      embed.py                # Embedding cache + contextual augmentation + context refinement
      moves.py                # Move detection, unchanged-source skip, document_id resolution, chunk cloning
  retrieval/
    dense.py
    rerank.py                 # llama_cpp HTTP + in-process ColBERT backends
    colbert_reranker.py       # In-process late-interaction (multi-vector) reranker
    hyde.py                   # Hypothetical Document Embeddings (HyDE) query rewriter
    router.py                 # Adaptive router (retrieve? / breadth: narrow|standard|broad)
    expansion.py              # Ontology + entity-embedding-neighbour expansion
    graph_routing.py          # Graph context augmentation for synthesis
    query_pipeline.py         # Retrieval orchestrator; PhaseCallback / NoticeCallback types
    eval.py                   # Retrieval-quality harness (Recall@10, MRR@10)
  synthesis/
    answering.py              # synthesize_answer (Ollama or sampler-driven) + streaming
    citation_alignment.py     # Sentence-level attribution parser
    sampler.py                # SamplerRequest / Sampler / SamplerFailure — client-sampling seam
  stores/
    schema.py                 # Numbered migrations + ensure_schema; PRAGMA user_version
    _base_ddl.py              # Authoritative CREATE TABLE / CREATE INDEX baseline DDL
    _sqlite_rows.py           # Row-to-record adapters (module-private to stores)
    sqlite/                   # Query/upsert subpackage — NO re-export façade
      __init__.py
      connection.py           # connect_sqlite, build_store_paths, initialize_schema
      _pool.py                # Per-thread schema-initialised connection pool (MCP request path)
      runs.py                 # Ingest-run lifecycle (begin / progress / finish)
      manifest.py             # corpus_manifest upsert / load / hash-grouped queries
      ontology.py             # Ontology snapshot, ingest-config snapshot, allowed-domain lookup
      chunks.py               # Chunk + mention persist; entity-mention search; centrality signals
      summary.py              # Aggregate counts + CorpusStatusSummary builder
      claims.py               # Claim insert / load / count
      kg_profiles.py          # Entity profiles, community assignments, community reports
      kg_relations.py         # Canonical relations, relation evidence, graph metadata
    lancedb.py                # Canonical vector store: chunk_vectors + entity_embeddings + native FTS + BTree scalar indexes
    lance_sql.py              # Safe LanceDB filter builders (eq_clause, in_clause)
    sql_helpers.py            # Safe SQLite `IN (?, ?, ...)` helpers
    llm_jobs.py               # Persistent LLM job queue API
    models.py                 # Typed store records
  eval/                       # Answer-quality harness (LLM-judged topic coverage)
    metrics.py
    models.py
    report.py
    runner.py
  mcp/
    server.py                 # FastMCP server + lifespan bundle + phase/notice/sampler bridges
    async_runtime.py          # run_tool: worker-thread wrapper + hard timeout for tool bodies
    tools.py                  # Tool orchestration helpers (still thin)
    models.py                 # Pydantic output models for every MCP tool + resource + prompt
  cli/
    __init__.py               # Typer app; command discovery
    __main__.py               # `python -m lxd.cli` entry point
    ingest.py
    status.py
    preflight.py              # Schema-integrity + corpus-readiness gate
    eval.py                   # Retrieval-quality harness (aliased as `pixi run retrieval-check`)
    eval_quality.py           # Answer-quality harness
    graph.py                  # build-graph / graph-status commands
  observability/
    logging.py                # structlog config, log_duration, scrub_secrets processor
```

This layout is binding unless a later design document replaces it explicitly.

## 3. Layer Responsibilities

### 3.1 `settings/`

Owns:

- typed settings models
- `config.yaml` and optional `config.{profile}.yaml` loading
- config validation

Must not own:

- ingest logic
- query logic
- storage side effects beyond loading configuration

### 3.2 `domain/`

Owns:

- canonical ID construction
- citation-label formatting
- status and lifecycle enums or value objects

Must be:

- framework-independent
- backend-independent
- usable from tests without launching services

### 3.3 `ontology/`

Owns:

- YAML loading and `!include` resolution
- `networkx.MultiDiGraph` construction
- matcher-term extraction
- matcher normalization
- `pyahocorasick` automaton construction

Must not own:

- CLI surface behavior
- MCP protocol behavior
- SQLite connection management

### 3.4 `ingest/`

Owns:

- corpus scanning
- file classification
- diffing and move detection
- markdown/Docling conversion
- chunk generation
- asset registration orchestration
- mention indexing orchestration
- ingest phase sequencing

### 3.5 `retrieval/`

Owns:

- dense retrieval
- rerank application
- query-time filtering
- evaluation metrics and retrieval benchmarking

Must not own:

- MCP protocol serialization
- CLI argument parsing

### 3.6 `synthesis/`

Owns:

- answer assembly from retrieved evidence
- no-answer and insufficient-evidence decisions

Must not own:

- retrieval execution
- store mutation

### 3.7 `stores/`

Owns:

- LanceDB access
- SQLite access
- schema bootstrap and migrations
- store-level query helpers

Must not own:

- ontology parsing
- answer composition
- MCP request validation

### 3.8 `mcp/`

Owns:

- FastMCP server wiring
- tool definitions
- tool input/output serialization

Must remain thin:

- call application/query/store modules
- do not embed retrieval policy
- do not embed ingest policy

### 3.9 `cli/`

Owns:

- command entrypoints
- CLI argument parsing
- progress display wiring

Must remain thin:

- call application/ingest/query modules
- do not duplicate business logic

### 3.10 `observability/`

Owns:

- structured logging setup (`configure_logging`, UTC timestamps, JSON/console renderer)
- the `log_duration` context manager that emits `<event>.started` / `<event>.completed` pairs with `duration_ms`
- the `scrub_secrets` structlog processor that redacts sensitive keys
- optional metrics/report helpers and OpenTelemetry / Prometheus wiring (gated by `observability.*` config)

Must not own:

- domain rules
- retrieval rules

### 3.11 `net/`

Owns:

- shared `httpx.Client` / `httpx.AsyncClient` factories with pool sizing
  and timeout defaults
- construction of user agent and auth headers for outbound HTTP

Must not own:

- business logic or response parsing; callers remain responsible for
  endpoint-specific behaviour

## 4. Dependency Direction

Allowed dependency direction:

- `cli` -> `app`, `ingest`, `retrieval`, `stores`, `observability`, `settings`, `net`
- `mcp` -> `app`, `retrieval`, `stores`, `ontology`, `observability`, `settings`, `net`
- `app` -> every non-adapter layer as wiring only
- `ingest` -> `domain`, `ontology`, `stores`, `settings`, `observability`, `net`
- `retrieval` -> `domain`, `stores`, `ontology`, `settings`, `observability`, `net`
- `synthesis` -> `domain`, `settings`, `net`
- `ontology` -> `domain`, `settings`
- `stores` -> `domain`, `settings`
- `net` -> `settings`

Disallowed dependencies:

- `domain` importing `cli`, `mcp`, `stores`, or framework libraries
- `stores` importing `mcp` or `cli`
- `ontology` importing `mcp` or `cli`
- `synthesis` importing `mcp`, `cli`, or storage clients directly
- circular imports between `ingest`, `retrieval`, `ontology`, and `stores`

## 5. Shared Module Rule

Shared code is allowed only when it has a **single clear reason to exist**.

Allowed examples:

- `domain/ids.py`
- `domain/citations.py`
- `ontology/normalization.py`
- `observability/logging.py`

Forbidden examples:

- `utils.py`
- `helpers.py`
- `common.py`
- `misc.py`

If a function cannot be named into a specific module with a stable domain purpose, it does not belong in shared code yet.

## 6. Service Rule

There must not be a global service-locator pattern.

Rules:

- services are constructed explicitly in `app/bootstrap.py`
- long-lived resources such as LanceDB handles, ontology graph, and settings objects may be owned by bootstrap/runtime wiring
- request-scoped resources such as SQLite connections are opened per operation

Dependency injection may be lightweight and explicit.
Do not introduce a container framework unless a later design document explicitly requires it.

## 7. Naming Rule

Rules:

- module names describe business purpose, not implementation vagueness
- prefer `matcher.py`, `query_pipeline.py`, `citations.py`, `sqlite.py`
- avoid names such as `base.py`, `manager.py`, `processor.py`, or `helpers.py` unless the role is truly singular and well-bounded

## 8. Testing Layout

Tests are tagged with pytest markers so the suite can be sliced by
layer. Test files live flat under `tests/` and are named
`test_<module_or_topic>.py`; only a few structural subdirectories
exist. Per-package subdirectory mirroring was proposed as an
aspiration but never adopted — the flat layout is what actually ships
and what CI runs.

```text
tests/
  conftest.py              # shared fixtures + --update-golden flag
  unit/                    # pure-logic tests (no disk, no network)
  integration/             # temp-dir SQLite/LanceDB + local FastMCP
  golden/                  # golden transcripts (mcp_tool_manifest.json)
  eval/                    # retrieval- and answer-quality gold sets (eval_set.json, golden_quality_set.json)
```

Markers (registered in `pyproject.toml`):

- `unit` — isolated pure-function tests
- `integration` — wire multiple components through temp dirs and local services
- `e2e` — CLI / MCP tool transcripts against a seeded corpus
- `property` — Hypothesis property tests
- `benchmark` — `pytest-benchmark` regression gates (opt-in)
- `slow` — long-running tests excluded from the default suite

Rules:

- pure logic should have unit tests near its package area
- store-backed behavior should have integration tests
- end-to-end ingest and MCP behavior should have smoke tests
- API-surface regressions (e.g. MCP tool manifest) are guarded by
  golden-file tests under `tests/golden/`; update with
  `pytest --update-golden`

## 9. Anti-Patterns

Do not introduce:

- business logic in `mcp/server.py`
- business logic in CLI command modules
- direct SQL scattered outside the `stores/sqlite/` subpackage and `stores/schema.py`
- hand-rolled `IN (?, ?, …)` clauses outside `stores/sql_helpers.py`
- direct LanceDB query construction scattered outside `stores/lancedb.py` and retrieval modules
- raw f-string interpolation into LanceDB `where` clauses outside `stores/lance_sql.py`
- YAML schema assumptions duplicated outside ontology modules
- citation formatting duplicated outside `domain/citations.py`
- more than one implementation of chunk ID generation
- ad-hoc `httpx.Client()` instances outside `net/http.py`
- new synchronous MCP tools that bypass `mcp/async_runtime.run_tool`

## 10. Source Of Truth Rule

This document governs implementation structure.

If another plan document conflicts with this one on package layout, module boundaries, or dependency direction, this document wins unless it is explicitly superseded.
