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
    bootstrap.py              # AppContext, config_digest, config.lock reconciliation
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
    chunking.py
    assets.py
    mentions.py
    relations.py              # LLM-based relation extraction
    claims.py                 # LLM-based claim extraction
    embedder.py               # Batched Ollama / OpenAI embedding with context-aware retry
    llm_client.py             # Shared synchronous/async LLM client facade
    pipeline.py
  retrieval/
    dense.py
    rerank.py
    graph_routing.py          # Graph context augmentation for synthesis
    query_pipeline.py
    eval.py
  synthesis/
    answering.py
  stores/
    schema.py                 # Numbered migrations + ensure_schema; PRAGMA user_version
    connection.py             # Pragma-tight SQLite connect + close hooks
    sqlite.py                 # Query/upsert API (thin orchestrator, no DDL)
    lancedb.py                # Canonical vector store (uses lance_sql helpers)
    lance_sql.py              # Safe LanceDB filter builders (eq_clause, in_clause)
    sql_helpers.py            # Safe SQLite `IN (?, ?, ...)` helpers
    llm_jobs.py               # Persistent LLM job queue API
    models.py                 # Typed store records
    _sqlite_rows.py           # Row-to-record adapters (module-private)
    _sqlite_legacy_migrations.py  # Pre-versioning upgrades (module-private)
    _base_ddl.py
  mcp/
    server.py                 # FastMCP server + lifespan bundle
    async_runtime.py          # run_tool: async wrapper + hard timeout for tool bodies
    tools.py                  # Tool orchestration helpers (still thin)
  cli/
    ingest.py
    status.py
    eval.py
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

Tests mirror the package layout and are tagged with pytest markers so the
suite can be sliced by layer:

```text
tests/
  conftest.py              # shared fixtures + --update-golden flag
  unit/                    # pure-logic tests (no disk, no network)
  integration/             # temp-dir SQLite/LanceDB + local FastMCP
  golden/                  # golden transcripts (e.g. mcp_tool_manifest.json)
  ontology/
  ingest/
  retrieval/
  synthesis/
  stores/
  mcp/
  eval/
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
