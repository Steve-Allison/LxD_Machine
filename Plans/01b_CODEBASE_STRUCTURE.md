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
    bootstrap.py
  settings/
    models.py
    loader.py
  domain/
    ids.py
    citations.py
    status.py
  ontology/
    loader.py
    graph.py
    matcher.py
    normalization.py
  ingest/
    scanner.py
    diff.py
    markdown.py
    docling.py
    chunking.py
    assets.py
    mentions.py
    pipeline.py
  retrieval/
    dense.py
    rerank.py
    query_pipeline.py
    eval.py
  synthesis/
    answering.py
  stores/
    sqlite.py
    lancedb.py
    models.py
  mcp/
    server.py
    tools.py
  cli/
    ingest.py
    status.py
    eval.py
  observability/
    logging.py
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

- structured logging setup
- optional metrics/report helpers

Must not own:

- domain rules
- retrieval rules

## 4. Dependency Direction

Allowed dependency direction:

- `cli` -> `app`, `ingest`, `retrieval`, `stores`, `observability`, `settings`
- `mcp` -> `app`, `retrieval`, `stores`, `ontology`, `observability`, `settings`
- `app` -> every non-adapter layer as wiring only
- `ingest` -> `domain`, `ontology`, `stores`, `settings`, `observability`
- `retrieval` -> `domain`, `stores`, `ontology`, `settings`, `observability`
- `synthesis` -> `domain`, `settings`
- `ontology` -> `domain`, `settings`
- `stores` -> `domain`, `settings`

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

Tests should mirror the package layout:

```text
tests/
  ontology/
  ingest/
  retrieval/
  synthesis/
  stores/
  mcp/
  eval/
```

Rules:

- pure logic should have unit tests near its package area
- store-backed behavior should have integration tests
- end-to-end ingest and MCP behavior should have smoke tests

## 9. Anti-Patterns

Do not introduce:

- business logic in `mcp/server.py`
- business logic in CLI command modules
- direct SQL scattered outside `stores/sqlite.py`
- direct LanceDB query construction scattered outside `stores/lancedb.py` and retrieval modules
- YAML schema assumptions duplicated outside ontology modules
- citation formatting duplicated outside `domain/citations.py`
- more than one implementation of chunk ID generation

## 10. Source Of Truth Rule

This document governs implementation structure.

If another plan document conflicts with this one on package layout, module boundaries, or dependency direction, this document wins unless it is explicitly superseded.
