# MLX Transition and System Bottleneck Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fully transition LxD Machine to Apple Silicon resident MLX (chat, reranking, and embeddings), update FastMCP 4.1.1 protocol contracts, eliminate SQLite full-table scans via indexes and connection pooling, compact LanceDB tables, and parallelize multi-query embeddings and design agent stages.

**Architecture:**
- MLX inference: in-process Qwen3-Reranker (4B/0.6B) and resident `mlx_lm.server` for synthesis, replacing Ollama and external llama-server.
- Protocol alignment: FastMCP 4.1.1 `ToolAnnotations` snake_case hints and `ctx.session.create_message` sampling.
- Storage optimization: Schema migration 0005 adding missing foreign key indexes (`chunk_rows`, `mention_rows`), SQLite PRAGMA `mmap_size`, LanceDB compaction, and connection pooling on retrieval paths.
- Pipeline concurrency: batched embedding for multi-query paraphrases and `asyncio.gather` for independent design agent drafting steps.

**Tech Stack:** Python 3.14.8, Apple Silicon MLX (0.32.4), mlx-lm (0.32.2), LanceDB 0.40.0, PyArrow 26.0, FastMCP 4.1.1, SQLite 3.54, Pydantic 2.14.

## Global Constraints
- Target platform: macOS Apple Silicon (`osx-arm64`).
- No architectural shortcuts; strict typed contracts with Pyright.
- Deterministic IDs and backward-compatible schema migrations via `PRAGMA user_version`.
- Reversible migrations with automatic `.bak` backups.

---

### Task 1: FastMCP 4.1.1 Protocol Alignment & Lint Cleanup

**Files:**
- Modify: `src/lxd/mcp/server.py:78-85, 180-197`
- Modify: `pyproject.toml:46-62`
- Modify: `src/lxd/mlx/embeddings.py:7`
- Modify: `src/lxd/mlx/embed_server.py:17-20, 145-150`
- Test: `tests/integration/test_mcp_tool_manifest.py`

**Interfaces:**
- Consumes: `mcp.types.ToolAnnotations(read_only_hint, idempotent_hint, open_world_hint, destructive_hint)`
- Consumes: `Context.session.create_message(...)`

- [x] **Step 1: Write/update test for tool annotations and context sampling**
Verify that `test_mcp_tool_manifest.py` validates annotations and server tool signatures against FastMCP 4.1.

- [x] **Step 2: Update `ToolAnnotations` in `src/lxd/mcp/server.py`**
Replace camelCase parameters (`readOnlyHint`, `idempotentHint`, `openWorldHint`) with snake_case (`read_only_hint`, `idempotent_hint`, `open_world_hint`).

- [x] **Step 3: Update `_do_sample` in `src/lxd/mcp/server.py`**
Use `await ctx.session.create_message(...)` instead of `ctx.sample(...)`.

- [x] **Step 4: Update `pyproject.toml` lint ignore and clean unused imports**
Add `"PLR0917"` to `tool.ruff.lint.ignore` in `pyproject.toml`. Remove unused `import httpx` in `src/lxd/mlx/embeddings.py`. Rename `_MODEL` and `_TOKENIZER` to lowercase `_model` and `_tokenizer` in `src/lxd/mlx/embed_server.py`.

- [x] **Step 5: Verify typecheck and lint pass**
Run: `pixi run lint` and `pixi run typecheck`.

---

### Task 2: MLX Transition & Reranker 4B Configuration

**Files:**
- Modify: `config.yaml:26-36, 107-124`
- Modify: `src/lxd/mlx/rerank.py:12-40, 56-95`
- Modify: `src/lxd/retrieval/rerank.py:16, 40-55, 198-233`
- Modify: `src/lxd/settings/models.py:57-65, 533-575`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Consumes: `config.reranker.backend == "mlx"`, `config.models.rerank == "mlx-community/Qwen3-Reranker-4B-mxfp8"`
- Produces: `score_documents(query, documents, model_id=..., instruction=..., max_length=...) -> list[float]`

- [x] **Step 1: Write test for MLX reranker probing and document scoring mock**
Add unit test in `tests/test_rerank.py` verifying that `rerank.backend == "mlx"` scores documents and handles empty candidate lists or fake scorers cleanly.

- [x] **Step 2: Guard MLX imports in `src/lxd/mlx/rerank.py`**
Ensure `import mlx.core as mx` and `from mlx_lm import load` are imported safely, and `probe_mlx_reranker` reports availability accurately.

- [x] **Step 3: Update `config.yaml` to use MLX**
Set `models.llm_backend: mlx`, `reranker.backend: mlx`, `models.rerank: mlx-community/Qwen3-Reranker-4B-mxfp8`, and configure the `mlx:` block.

- [x] **Step 4: Run reranker test suite**
Run: `pixi run pytest tests/test_rerank.py -v`.

---

### Task 3: SQLite Indexing Migration 0005 & MMap Optimization

**Files:**
- Create: `src/lxd/stores/migrations/0005_chunk_mention_indexes.py` (or update `src/lxd/stores/schema.py`)
- Modify: `src/lxd/stores/_base_ddl.py:55-88`
- Modify: `src/lxd/stores/sqlite/connection.py:13-43`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: Indexes `idx_chunk_rows_source_rel_path`, `idx_chunk_rows_document_id`, `idx_mention_rows_source_rel_path`, `idx_mention_rows_chunk_id`.
- Produces: `PRAGMA mmap_size = 268435456;` on all connections.

- [x] **Step 1: Write test for schema migration 0005 and index existence**
Verify in `tests/test_schema.py` that migration 0005 applies cleanly and creates the required indexes.

- [x] **Step 2: Implement migration 0005 in `stores/schema.py` and update `_base_ddl.py`**
Add numbered migration 0005 to `_MIGRATIONS` in `src/lxd/stores/schema.py`, updating schema version to 5 and adding index definitions to `_base_ddl.py`.

- [x] **Step 3: Add `PRAGMA mmap_size` in `src/lxd/stores/sqlite/connection.py`**
Add `connection.execute("PRAGMA mmap_size = 268435456;")` (256 MB) to `connect_sqlite`.

- [x] **Step 4: Run schema test suite**
Run: `pixi run pytest tests/test_schema.py -v`.

---

### Task 4: SQLite Connection Pooling & LanceDB Compaction on Hot Paths

**Files:**
- Modify: `src/lxd/retrieval/stores.py:25-37`
- Modify: `src/lxd/stores/lancedb.py:30-47, 320-330`
- Modify: `src/lxd/ingest/pipeline/orchestrator.py:640-660`
- Test: `tests/test_retrieval_stores.py` / `tests/test_query_pipeline.py`

**Interfaces:**
- Consumes: `pooled_connection(paths.sqlite_path)`
- Produces: `compact_chunk_table(table)` in `stores/lancedb.py` called post-ingest.

- [x] **Step 1: Write test for pooled connection reuse in retrieval stores**
Verify that multiple calls to `open_retrieval_stores` in the same thread share the schema-initialised connection without re-running PRAGMA overhead.

- [x] **Step 2: Update `open_retrieval_stores` to use `pooled_connection`**
Modify `open_retrieval_stores` in `src/lxd/retrieval/stores.py` to yield from `pooled_connection(paths.sqlite_path)`.

- [x] **Step 3: Implement LanceDB compaction function**
In `src/lxd/stores/lancedb.py`, add `def compact_chunk_table(table: Any) -> None:` calling `table.compact_files()` and `table.cleanup_old_versions()`.

- [x] **Step 4: Call compaction at the conclusion of `run_ingest`**
In `src/lxd/ingest/pipeline/orchestrator.py`, call `compact_chunk_table(vector_table)` and `compact_chunk_table(cache_table)`.

- [x] **Step 5: Run tests**
Run: `pixi run pytest tests/test_query_pipeline.py -v`.

---

### Task 5: Pipeline & Agent Concurrency Optimizations

**Files:**
- Modify: `src/lxd/retrieval/query_pipeline.py:284-290`
- Modify: `src/lxd/agents/design.py:140-200`
- Test: `tests/test_query_pipeline.py`, `tests/test_design_agent.py`

**Interfaces:**
- Produces: Batch embedding `embed_texts(config, [question, *paraphrases])` in `query_pipeline.py`.
- Produces: Parallelized stage drafting in `src/lxd/agents/design.py`.

- [x] **Step 1: Write test for batched query embedding**
Test that `multi_query_applied` invokes a single batched embedding call rather than serial single-item embeddings.

- [x] **Step 2: Update multi-query embedding in `query_pipeline.py`**
Replace serial list comprehension with `embed_texts(config, variant_texts)`.

- [x] **Step 3: Update `_design_learning_async` in `src/lxd/agents/design.py`**
Use `asyncio.gather` for Stage 1 (`objectives` and `modality_plan`) and Stage 2 (`outline` and `assessment`).

- [x] **Step 4: Run agent and query pipeline tests**
Run: `pixi run pytest tests/test_query_pipeline.py tests/test_design_agent.py -v`.

---

### Task 6: Full Verification Suite

**Files:**
- Run: Entire test suite, lint, format, and typecheck

- [x] **Step 1: Run linter**
`pixi run lint`

- [x] **Step 2: Run typechecker**
`pixi run typecheck`

- [x] **Step 3: Run full test suite**
`pixi run test`
