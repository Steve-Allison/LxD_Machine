# Code Review Bottlenecks & Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all identified bottlenecks, N+1 query patterns, memory thrashing, and connection churn across knowledge graph profiling, MLX inference, retrieval, and ingest persistence.

**Architecture:** 
1. Batch pre-load claims, predicates, and community counts in `src/lxd/ontology/profiles.py` to turn $O(N)$ database round-trips into $O(1)$ batch queries, and batch persist profile records.
2. Remove inner-loop `mx.clear_cache()` in `src/lxd/mlx/rerank.py` and `src/lxd/mlx/embed_server.py` to allow Apple Silicon unified memory caching across documents.
3. Wire `src/lxd/retrieval/rerank.py` to use `lxd.net.http.get_sync_client` for HTTP connection pooling and keep-alive reuse.
4. Wire `src/lxd/retrieval/expansion.py` to use `lxd.stores.sqlite._pool.pooled_connection` instead of opening/closing ad-hoc connections.
5. Throttle `_persist_ingest_progress` in `src/lxd/ingest/pipeline/orchestrator.py` to avoid per-file database writes.
6. Adopt `in_clause_for` and chunk dynamic SQLite parameter lists in `src/lxd/stores/sqlite/chunks.py`.

**Tech Stack:** Python 3.14, SQLite 3, LanceDB, MLX / MLX-LM, HTTPX, Pytest, Ruff, Pyright.

## Global Constraints
- Target Python 3.14 (`pyproject.toml`, `pixi.toml`).
- Strict type checking (`pyright src`) must pass with 0 errors.
- Ruff linter and formatter (`pixi run lint`, `pixi run fmt`) must pass with 0 errors.
- Preflight gate (`pixi run preflight`) must pass.
- All existing tests (`pixi run test`) must pass (556 tests).

---

### Task 1: Batch Pre-loads and Bulk Persistence in Entity Profiles & Community Reports

**Files:**
- Modify: `src/lxd/ontology/profiles.py`
- Test: `tests/test_community_enrichment_composite_pk.py`
- Test: `tests/unit/test_ontology_profiles.py` (or create new unit tests)

**Interfaces:**
- `build_entity_profiles`: Signature remains unchanged, but internal SQL queries for claims, predicates, and community member counts are pre-loaded in single passes.
- `build_community_reports`: Reuses `load_entity_profile_source_hashes` dictionary rather than executing SQL in loop.

- [x] **Step 1: Write tests for batch pre-loaded profiling**
Add unit test asserting `build_entity_profiles` produces identical profiles while executing batch queries.

- [x] **Step 2: Run test to verify current state**
Run: `pixi run pytest tests/test_community_enrichment_composite_pk.py`

- [x] **Step 3: Implement batch pre-loading in `src/lxd/ontology/profiles.py`**
1. Add helper `_load_all_top_predicates(connection)` to load relation counts grouped by `subject_entity_id`.
2. Add helper `_load_claims_grouped_by_entity(connection)` to load claims grouped by entity ID.
3. Precompute `community_member_counts = Counter(community_assignments.values())`.
4. Replace per-entity `upsert_entity_profile` with bulk transaction or batched insert.
5. In `build_community_reports`, pass the pre-loaded `entity_profile_hashes` directly.

- [x] **Step 4: Run tests and verify**
Run: `pixi run pytest tests/test_community_enrichment_composite_pk.py`
Expected: PASS

- [x] **Step 5: Commit**
```bash
git add src/lxd/ontology/profiles.py tests/
git commit -m "perf: eliminate N+1 query cascades in entity profiling and community reports"
```

---

### Task 2: Fix MLX Inner-Loop Memory Thrashing

**Files:**
- Modify: `src/lxd/mlx/rerank.py:76-90`
- Modify: `src/lxd/mlx/embed_server.py:91-109`
- Test: `tests/test_rerank.py`

**Interfaces:**
- `score_documents`: Evaluates all documents while retaining cache between candidates; clears cache once at the end of the batch.
- `_embed`: Processes texts without wiping the memory cache inside the text loop.

- [x] **Step 1: Run existing rerank tests**
Run: `pixi run pytest tests/test_rerank.py`
Expected: PASS

- [x] **Step 2: Update `src/lxd/mlx/rerank.py` and `src/lxd/mlx/embed_server.py`**
Remove `mx.clear_cache()` from inside `for document in documents:` and `for text in texts:`. Move `mx.clear_cache()` to after the loop.

- [x] **Step 3: Run rerank tests and verify**
Run: `pixi run pytest tests/test_rerank.py`
Expected: PASS

- [x] **Step 4: Commit**
```bash
git add src/lxd/mlx/rerank.py src/lxd/mlx/embed_server.py
git commit -m "perf: eliminate inner-loop mx.clear_cache() memory churn in MLX inference"
```

---

### Task 3: HTTP Connection Pooling & SQLite Pooled Connection Usage

**Files:**
- Modify: `src/lxd/retrieval/rerank.py:414-420`
- Modify: `src/lxd/retrieval/expansion.py:98-120`
- Test: `tests/test_rerank.py`
- Test: `tests/test_query_pipeline.py`

**Interfaces:**
- `_client` in `src/lxd/retrieval/rerank.py`: Uses `lxd.net.http.get_sync_client`.
- `_expand_from_corpus` in `src/lxd/retrieval/expansion.py`: Uses `pooled_connection`.

- [x] **Step 1: Run query pipeline tests**
Run: `pixi run pytest tests/test_query_pipeline.py`
Expected: PASS

- [x] **Step 2: Implement pooled client and database connection**
1. In `src/lxd/retrieval/rerank.py`, update `_client(config)` to return `get_sync_client(base_url=str(config.reranker.url), timeout_secs=float(config.reranker.timeout_secs))`. Note: do not use `with _client(config) as client:` context manager if closing is not desired; or adapt callers to not close the pooled client.
2. In `src/lxd/retrieval/expansion.py`, use `with pooled_connection(store_paths.sqlite_path) as connection:`.

- [x] **Step 3: Verify tests pass**
Run: `pixi run pytest tests/test_rerank.py tests/test_query_pipeline.py`
Expected: PASS

- [x] **Step 4: Commit**
```bash
git add src/lxd/retrieval/rerank.py src/lxd/retrieval/expansion.py
git commit -m "perf: reuse pooled HTTP clients and pooled SQLite connections on retrieval hot paths"
```

---

### Task 4: Ingest Progress Throttling & SQLite Parameter Safety

**Files:**
- Modify: `src/lxd/ingest/pipeline/orchestrator.py`
- Modify: `src/lxd/stores/sqlite/chunks.py`
- Test: `tests/integration/test_ingest_resilience.py`
- Test: `tests/test_dense.py`

**Interfaces:**
- `_persist_ingest_progress`: Called every $N$ files (e.g. 10) and on completion/failure/boundary, not every single file.
- `load_chunk_centrality_signals`, `load_corpus_related_entity_ids`: Use `in_clause_for` from `stores/sql_helpers.py` with parameter chunking.

- [x] **Step 1: Run ingest and chunk store tests**
Run: `pixi run pytest tests/integration/test_ingest_resilience.py tests/test_dense.py`
Expected: PASS

- [x] **Step 2: Implement throttling and chunked SQL helper adoption**
1. In `orchestrator.py`, update the loop so progress is persisted only if `files_completed % 10 == 0` or upon loop exit.
2. In `src/lxd/stores/sqlite/chunks.py`, refactor `load_chunk_centrality_signals` and `load_corpus_related_entity_ids` to use `in_clause_for`.

- [x] **Step 3: Verify tests pass**
Run: `pixi run pytest tests/integration/test_ingest_resilience.py tests/test_dense.py`
Expected: PASS

- [x] **Step 4: Commit**
```bash
git add src/lxd/ingest/pipeline/orchestrator.py src/lxd/stores/sqlite/chunks.py
git commit -m "perf: throttle ingest progress persistence and harden SQLite dynamic clauses"
```

---

### Task 5: Full Verification Suite

**Files:**
- All modified files

- [x] **Step 1: Run format check**
Run: `pixi run fmt`

- [x] **Step 2: Run linter**
Run: `pixi run lint`

- [x] **Step 3: Run strict typecheck**
Run: `pixi run typecheck`

- [x] **Step 4: Run preflight check**
Run: `pixi run preflight`

- [x] **Step 5: Run full test suite**
Run: `pixi run test`
Expected: 556 passed, 0 failures.

- [x] **Step 6: Push to remote repository**
```bash
git push origin main
```

---

### Task 6: Latent Bottlenecks & Retrieval Lane Optimizations (Completed)

1. **Eliminated Full-Corpus Manifest & Mention Scans on Every Search Query**: Replaced heavy `summarize_store` call with lightweight `has_chunks(connection)` in `src/lxd/retrieval/query_pipeline.py`.
2. **Batch Chunk Loading**: Implemented `load_chunk_records_by_ids` and converted sequential `load_chunk_record_by_id` loops in `src/lxd/retrieval/graph_lane.py` and `_append_missing_graph_lane_chunks` in `query_pipeline.py` into $O(1)$ batch queries.
3. **Batch Profile and Community Report Loading**: Implemented `load_entity_profiles_by_ids` and `load_community_reports_by_ids` in `src/lxd/stores/sqlite/kg_profiles.py` to eliminate N+1 loops in `build_graph_context` and `_matched_community_ids`.
4. **LanceDB Index Query Deduplication**: Eliminated 4x redundant `table.list_indices()` queries per table open in `src/lxd/stores/lancedb.py`.
5. **Streaming BLAKE3 Corpus Hashing**: Implemented genuine stream-chunked BLAKE3 hashing in `src/lxd/ingest/scanner.py`.
6. **All Verified**: Formatted, 0 ruff lint errors, 0 pyright errors, 556/556 tests passing. Committed and pushed to `main`.
