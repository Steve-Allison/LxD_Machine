# LxD Machine — SOTA Implementation Plan (Items 1–20)

## Context

A 2026-05-05 critical audit identified 20 architectural and code-quality gaps across retrieval, knowledge-graph signal use, code structure, robustness, and observability. The audit grounded each finding in `file:line` references; an Explore-agent pass against the live codebase has now confirmed exact integration points for every item.

This plan organises the 20 items into a multi-session implementation schedule prioritised by **return on investment** (impact ÷ effort), with two structural sequencing rules:

1. **Eval gate goes first.** Without it every subsequent change ships blind. We measure baseline, then every quality-affecting change is evaluated against that baseline before it lands.
2. **Foundation before refinement.** Dead-code removal and architectural consolidation precede new features so we don't refactor newly-added code twice.

The pending wiki swap (already 95% done — config flipped, schema v6 ready, parser + tests landed but full ingest not yet run) is the **prerequisite** for items #2 and #3 (KG signal lift). Session 1 closes that out before anything else.

The fixed-cost expectation: **~60 hours over ~10–11 sessions**, half-day to full-day each. Each session ends with green CI (lint + typecheck + tests + eval) and a single commit. Sessions are not blocking — items within a session can be reordered, but inter-session prerequisites are enforced.

---

## Items 1–20 — Reordered by Priority (ROI)

The original audit numbering is preserved in `[#N]` for traceability; the order below is the implementation order.

### Tier 1 — Foundation & measurement (must precede everything else)

#### `[#15]` Delete `_sqlite_legacy_migrations.py` — dead code

**Why first**: 887 lines of pre-versioning migration code, called from exactly one site (`sqlite.py:111` in `initialize_schema`). Schema is now at v6 with proper numbered migrations under `PRAGMA user_version`; the legacy bridge has done its job. Removing it shrinks `stores/` blast radius and removes the lurking trap of two parallel migration paths.

**Files**: `src/lxd/stores/_sqlite_legacy_migrations.py` (delete), `src/lxd/stores/sqlite.py:11,111` (remove import + call site).

**Action**: Verify no callers outside `sqlite.py` (Explore confirmed only one call site). Replace `migrate_legacy_schema(connection)` call with a fast `_assert_no_v2_legacy_table(connection)` guard that raises a clear error if a stale legacy table is encountered. Delete the file. Run integration tests.

**Verify**: `pixi run lint && pixi run typecheck && pixi run test` clean. `tests/test_schema_migrations.py::test_migration_0004_repairs_ghost_fk_in_extracted_relations` still passes (this is the regression test for the historic ghost-FK incident).

**Effort**: 30 min.

---

#### `[#6]` Eval as CI gate — `pixi run eval` runs on every commit, fails on regression

**Why second**: Every subsequent quality-affecting change (#1, #2, #3, #4, #9, #13, #17) needs a measurable delta. Without a baseline + gate, optimisations are folklore.

**Files**:
- `src/lxd/cli/eval.py:16-58` — current `eval_command()` returns metrics but doesn't fail on threshold.
- `src/lxd/retrieval/eval.py:44-75,117-160` — `recall_at_k`, `mrr_at_k`, `run_eval` (`EvalSummary` with `mean_recall_at_10`, `mean_mrr_at_10`, per-case results).
- `tests/eval/eval_set.json` — 20 cases, `{question, expected_source_files, domain}`.
- `pixi.toml` — add `eval-gate` task.
- `.github/workflows/` (if present) or pre-push hook.

**Actions**:
1. Run `pixi run eval` against current state, save `tests/eval/baseline.json` with `recall_at_10` and `mrr_at_10` snapshot.
2. Add `--gate-against=baseline.json --max-recall-drop=0.05 --max-mrr-drop=0.05` flag to `eval_command`.
3. New pixi task `eval-gate` that calls eval with the gate.
4. Wire into pre-push hook (or CI if present).
5. Document baseline-refresh procedure (after a deliberate quality-affecting change is verified, refresh baseline).

**Verify**: Run `pixi run eval-gate` against current state — should pass. Manually degrade a chunk of `eval_set.json` to confirm the gate fails.

**Effort**: 4 h.

**Prerequisite**: Wiki swap rebuild must have run, so the baseline reflects the actual production corpus. **Run the deferred ingest before this session.**

---

### Tier 2 — Highest-ROI retrieval & KG wins

#### `[#1]` Hybrid retrieval via LanceDB FTS5 (replaces hand-rolled lexical scoring)

**Why third**: LanceDB has native BM25 via tantivy. Current `_lexical_signal_score` (`query_pipeline.py:466-487`) is hand-rolled keyword counting with no IDF, length normalisation, or positional weighting. Same RRF fusion, much better lexical signal. **No new infra.**

**Files**:
- `src/lxd/stores/lancedb.py:33-52` — `open_chunk_table` is where FTS5 index is created; add `table.create_fts_index(["text"])` call after table creation. Apply also on `replace_source_chunks` if needed (LanceDB rebuilds indexes lazily).
- `src/lxd/retrieval/query_pipeline.py:459-487` — replace `_lexically_ranked` body with a LanceDB FTS5 query against the same table; preserve the `_fuse_ranked_prefix` interface.
- `src/lxd/retrieval/query_pipeline.py:329-373` — add a sibling `_lexical_ranked_candidates` to `_dense_ranked_candidates`, then fuse.

**Actions**:
1. Add FTS5 index creation to `open_chunk_table` (idempotent; LanceDB handles "exists").
2. Replace `_lexical_signal_score` with `table.search(question, query_type="fts").limit(N).to_list()`.
3. Keep RRF fusion in `_fuse_ranked_prefix` unchanged — the lexical lane just gets better candidates.
4. **Migration**: existing tables have no FTS5 index; the `--full` rebuild after the wiki swap creates them fresh. For incremental users, document the one-shot `pixi run reindex-fts` task.

**Verify**:
- `pixi run eval-gate` — measure recall delta vs baseline. Expected: +5-15% on lexically-driven queries.
- `tests/test_query_pipeline.py` (extend with FTS5 fusion test).

**Effort**: 3-4 h.

---

#### `[#14]` Switch to remote rerank API (Cohere or Voyage rerank-2)

**Why fourth**: Current reranker (`rerank.py:156-197 _ensure_reranker_service`) auto-spawns `llama-server` from inside a query path. Fragile under concurrent MCP load, blocks 30s on missing-binary, breaks on model-download race. Remote APIs are 5-15% better on MTEB and removed-failure-mode in one swap.

**Files**:
- `src/lxd/retrieval/rerank.py:61-114` — `rerank_chunks` HTTP POST (lines 86-95) is the swap target.
- `src/lxd/retrieval/rerank.py:117-197` — `_probe_reranker_uncached`, `_probe_reranker_http`, `_ensure_reranker_service`, `_client` — the auto-spawn machinery to deprecate.
- `src/lxd/settings/models.py` — `RerankerConfig`: add `backend: Literal["cohere", "voyage", "llama_cpp"]`, API-key env, model name. Pydantic discriminated union shape.
- `src/lxd/net/http.py` — reuse pooled `httpx.Client` for the new backend.

**Actions**:
1. Add `RerankerConfig.backend` with discriminated union for cohere / voyage / llama_cpp.
2. Implement `_rerank_via_cohere(question, candidates)` and `_rerank_via_voyage(question, candidates)` using the pooled `httpx` client.
3. Keep llama_cpp path as fallback for offline / cost-controlled runs but remove auto-spawn — require user to start it manually with a clear "rerank unavailable" warning.
4. Update `config.yaml` to document the choice.

**Verify**:
- `pixi run eval-gate` — measure recall@10 + mrr@10 delta. Expected: +3-8%.
- New unit tests for each backend's adapter (with mocked HTTP).
- Concurrent-rerank stress test (no longer races on subprocess spawn).

**Effort**: 3 h.

---

#### `[#3]` Wiki-link `[[slug]]` → entity mapping; emit `(entity, REFERENCES, entity)` relations

**Why fifth**: We just added `wiki_metadata.py` to extract `[[slug]]` references but they currently dead-end at the chunk row. Each parsed slug that maps to an ontology canonical_id becomes a free `(entity, REFERENCES, entity)` relation for the entity graph. **No LLM cost.** ~1000+ relations expected from the curated wiki.

**Files**:
- `src/lxd/ingest/wiki_metadata.py:56-113` — parser already exists; emits `wiki_links: tuple[str, ...]`.
- `src/lxd/ontology/loader.py` — entity definitions have `canonical_id`. Slug pattern (`mayers-multimedia-principles`) ↔ canonical_id pattern (`mayers_multimedia_principles` or `mayers-multimedia-principles`) — explorer confirmed direct lowercased match.
- `src/lxd/stores/sqlite.py:1590-1621` — `replace_canonical_relations` is the bulk-insert target.
- New file: `src/lxd/ingest/wiki_relations.py` — derive relations from wiki_links + parsed sources.

**Actions**:
1. New `wiki_relations.py` with `derive_wiki_link_relations(chunk_records, ontology) -> list[CanonicalRelationRecord]` and `derive_citation_relations(chunk_records) -> list[CanonicalRelationRecord]` (the latter from the Sources line we already parse).
2. Call from `pipeline.py` after chunk persistence; emit into the same `extracted_relations` / `relations` tables. Use a synthetic predicate `wiki_references` (for `[[slug]]`) and `wiki_cites` (for Sources). Source: `extraction_model="wiki_metadata"` so they're distinguishable from LLM-extracted.
3. Slug normaliser: lowercase, replace hyphens-with-underscores, strip extensions if present. Skip slugs that don't resolve to an ontology canonical_id (log dangling-link count).
4. Add a "dangling slug" report to `pixi run status`.

**Verify**:
- `pixi run eval-gate` — measure delta (expected: +2-5% on graph-enabled questions; small but meaningful).
- New tests in `tests/test_wiki_relations.py` for slug normalisation, canonical_id matching, citation tuple shape.
- Manual: `pixi run python -c "from lxd.stores.sqlite import load_all_extracted_relations; ..."` to spot-check.

**Effort**: 4 h.

---

#### `[#2]` Use centrality + community signals in retrieval (boost + diversification)

**Why sixth**: Centrality is computed (`ontology/profiles.py:34-100` builds `EntityProfileRecord` with 6 metrics) but **never reaches retrieval**. Communities are the same. Both are pure cost we're not amortising. Adding centrality as a fusion lane and community-aware MMR is straightforward.

**Files**:
- `src/lxd/retrieval/query_pipeline.py:72-98` — `RankedChunk`: add `central_entity_score: float = 0.0`, `community_id: int | None = None`.
- `src/lxd/retrieval/query_pipeline.py:329-373` — `_dense_ranked_candidates`: load mention-implied centrality + community via SQLite lookup.
- `src/lxd/retrieval/query_pipeline.py:421-456` — `_fuse_ranked_prefix`: add a centrality-weighted RRF lane.
- `src/lxd/retrieval/graph_routing.py:33-105` — already loads community reports for the synthesis prompt; lift the same loader for retrieval-time diversification.
- New: `_diversify_by_community(ranked, k)` — MMR-style: pick top-1 from each community before going deep into any one community.

**Actions**:
1. Extend `RankedChunk` with `central_entity_score` (max centrality across mentioned entities) + `community_ids: tuple[int, ...]`.
2. After dense+lexical+rerank fuse, apply community-aware MMR: pick top-K with `λ * relevance + (1-λ) * community_diversity`. Default `λ=0.7`, configurable.
3. Add a centrality-weighted RRF lane to `_fuse_ranked_prefix`.
4. New retrieval-config knobs: `centrality_fusion_weight`, `community_diversity_lambda`.

**Verify**:
- `pixi run eval-gate` — expected: +5-10% recall@10, with bigger gains on broad queries that span concepts.
- Manual: query "compare ADDIE to SAM" — should return chunks from both ADDIE community and SAM community, not 8 ADDIE chunks.
- Tests for `_diversify_by_community` (deterministic given fixed candidate set + community map).

**Effort**: 6 h.

---

#### `[#4]` Synthesis: OpenAI primary + Ollama fallback

**Why seventh**: Asymmetric stack — relations and claims use OpenAI primary + Ollama fallback (`ingest/llm_client.py`); synthesis is **Ollama-only locked to `qwen3:14b`** (`synthesis/answering.py:81`). User-facing answer quality matters more than ingest-time extraction quality, yet we're using the smaller model on the user-facing path.

**Files**:
- `src/lxd/synthesis/answering.py:60-104` — `synthesize_answer` calls `_client(config).generate()` only; replace with `lxd.ingest.llm_client.call_with_fallback_async`.
- `src/lxd/settings/models.py` — extend `SynthesisConfig` with `backend: Literal["openai", "ollama"]`, `fallback_backend: Literal["ollama", "openai"]`, `openai_model`, `ollama_model`.

**Actions**:
1. Reuse `call_with_fallback_async` exactly as relations/claims do — single source of truth for backend dispatch.
2. Default config: `backend="openai"`, `openai_model="gpt-4o-mini"` (fast, cheap, smart enough for this), `fallback_backend="ollama"`, `ollama_model="qwen3:14b"`.
3. Make synthesis async end-to-end (it's currently sync inside async MCP wrapper — `run_tool` handles the bridge but it's wasteful).

**Verify**:
- `pixi run eval-gate` — measure with the new synthesis backend. Expected: meaningful answer-quality lift on a manual rubric (no automated metric for synthesis quality unless we add one).
- Manual: spot-check 5 queries before/after.
- Existing tests in `tests/test_synthesis_answering.py` (or equivalent) still pass with mocked LLM.

**Effort**: 2-3 h.

---

### Tier 3 — Production guardrails

#### `[#11]` Per-run request budget cap

**Why**: No spend ceiling exists today. A misconfigured `--full` against a large corpus can run unbounded. Circuit breaker stops on errors, not on cost.

**Files**:
- `src/lxd/settings/models.py` — add `IngestBudget` (Pydantic): `max_embedding_tokens_per_run`, `max_llm_calls_per_run`, `max_estimated_cost_usd_per_run`.
- `src/lxd/ingest/pipeline.py` — track running totals; abort on threshold with a clear error.
- `src/lxd/stores/llm_jobs.py` — record per-call cost.

**Actions**:
1. New `IngestBudget` config section with sensible defaults.
2. Threshold check before each batch — refuse to start if projected cost exceeds remaining budget.
3. Already-spent telemetry rolls into the existing `ingest_runs.estimated_cost_usd` column (added in migration 0005).

**Verify**:
- New test: ingest with `max_estimated_cost_usd_per_run=0.001` aborts cleanly with a budget error after first batch.
- Existing tests unchanged.

**Effort**: 2 h.

---

#### `[#12]` OpenTelemetry instrumentation (actually implement what settings claim)

**Why**: `settings/models.py:257-281` declares `otel_enabled`, `otel_endpoint`, `prometheus_enabled`, `prometheus_port`. **Explorer confirmed: no runtime code reads these fields.** Either implement or remove the false promise.

**Files**:
- `src/lxd/observability/logging.py` — already configures structlog; add OTel exporter wiring.
- `src/lxd/mcp/async_runtime.py:38-72` — `run_tool` is the central span insertion point for all 20 MCP tools.
- `src/lxd/ingest/pipeline.py` — span around `run_ingest` and per-source.
- `src/lxd/retrieval/query_pipeline.py` — spans around dense / lexical / rerank / fuse / synth phases.
- New: `src/lxd/observability/tracing.py` — OTel SDK setup + decorators.

**Actions**:
1. `pixi add opentelemetry-sdk opentelemetry-exporter-otlp` (and prometheus client if implementing that lane too).
2. New `configure_tracing(config)` called from `app/bootstrap.py` if `otel_enabled=True`.
3. Add `@traced("mcp.tool.search_corpus")` etc. via `run_tool` wrapper.
4. Per-call attributes: `corpus_id`, `tool_name`, `query_length`, `result_count`, `latency_ms`, `embedding_cache_hits`.
5. **Or** delete the unused settings if you decide OTel isn't worth the complexity.

**Verify**:
- `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 pixi run mcp` — spans show up in a local Jaeger/Tempo.
- Metric: per-tool p99 latency surfaces in Prometheus.

**Effort**: 4 h (real implementation) or 30 min (delete the false promise).

**Decision needed**: implement vs delete? Default in plan: implement.

---

#### `[#10]` Persistent circuit breaker (survives crashes)

**Why**: `SystemicErrorCircuitBreaker` (`ingest/error_classification.py`) is per-process. Crashed pid loses state; restart immediately tries again and gets the same 3 SYSTEMIC errors. SQLite-backed breaker survives.

**Files**:
- `src/lxd/ingest/error_classification.py` — extend with `PersistentCircuitBreaker(connection)` that reads/writes a `circuit_breaker_state` table.
- `src/lxd/stores/_base_ddl.py` — add `circuit_breaker_state` table.
- `src/lxd/stores/schema.py` — migration 7.

**Actions**:
1. New table: `circuit_breaker_state(scope TEXT PRIMARY KEY, consecutive_failures INTEGER, last_error TEXT, tripped_at TEXT, last_failure_at TEXT)`.
2. `PersistentCircuitBreaker` reads on construct, writes on each `record_failure` / `record_success`. Configurable scope (default `"ingest_default"`).
3. `pixi run ingest --reset-breaker` to clear state manually.

**Verify**:
- New integration test: trip breaker, simulate process restart, confirm breaker state persists.
- Existing test unchanged.

**Effort**: 2 h.

---

### Tier 4 — Refactor surface

#### `[#7]` Split `stores/sqlite.py` (2001 lines → subpackage)

**Why**: 72 functions in one file = high blast radius. Explorer mapped 9 natural groups. Splitting makes the file navigable and future migrations safer.

**Files** (after refactor):
- `src/lxd/stores/sqlite/__init__.py` — re-exports for backward compat.
- `src/lxd/stores/sqlite/_connection.py` (lines 48-117 of old).
- `src/lxd/stores/sqlite/_runs.py` (lines 134-239).
- `src/lxd/stores/sqlite/_manifest.py` (lines 290-516).
- `src/lxd/stores/sqlite/_ontology.py` (lines 516-643).
- `src/lxd/stores/sqlite/_chunks.py` (lines 718-1108).
- `src/lxd/stores/sqlite/_claims.py` (lines 1241-1318).
- `src/lxd/stores/sqlite/_kg_profiles.py` (lines 1319-1577).
- `src/lxd/stores/sqlite/_kg_relations.py` (lines 1590-1873).
- `src/lxd/stores/sqlite/_summary.py` (lines 1128-1219).

**Actions**:
1. Create the subpackage; move functions; preserve the public API via `__init__.py` re-exports so no caller-side change is needed.
2. Each module under 300 lines.
3. `mypy --strict` / `pyright` zero errors.
4. **Tests are the safety net.** Run full suite after each module move.

**Verify**:
- `pixi run lint && pixi run typecheck && pixi run test` clean.
- `git grep "from lxd.stores.sqlite import"` returns same import paths.

**Effort**: 4 h.

---

#### `[#8]` Pydantic discriminated unions for backend dispatch

**Why**: 4+ `if cfg.backend == "openai"` chains across `relations.py:381,397,528`, `llm_client.py:181`, `claims.py:486`. Pydantic v2 `Discriminator` + `Tag` makes the union type-safe and replaces runtime branching with structural pattern matching (`match`/`case`) where it remains.

**Files**:
- `src/lxd/settings/models.py` — refactor `RelationExtractionConfig`, `ClaimExtractionConfig`, `RerankerConfig`, `SynthesisConfig` (after #4) to use discriminated unions.
- `src/lxd/ingest/relations.py:381-402,528` — replace if-chain with `match cfg: case OpenAIBackend(): ... case OllamaBackend(): ...`.
- `src/lxd/ingest/claims.py:486` — same pattern.
- `src/lxd/ingest/llm_client.py:181` — same pattern.

**Actions**:
1. Define `OpenAIBackendConfig(BaseModel, kind: Literal["openai"])` and `OllamaBackendConfig(BaseModel, kind: Literal["ollama"])`.
2. Use `Annotated[Union[OpenAI, Ollama], Discriminator("kind")]` for the backend field.
3. Replace if-chains with `match`/`case` everywhere.

**Verify**: existing tests cover both backends; both paths must continue to pass.

**Effort**: 2 h.

---

### Tier 5 — Advanced retrieval (A/B-dependent)

#### `[#9]` Late chunking (or contextual retrieval) — A/B test

**Why**: Current `hybrid_docling` chunking embeds chunks independently. **Late chunking** (Jina v3 / others — embed full doc, then split the embedding) preserves cross-chunk context. **Contextual retrieval** (Anthropic's pattern: prepend a chunk-level summary before embedding) gives 35-49% retrieval improvement on long docs in their study.

**Files**:
- `src/lxd/ingest/chunking.py:84-90,188` — chunking entry. Late chunking restructures.
- `src/lxd/ingest/embedder.py` — needs an "embed-document-then-split" path for late chunking.
- New: `src/lxd/ingest/contextual_chunker.py` — alternative strategy.

**Actions**:
1. Implement contextual retrieval first (cheaper, better-validated): for each chunk, generate a 1-sentence "this chunk discusses X in the context of Y" via a fast LLM, prepend to chunk text **before embedding only** (not before storing — the user-visible text stays clean). Cost is ~$0.30 per 1k chunks at gpt-4o-mini.
2. A/B against current chunking via `pixi run eval-gate` with both strategies tagged.
3. If contextual wins by ≥5%, ship it. If not, try late chunking.

**Verify**:
- `pixi run eval-gate --strategy=contextual` vs `pixi run eval-gate --strategy=hybrid_docling`.
- Document the winner in `Plans/03_INGEST_SPEC.md`.

**Effort**: 6 h.

---

#### `[#17]` HyDE / query rewriting

**Why**: Every query is embedded as-is. SOTA: generate a *hypothetical answer* via a fast LLM, embed *that*, retrieve on it. Gives ~10-20% recall lift on under-specified questions.

**Files**:
- `src/lxd/retrieval/query_pipeline.py:206` — `answer_question` is the orchestrator; HyDE goes before dense search.
- New: `src/lxd/retrieval/hyde.py` — generate hypothetical answer using `call_with_fallback_async`.

**Actions**:
1. New `hypothetical_answer(question, config) -> str` using OpenAI primary / Ollama fallback. Single LLM call, max 200 tokens.
2. Embed the hypothetical answer instead of (or in addition to — fuse two dense lanes) the original question.
3. Toggle: `retrieval.hyde_enabled` config knob, default off until eval shows a win.

**Verify**:
- `pixi run eval-gate --hyde=true` vs default.
- Manual: under-specified queries like "best practice for assessment" should improve.

**Effort**: 4 h.

---

#### `[#13]` LanceDB native hybrid query

**Why**: After #1 we have FTS5 indexes. LanceDB supports running both dense + FTS in one query via `query_type="hybrid"` instead of two parallel queries we manually fuse. Cleaner and slightly faster.

**Files**:
- `src/lxd/stores/lancedb.py:103-149` — `search_chunks` currently runs vector-only; add a `search_chunks_hybrid` variant.
- `src/lxd/retrieval/query_pipeline.py:329-373` — replace the two-lane fetch with one hybrid call.

**Actions**:
1. New `search_chunks_hybrid(table, query_vector, query_text, ...)` calling LanceDB's hybrid search.
2. Switch `_dense_ranked_candidates` to use it; remove the separate FTS lane (the rerank + RRF fusion above it stays).

**Verify**:
- `pixi run eval-gate` — should match #1's result (this is a refactor for cleanliness, not a quality change).
- Latency benchmark: ~10-30% faster than two parallel queries.

**Effort**: 3 h.

---

#### `[#5]` Streaming synthesis

**Why**: User waits for full answer to materialise. MCP supports streaming responses; partial answers are immediately useful for "what does X mean?" type queries.

**Files**:
- `src/lxd/synthesis/answering.py:60-104` — `synthesize_answer` is sync one-shot; needs an async streaming variant.
- `src/lxd/mcp/tools.py` — `search_knowledge` etc. need to yield partial chunks.
- FastMCP 3.0 streaming patterns (need to check current MCP SDK support).

**Actions**:
1. New `stream_synthesize_answer(question, evidence, config)` async generator yielding partial text + final envelope.
2. Update `search_knowledge` and `search_knowledge_deep` to support streaming via FastMCP's async iterator pattern.
3. Behind a `synthesis.stream` config flag.

**Verify**:
- Manual: query via MCP client; see partial output before full answer.
- Existing non-streaming tests unchanged.

**Effort**: 3 h.

---

### Tier 6 — Polish, testing, modernisation

#### `[#16]` Property-based tests for RRF, lexical, chunking

**Why**: `hypothesis` is installed but not used. RRF score + chunking invariants + lexical scoring are perfect property targets.

**Files**: `tests/test_query_pipeline.py` (extend), `tests/test_chunking_properties.py` (new).

**Actions**:
1. Property: RRF score is monotonic in rank.
2. Property: chunking preserves text length within `±chunk_overlap` of input length.
3. Property: lexical fusion never produces negative scores.

**Effort**: 4 h.

---

#### `[#18]` `asyncio.TaskGroup` in `llm_client.py:254`

**Why**: Currently uses `asyncio.gather(...)` (pre-3.11 pattern). `TaskGroup` gives proper cancellation semantics + `ExceptionGroup` integration. Already aligned with the embedder pattern we adopted.

**Files**: `src/lxd/ingest/llm_client.py:254`.

**Effort**: 1 h.

---

#### `[#19]` `@override`, `Self`, PEP 695 `type` aliases

**Why**: Cosmetic but keeps the codebase modern. `type X = Y` (not `X: TypeAlias = Y`) for Python 3.14.

**Files**: `src/lxd/stores/llm_jobs.py:32` (`JobStatus`), `src/lxd/domain/status.py`, settings models with overridden Pydantic hooks.

**Effort**: 2 h.

---

#### `[#20]` PEP 750 t-string prompts

**Why**: t-strings give automatic interpolation safety and structured logging integration. Refactor `_build_prompt` and the `_RELATION_BASE_PROMPT` / `_CLAIM_BASE_PROMPT` builders.

**Files**: `src/lxd/synthesis/answering.py:_build_prompt`, `src/lxd/ingest/relations.py:_RELATION_BASE_PROMPT`, `src/lxd/ingest/claims.py:_CLAIM_BASE_PROMPT`.

**Effort**: 3 h.

---

## Multi-Session Schedule

Sessions are ordered for clean ROI sequencing; each ends with a green build + commit. Some sessions can run in parallel if a teammate joins, but the eval gate (Session 2) is a hard prerequisite for Sessions 3–10.

| # | Session | Items | Effort |
|---|---|---|---|
| **1** | **Foundation: cleanup + wiki rebuild + baseline** | `#15`, plus complete the deferred wiki swap rebuild (run `pixi run preflight && ingest --full && build-graph --full`) | 3 h |
| **2** | **Eval as CI gate** | `#6` | 4 h |
| **3** | **Retrieval upgrade** | `#1`, `#14` | 6-7 h |
| **4** | **KG signal lift — wiki edges** | `#3` (isolated to measure delta) | 4 h |
| **5** | **KG signal lift — centrality + community** | `#2` | 6 h |
| **6** | **Synthesis quality + streaming** | `#4`, `#5` | 5-6 h |
| **7** | **Production guardrails** | `#11`, `#12` (or delete OTel stub if not implementing) | 6 h |
| **8** | **Persistent breaker + refactor** | `#10`, `#7` | 6 h |
| **9** | **Backend dispatch refactor** | `#8` (after `#7` so the new modules absorb the change cleanly) | 2 h |
| **10** | **Advanced retrieval** | `#9`, `#17` | 10 h (split if needed) |
| **11** | **Hybrid native + polish** | `#13`, `#16`, `#18`, `#19`, `#20` | 13 h (split if needed) |

**Total: ~65 h, 10–11 sessions.** Realistic for ~2 weeks of focused work, ~4 weeks part-time.

---

## To-do list — strictly ordered for execution

```
[ ] S1.1  Run `pixi run preflight && pixi run ingest --full && pixi run build-graph --full` — close the wiki swap
[ ] S1.2  Capture baseline: `pixi run eval > tests/eval/baseline-pre-fixes.json`
[ ] S1.3  [#15] Delete `_sqlite_legacy_migrations.py`, replace caller with assertion guard
[ ] S1.4  Commit: "Close wiki swap + drop legacy migrations file"

[ ] S2.1  [#6] Add `--gate-against` flag + thresholds to `eval_command`
[ ] S2.2  [#6] New `pixi run eval-gate` task
[ ] S2.3  [#6] Wire pre-push hook (or CI workflow)
[ ] S2.4  Commit: "Add eval CI gate"

[ ] S3.1  [#1] Add `create_fts_index(["text"])` to `open_chunk_table`
[ ] S3.2  [#1] Replace `_lexical_signal_score` with LanceDB FTS5 query
[ ] S3.3  [#1] Eval: confirm recall lift
[ ] S3.4  [#14] Add `RerankerConfig.backend` discriminated union
[ ] S3.5  [#14] Implement Cohere + Voyage adapters; deprecate auto-spawn path
[ ] S3.6  [#14] Eval: confirm rerank lift
[ ] S3.7  Commit: "Hybrid retrieval + remote rerank"

[ ] S4.1  [#3] New `wiki_relations.py` with slug-to-canonical-id mapping
[ ] S4.2  [#3] Emit `wiki_references` and `wiki_cites` relations during ingest
[ ] S4.3  [#3] Add dangling-slug report to `pixi run status`
[ ] S4.4  [#3] Eval: confirm graph-question lift
[ ] S4.5  Commit: "Map wiki [[slug]] cross-refs into knowledge graph"

[ ] S5.1  [#2] Add `central_entity_score`, `community_ids` to `RankedChunk`
[ ] S5.2  [#2] Load centrality + community in `_dense_ranked_candidates`
[ ] S5.3  [#2] Implement community-aware MMR (`_diversify_by_community`)
[ ] S5.4  [#2] Add centrality-weighted RRF lane
[ ] S5.5  [#2] Eval: confirm broad-query diversity lift
[ ] S5.6  Commit: "Use centrality + community signals in retrieval"

[ ] S6.1  [#4] Replace synthesis Ollama-only call with `call_with_fallback_async`
[ ] S6.2  [#4] Default config: OpenAI primary
[ ] S6.3  [#4] Eval (manual rubric): synthesis quality
[ ] S6.4  [#5] Stream synthesise_answer; surface via FastMCP iterator
[ ] S6.5  Commit: "Synthesis: OpenAI primary + streaming"

[ ] S7.1  [#11] New `IngestBudget` config + threshold checks
[ ] S7.2  [#11] Test: ingest aborts cleanly on budget exceeded
[ ] S7.3  [#12] Decide: implement OTel or delete unused settings
[ ] S7.4  [#12] If implementing: configure_tracing, span wrappers in run_tool, span at ingest+retrieval boundaries
[ ] S7.5  [#12] Validate spans in local Jaeger
[ ] S7.6  Commit: "Production guardrails: budget + observability"

[ ] S8.1  [#10] Add `circuit_breaker_state` table + migration v7
[ ] S8.2  [#10] Implement `PersistentCircuitBreaker`
[ ] S8.3  [#10] Test: state survives process restart
[ ] S8.4  [#7] Create `stores/sqlite/` subpackage
[ ] S8.5  [#7] Move modules per the 9-group split; preserve public API via __init__
[ ] S8.6  [#7] Run full test suite; lint; typecheck
[ ] S8.7  Commit: "Persistent breaker + sqlite.py split"

[ ] S9.1  [#8] Define backend discriminated unions in settings/models.py
[ ] S9.2  [#8] Replace if-chains with match/case in relations.py, claims.py, llm_client.py
[ ] S9.3  Commit: "Backend dispatch: discriminated unions + match/case"

[ ] S10.1 [#9] Implement contextual retrieval chunker (chunk-summary prepend)
[ ] S10.2 [#9] A/B vs hybrid_docling; document winner
[ ] S10.3 [#17] Implement HyDE pre-retrieval step
[ ] S10.4 [#17] A/B; ship behind config flag
[ ] S10.5 Commit: "Contextual retrieval + HyDE"

[ ] S11.1 [#13] LanceDB native hybrid query (replaces parallel two-lane)
[ ] S11.2 [#16] Property-based tests for RRF / lexical / chunking
[ ] S11.3 [#18] asyncio.TaskGroup in llm_client.py
[ ] S11.4 [#19] @override / Self / PEP 695 type aliases sweep
[ ] S11.5 [#20] PEP 750 t-string prompts in synthesis + extraction
[ ] S11.6 Commit: "Native hybrid query + polish + Python 3.14 idioms"
```

---

## Verification (overall)

Each session ends with the same gate:
```bash
pixi run lint && pixi run typecheck && pixi run test && pixi run eval-gate
```

Quality-affecting sessions (#3, #4, #5, #6, #9, #10) additionally:
1. Manual spot-check via MCP: 5 representative queries before/after.
2. Compare `pixi run eval` JSON output against baseline; record delta in commit message.
3. Refresh `tests/eval/baseline.json` only after the change is verified to win.

Refactor sessions (#7, #8) additionally:
1. `git grep` confirms no public API shifted.
2. Module sizes after split: each <300 LOC.

End-to-end smoke (any session that touches retrieval/synthesis):
```bash
pixi run mcp &
# In another shell:
mcp-client search_knowledge "What does ADDIE owe to backward design?"
mcp-client search_corpus "cognitive load theory" --limit 5
mcp-client get_community_context --entity_id "addie-model"
```

---

## Critical files to be modified (by item)

| Item | Files |
|---|---|
| #1 | `src/lxd/stores/lancedb.py:33-52`, `src/lxd/retrieval/query_pipeline.py:329-373,459-487` |
| #2 | `src/lxd/retrieval/query_pipeline.py:72-98,329-373,421-456`, `src/lxd/retrieval/graph_routing.py:33-105` |
| #3 | `src/lxd/ingest/wiki_metadata.py:56-113`, `src/lxd/ingest/wiki_relations.py` (new), `src/lxd/stores/sqlite.py:1590-1621` |
| #4 | `src/lxd/synthesis/answering.py:60-104`, `src/lxd/settings/models.py` |
| #5 | `src/lxd/synthesis/answering.py:60-104`, `src/lxd/mcp/tools.py` |
| #6 | `src/lxd/cli/eval.py:16-58`, `tests/eval/eval_set.json`, `pixi.toml` |
| #7 | `src/lxd/stores/sqlite.py` → `src/lxd/stores/sqlite/__init__.py` + 9 modules |
| #8 | `src/lxd/settings/models.py`, `src/lxd/ingest/relations.py:381-402`, `src/lxd/ingest/claims.py:486`, `src/lxd/ingest/llm_client.py:181` |
| #9 | `src/lxd/ingest/chunking.py`, `src/lxd/ingest/contextual_chunker.py` (new) |
| #10 | `src/lxd/ingest/error_classification.py`, `src/lxd/stores/_base_ddl.py`, `src/lxd/stores/schema.py` (migration 7) |
| #11 | `src/lxd/settings/models.py`, `src/lxd/ingest/pipeline.py` |
| #12 | `src/lxd/observability/logging.py`, `src/lxd/mcp/async_runtime.py:38-72`, new `src/lxd/observability/tracing.py` |
| #13 | `src/lxd/stores/lancedb.py:103-149`, `src/lxd/retrieval/query_pipeline.py:329-373` |
| #14 | `src/lxd/retrieval/rerank.py:61-114,156-197`, `src/lxd/settings/models.py`, `src/lxd/net/http.py` |
| #15 | DELETE `src/lxd/stores/_sqlite_legacy_migrations.py`; edit `src/lxd/stores/sqlite.py:11,111` |
| #16 | `tests/test_query_pipeline.py`, new `tests/test_chunking_properties.py` |
| #17 | `src/lxd/retrieval/query_pipeline.py:206`, new `src/lxd/retrieval/hyde.py` |
| #18 | `src/lxd/ingest/llm_client.py:254` |
| #19 | `src/lxd/stores/llm_jobs.py:32`, `src/lxd/domain/status.py`, settings models |
| #20 | `src/lxd/synthesis/answering.py:_build_prompt`, `src/lxd/ingest/relations.py:_RELATION_BASE_PROMPT`, `src/lxd/ingest/claims.py:_CLAIM_BASE_PROMPT` |

---

## Reusable utilities already in the codebase

The plan deliberately reuses these rather than introducing parallel implementations:

- **`lxd.ingest.llm_client.call_with_fallback_async`** (lines 87+) — used today by relations and claims for OpenAI primary + Ollama fallback. Reused by **#4** (synthesis), **#17** (HyDE).
- **`lxd.net.http`** — pooled `httpx.Client` / `httpx.AsyncClient` factories. Reused by **#14** (remote rerank).
- **`lxd.stores.sqlite.replace_canonical_relations`** + **`replace_relation_evidence`** (lines 1590-1708) — bulk-insert targets. Reused by **#3** (wiki-link relations).
- **`lxd.retrieval.graph_routing.build_graph_context`** (lines 33-105) — already loads community reports + entity profiles. The same loader is lifted for retrieval-time community-aware MMR in **#2**.
- **`lxd.mcp.async_runtime.run_tool`** (lines 38-72) — central wrapper for sync→async tool bridging. Natural span insertion point for **#12** (OTel).
- **`lxd.ingest.error_classification.SystemicErrorCircuitBreaker`** — extended to `PersistentCircuitBreaker` in **#10**.
- **`lxd.retrieval.eval.run_eval`** (lines 117-160) — emits `EvalSummary` with `mean_recall_at_10`, `mean_mrr_at_10`. Reused by **#6** (CI gate).

---

## What this plan does NOT include

- **Embedding model swap (3-small → 3-large @1536)** — already discussed and decided against; gain too small versus the structural items here.
- **Multi-corpus support** (wiki + Knowledge_Base together) — was a possibility raised but the user chose wiki-only.
- **Full ontology re-tuning** against the wiki — flagged in audit but is its own multi-day effort.
- **SECURITY review** — no threat model, no input sanitisation pass. Out of scope for this plan; recommended as a separate review.
- **Plans/ design-spec drift** — CLAUDE.md was just refreshed; the `Plans/00–08*.md` specs may also be stale, and a separate audit/refresh is warranted but not included here.

---

## Tier 7 — Backlog (known debt beyond items 1–20)

The 2026-05-05 SOTA audit surfaced additional findings that are real but lower-ROI than items 1–20. They are **not scheduled** in the session plan above. Capturing them here so they don't get lost between audit and execution. Each entry is grouped by category, with file:line references where the audit was specific. Before promoting any backlog item into a session, re-evaluate ROI against the then-current state of the codebase — some items may be obviated by Tier 1–6 work.

### B-KG — Knowledge graph (further)

| ID | Finding | File / location | Note |
|---|---|---|---|
| `B-KG-1` | **Claim verification / contradiction detection** | `src/lxd/ingest/claims.py` (extraction); no consumer exists | Claims are extracted and stored but never cross-checked. A contradicting pair (claim A: X→Y; claim B: X→¬Y) sits silently in the corpus and amplifies hallucination risk. Would need a post-extraction pass that flags contradictions for human review. |
| `B-KG-2` | **Entity disambiguation is naive** | `src/lxd/ontology/matcher.py` (Aho-Corasick); `src/lxd/ingest/mentions.py` | Surface form → entity_id is exact-match. No fuzzy matching, no embedding-based mention disambiguation. Acronyms with multiple expansions (e.g. "ID" = instructional design vs identifier) resolve by first-rule-wins. |
| `B-KG-3` | **`entity_embeddings` table built but never queried at retrieval time** | `src/lxd/stores/lancedb.py:232` (table); used only by `get_similar_entities` MCP tool | Could drive query expansion: query embed → top-k entities by cosine → expand with related concepts. |
| `B-KG-4` | **Graph context has no token budget** | `src/lxd/retrieval/graph_routing.py:_build_graph_context_prompt` | Could push synthesis past the model's context window; needs explicit `max_graph_tokens` cap with a tier-based truncation order (entities → communities → claims). |
| `B-KG-5` | **Graph build is en-bloc per phase, not chunk-incremental** | `src/lxd/cli/graph.py` orchestrator | A new document still triggers full claim re-extraction for related entities. Should be additive at the chunk level. |

### B-CODE — Code structure (further)

| ID | Finding | File / location | Note |
|---|---|---|---|
| `B-CODE-1` | **`ingest/pipeline.py` is 1143 lines (21 funcs)** | `src/lxd/ingest/pipeline.py` | Item #7 splits `sqlite.py`; the same treatment applies here. Natural splits: scan/diff, embed-with-cache, chunk-build, persist, move-detection, clone-records, snapshot. |
| `B-CODE-2` | **No `Pydantic TypeAdapter` for hot-path validation** | `src/lxd/stores/sqlite.py` row → record adapters | `manifest_record_from_row`, `chunk_from_row` etc. construct dataclasses directly. `TypeAdapter[ChunkRecord]` is faster on repeated parsing and gives validation for free. |
| `B-CODE-3` | **No `ComputedField`, `RootModel`, `BeforeValidator`/`AfterValidator`** | `src/lxd/settings/models.py` | Settings have one custom validator (`_normalize_query_instruction`); newer Pydantic v2 idioms would tighten the rest. |

### B-ROBUST — Robustness (further)

| ID | Finding | File / location | Note |
|---|---|---|---|
| `B-ROBUST-1` | **OpenAI sync client created per-batch** | `src/lxd/ingest/embedder.py:275` (`_openai_embed_texts`) | `client = openai.OpenAI(api_key=...)` runs inside the function. Bypasses our pooled `httpx` factory in `net/http.py`. Should pass `http_client=` into `openai.OpenAI(...)` referencing the shared pool. |
| `B-ROBUST-2` | **Aho-Corasick matcher rebuilt on every CLI invocation** | `src/lxd/ontology/matcher.py` + `src/lxd/retrieval/expansion.py:20` | Pickle to disk keyed on ontology hash. ~1-2s per CLI start; meaningful for short-running commands. |
| `B-ROBUST-3` | **Empty wiki frontmatter still triggers full mention / relation pipelines** | `src/lxd/ingest/pipeline.py` | No early exit for "this page has no extractable signal." Wasted API calls on edge-case pages. |

### B-PERF — Performance (further)

| ID | Finding | File / location | Note |
|---|---|---|---|
| `B-PERF-1` | **Tokenizer encodes full document text twice** | `src/lxd/ingest/chunking.py:84,188` | Same text fed through `tiktoken` in both code paths; cache the encoded token list. |
| `B-PERF-2` | **`_unique_source_prefix` re-iterates ranked list on each dense-search retry** | `src/lxd/retrieval/query_pipeline.py:408` | Use a set early; current pattern is O(N²) in the worst case. |
| `B-PERF-3` | **SQLite connection opened with WAL + tuned pragmas per call** | `src/lxd/stores/sqlite.py:48` (`connect_sqlite`) | A per-thread connection pool would amortise the pragma cost. Matters for the long-lived MCP server, less for one-shot CLI. |
| `B-PERF-4` | **Embedding cache lookup iterates per-chunk in Python** | `src/lxd/ingest/embedding_cache.py` (`lookup`) | Fine for batches of 1k. At 100k+ becomes the bottleneck. Vectorise by returning Arrow and joining in pyarrow. |

### B-STACK — Tech-stack underutilisation (further)

| ID | Capability | Status | Note |
|---|---|---|---|
| `B-STACK-1` | LanceDB scalar quantisation + IVF_PQ | unused | At current scale (~25k chunks) not needed; flag once we cross 1M chunks. |
| `B-STACK-2` | LanceDB version branching / time-travel queries | unused | Would let us A/B retrieval changes without rebuilding. |
| `B-STACK-3` | LanceDB secondary indexes on `source_domain`, `source_rel_path` | unused | Would speed up the per-source delete-and-replace path. |
| `B-STACK-4` | FastMCP **Resources** (e.g. `lxd://corpus/{path}`) | unused | Would let MCP clients fetch raw source files referenced in citations. |
| `B-STACK-5` | FastMCP **Prompts** (parameterised templates) | unused | Could expose `lxd_search_prompt` / `lxd_synthesis_prompt` to clients for transparency. |
| `B-STACK-6` | FastMCP **Sampling** (server-initiated LLM) | unused | Could let the server delegate LLM calls back to the client model. |
| `B-STACK-7` | FastMCP **structured tool input schemas** | partial | We accept loose dicts in some tools; tighter Pydantic schemas would improve client autocomplete and validation. |
| `B-STACK-8` | structlog `bind_contextvars` per request | partial | Used at startup; not propagated per MCP-tool-call. Item #12 (OTel) overlaps. |
| `B-STACK-9` | structlog sampled logging for high-volume events | unused | At ingest-scale chunk events flood the log. |
| `B-STACK-10` | `tiktoken` for budget-aware chunking, prompt truncation, pre-flight cost estimation | unused | Would let `pixi run preflight` show "this ingest will cost ~$X" before running. Pairs naturally with item #11. |
| `B-STACK-11` | NetworkX advanced (HITS, TF-IDF weighted paths, k-core, motif detection) | unused | Would unlock new graph queries; worth surfacing once item #2 lands and centrality starts paying off. |
| `B-STACK-12` | Polars / Arrow-native DataFrames | unused | LanceDB returns Arrow natively; some KG analyses currently round-trip through SQLite that Polars-on-Arrow would do in microseconds. |

### B-TEST — Testing (further)

| ID | Finding | File / location | Note |
|---|---|---|---|
| `B-TEST-1` | **No synthesis end-to-end test** | `tests/` | The LLM is mocked everywhere. We don't actually verify the synthesis prompt produces sensible output. Pair with `pixi run eval` end-to-end. |
| `B-TEST-2` | **No mutation testing** | (n/a) | Not normally needed at this scale, but `mutmut` against `query_pipeline` would surface dead branches. Low priority. |

### Summary

- **B-KG**: 5 items
- **B-CODE**: 3 items
- **B-ROBUST**: 3 items
- **B-PERF**: 4 items
- **B-STACK**: 12 items
- **B-TEST**: 2 items

**Total backlog: 29 additional items beyond the 20 in the executable plan.**

These are not scheduled. When a session opens with bandwidth, pick a backlog item that complements the just-finished work (e.g. `B-STACK-10` after item #11 because both touch cost estimation; `B-KG-3` after item #2 because both leverage the centrality work). Promote the chosen item into the next session header and update this backlog section with a strikethrough or "promoted to S<N>" annotation.

---

*Plan created: 2026-05-05. Source audit: 2026-05-05 SOTA review. Prerequisite: complete the deferred wiki swap rebuild (Session 1.1 first). Backlog (Tier 7) added after first review — captures audit findings not in the 20-item executable plan.*
