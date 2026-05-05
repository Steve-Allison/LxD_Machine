# LxD Machine — SOTA Implementation Plan (18 items)

## Context

A 2026-05-05 critical audit identified 20 architectural and code-quality gaps across retrieval, knowledge-graph signal use, code structure, robustness, and observability. The audit grounded each finding in `file:line` references; an Explore-agent pass against the live codebase has now confirmed exact integration points for every item.

This plan organises the 18 remaining items (item #6 was a measurement-chore item not a SOTA improvement, removed 2026-05-05; item #14 was a remote-rerank proposal incompatible with the local-first design, struck 2026-05-05) into a multi-session implementation schedule prioritised by **return on investment** (impact ÷ effort), with one structural sequencing rule: **foundation before refinement**. Dead-code removal and architectural consolidation precede new features so we don't refactor newly-added code twice.

**Important — measurement is the user's call, not a plan-imposed gate.** This plan does NOT run ingest, does NOT capture eval baselines, and does NOT impose `pixi run eval` as a CI gate. Each item's verification is code-level: lint, typecheck, targeted tests, manual MCP smoke where applicable. The user runs ingest and eval when they choose; quality measurement is layered on top of this work, not gated by it.

The fixed-cost expectation: **~55 hours over ~9 sessions**, half-day to full-day each. Each session ends with green local checks (lint + typecheck + tests) and a single commit. Sessions are not blocking — items within a session can be reordered.

---

## Items — Reordered by Priority (ROI)

The original audit numbering is preserved in `[#N]` for traceability; the order below is the implementation order. Item #6 (eval as CI gate) was removed: it is a measurement chore, not a SOTA improvement. Quality measurement is the user's call, layered on top of this work, not gated by it.

### Tier 1 — Foundation: delete every legacy and dead artefact

This tier is a precondition for the rest of the plan. Modernising on top of legacy doubles the surface area we have to touch; we delete first so each subsequent SOTA pattern lands in a clean codebase.

**Principles** (apply to every item below, but called out here because Tier 1 enforces them):

- **No legacy.** Pre-versioning code paths, stale schema bridges, dead branches — gone.
- **No backwards-compatibility shims.** No re-export façades, no compat aliases for renamed config fields, no `def old(*a, **kw): return new(*a, **kw)` wrappers, no "deprecated, still works" docstrings.
- **No tech debt accumulators.** No `# TODO: clean up`, no `# FIXME later`, no `# kept for compatibility`, no half-finished refactors.
- **Forward-only migrations.** The user runs rebuilds when *they* choose; we do not preserve old schema migrations to spare them.

#### `[#15]` Delete every legacy and dead artefact across the codebase

**Status**: ✅ **DONE 2026-05-05.** Seven commits across `src/` + `Plans/` + top-level docs. End-state assertion holds: `git grep -E '#\s*(legacy|deprecated|TODO|FIXME|HACK|XXX|back-compat|backwards|kept for compat)' src/` returns zero hits; `git grep -E 'Phase [0-9]+|Wave [0-9]+' AGENTS.md CLAUDE.md` returns zero hits. Lint, typecheck, all 201 tests green throughout.

**Commits:**

- `9227861` — drop `_sqlite_legacy_migrations.py` (887 LOC) + replace with 14-line `assert_no_v2_legacy_tables` guard
- `d1cc27d` — drop dead `timestamp` parameter from `_resolve_document_id` ("retained for legacy callers" + immediate `del timestamp`)
- `e29fb7c` — delete `ObservabilityConfig` (false-promise OTel/Prometheus settings: declared, never read at runtime); remove its dead test
- `d524f82` — strip historical "Phase 5" / "Wave 5+" tags from durable code (4 source files)
- `0830fed` — strip stale framing from `Plans/` specs (`01_ARCHITECTURE.md`, `02_DATA_SCHEMA.md`, `02b_CONFIG_SPEC.md`, `05_MCP_SPEC.md`)
- `e1d9529` — delete `Plans/06_BUILD_PLAN.md` (a completed build plan written as forward-looking); strip "(Phase 5)" from `08_KNOWLEDGE_GRAPH_SPEC.md` title and 5 "Phases 0–4" external references
- `d09e0fd` — strip "(Phase 4)" / "(Phase 5)" tool-group labels from `CLAUDE.md` and `AGENTS.md`
- `1a3f126` — rename misleading `_for_legacy_snapshot` test (the function it tests is a generic fallback, not a legacy migration)

**Survey results — codebase was already cleaner than the audit suggested:**

- Zero `# TODO`, `# FIXME`, `# HACK`, `# XXX`, `# deprecated`, `# noqa-without-justification` hits.
- Zero pre-3.13 idioms (`Optional[X]`, `TypeAlias`, `lru_cache`, `utcnow`, `os.path`).
- Zero `pytest.mark.skip("legacy")` / dead-skipped tests.
- Zero `if False:`, `pragma: no cover`, dead `raise NotImplementedError`.
- Zero re-export shims / compat aliases in any `__init__.py`.
- Zero `_legacy_*` / `_old_*` / `_deprecated_*` / `_v2_*` symbol names (the `*_v2_legacy` table-name references are real, current regression-test code for the migration v4 ghost-FK repair).
- The existing `# type: ignore[...]` comments are all legitimate (graspologic optional import; ahocorasick missing type args; etc.).

**Backlog item flagged for Tier 7:** `Plans/08_KNOWLEDGE_GRAPH_SPEC.md` still uses `Phase 5.0` … `Phase 5.8` as internal subsection identifiers (~30 cross-references + a dependency diagram). The deeper rename to descriptive step names is a dedicated doc-rewrite project.

**Why first**: We already proved the value on `_sqlite_legacy_migrations.py` — a clean delete plus a 14-line guard replaced 887 lines and removed the trap of two parallel migration paths. The same trap exists in any module that still carries a "legacy" / "deprecated" / "back-compat" surface; before we land any new SOTA pattern, we kill all of those.

**Scope** (the sweep finds these, not a fixed list):

1. **Markers** — `# legacy`, `# deprecated`, `# back-compat`, `# backwards`, `# kept for compat`, `# was`, `# previously`, `# TODO`, `# FIXME`, `# XXX`, `# HACK`, `# noqa` without a justifying comment. Each occurrence is either *finished* or *deleted*.
2. **Compat shims** — wrapper functions whose only job is to call a renamed implementation; aliased re-exports in `__init__.py` files; `model_validator` shims that map old config field names to new ones.
3. **Dead branches** — `if False:`, unreachable `else` arms after `raise`, error handlers for impossible scenarios, optional toggles where one branch is never taken in practice.
4. **Unused symbols** — functions, classes, imports, constants flagged by `ruff --select F401,F811,F841` and by reading each module top-to-bottom.
5. **Pre-versioning migration paths** — anything older than the current schema floor. (`stores/_sqlite_legacy_migrations.py` is one such; we look for siblings.)
6. **Stale settings fields** — Pydantic model fields declared but never read at runtime (the audit already flagged `otel_enabled`, `otel_endpoint`, `prometheus_enabled`, `prometheus_port` as such).
7. **Stale tests** — tests that assert pre-wiki-swap state, tests for code that no longer exists, tests skipped with `pytest.mark.skip("legacy")`.
8. **Stale docs / specs** — any reference in `Plans/`, `CLAUDE.md`, `.claude/rules/`, or `README.md` that no longer reflects the codebase.

**Process**:

1. Survey the codebase systematically (grep + read). Produce a categorised list of every candidate.
2. Group candidates into coherent batches (e.g. "stale OTel settings", "compat aliases in `settings/models.py`", "dead branches in `retrieval/`"). Each batch becomes one commit.
3. For each batch: delete; update callers; run lint + typecheck + tests; commit. No batch lands red.
4. End-state assertion: `git grep -E '#\s*(legacy|deprecated|TODO|FIXME|HACK|XXX|back-compat|backwards|kept for compat)'` returns zero hits in `src/`. Same for `pixi run lint` reports of unused symbols.

**Verify**: `pixi run lint && pixi run typecheck && pixi run test` clean after every batch. `git grep` for legacy markers returns empty. `vulture src/` (one-shot manual check; not added as a permanent dep) reports zero high-confidence unused functions.

**Effort**: 4-6 h (most of Session 1). Bigger than originally scoped; reflects the codebase-wide reach.

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
- New unit test in `tests/test_query_pipeline.py`: FTS5 lane returns BM25-scored candidates that fuse correctly into the existing RRF prefix.
- New integration test: full hybrid query via `answer_question` returns chunks; lexical-heavy queries route through the BM25 lane.
- `pixi run lint && pixi run typecheck && pixi run test` clean.

**Effort**: 3-4 h.

---

#### `[#14]` ❌ STRUCK 2026-05-05 — "Switch to remote rerank API" was wrong for this project

The original audit recommendation was: swap the local `llama-server` + `qwen3-reranker:0.6b` reranker for Cohere `/v2/rerank` or Voyage `/v1/rerank`.

**This is permanently rejected.** The user has stated as a hard rule: *"NEVER use a non-local reranker — remove that COMPLETELY and go back to ollama and the local reranker. If there is an issue with that code you should have raised it as an issue."*

LxD is a local-first system: embeddings (Ollama / OpenAI batch), synthesis (Ollama), vector store (LanceDB), metadata store (SQLite), MCP over stdio. The reranker stays local. A remote rerank-as-an-API call on every retrieval would break the local-first guarantee, add per-query API spend, and introduce a remote-failure mode on the read path. Saved as feedback rule `feedback_local_only_no_remote_rerank.md`.

The legitimate observation buried under [#14] — that `rerank.py` auto-spawns `llama-server` from inside a query path — is a *local-code* concern, not a reason to leave the machine. It is parked in **Tier 7 backlog as `B-LOCAL-1`** as an issue to *raise and discuss with the user before changing anything*, not a silent rewrite.

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
- New tests in `tests/test_wiki_relations.py` for slug normalisation, canonical_id matching, citation tuple shape.
- Integration test: ingest a fixture wiki page with `[[slug]]` references, assert relations land in `extracted_relations` with `extraction_model="wiki_metadata"`.
- `pixi run lint && pixi run typecheck && pixi run test` clean.

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
- Tests for `_diversify_by_community` (deterministic given fixed candidate set + community map).
- Tests for the centrality-weighted RRF lane (deterministic given fixed centrality scores + base ranking).
- `pixi run lint && pixi run typecheck && pixi run test` clean.

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
- Existing tests in `tests/test_synthesis_answering.py` (or equivalent) still pass with mocked LLM.
- New test: synthesis backend dispatch is exercised for both backends via `call_with_fallback_async`.
- `pixi run lint && pixi run typecheck && pixi run test` clean.

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
- New tests for the contextual chunker: chunk text is unchanged in storage; only the embedded-form is augmented.
- Integration test: a fixture document round-trips through the new chunker, embeddings differ from baseline (proves prepend works), retrieval still returns the right chunk.
- `pixi run lint && pixi run typecheck && pixi run test` clean.

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
- New unit test for `hypothetical_answer` (mocked LLM): returns a string under the token budget; emits the right prompt shape.
- Integration test: HyDE-enabled query path retrieves on the hypothetical-answer embedding instead of the literal question.
- `pixi run lint && pixi run typecheck && pixi run test` clean.

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
- Existing retrieval tests still pass (this is a refactor for cleanliness, not a behaviour change).
- New unit test against the hybrid query API confirms the result shape.
- Latency micro-benchmark in `tests/`: native hybrid is at least as fast as two parallel queries plus manual fusion.
- `pixi run lint && pixi run typecheck && pixi run test` clean.

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

Sessions are ordered for clean ROI sequencing; each ends with a green build + commit. Sessions are not blocking — items within a session can be reordered.

| #      | Session                                     | Items                                                                                                                             | Effort                 |
| ------ | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| **1**  | **Delete every legacy and dead artefact**   | `#15` (codebase-wide sweep: legacy markers, compat shims, dead branches, unused symbols, stale settings, stale tests, stale docs) | 4-6 h                  |
| **2**  | **Retrieval upgrade**                       | `#1` (`#14` removed — see below)                                                                                                  | 3-4 h                  |
| **3**  | **KG signal lift — wiki edges**             | `#3`                                                                                                                              | 4 h                    |
| **4**  | **KG signal lift — centrality + community** | `#2`                                                                                                                              | 6 h                    |
| **5**  | **Synthesis quality + streaming**           | `#4`, `#5`                                                                                                                        | 5-6 h                  |
| **6**  | **Production guardrails**                   | `#11`, `#12` (implement OTel or delete unused fields outright per the no-tech-debt rule)                                          | 6 h                    |
| **7**  | **Persistent breaker + sqlite.py split**    | `#10`, `#7` (callers update; no `__init__.py` re-export façade)                                                                   | 6 h                    |
| **8**  | **Backend dispatch refactor**               | `#8` (after `#7` so the new modules absorb the change cleanly)                                                                    | 2 h                    |
| **9**  | **Advanced retrieval**                      | `#9`, `#17`                                                                                                                       | 10 h (split if needed) |
| **10** | **Hybrid native + polish**                  | `#13`, `#16`, `#18`, `#19`, `#20`                                                                                                 | 13 h (split if needed) |

**Total: ~55 h, 9 sessions.** Realistic for ~2 weeks of focused work, ~4 weeks part-time.

---

## To-do list — strictly ordered for execution

```
[x] S1   [#15] Codebase-wide legacy and dead-code purge — done 2026-05-05 across 8 commits (`9227861`, `d1cc27d`, `e29fb7c`, `d524f82`, `0830fed`, `e1d9529`, `d09e0fd`, `1a3f126`). End-state assertion holds. Detail in the item header above.

[x] S2.1  [#1] Add `create_fts_index(["text"])` to `open_chunk_table` — done in commit `95ca6f6`.
[x] S2.2  [#1] Replace hand-rolled `_lexical_signal_score` with LanceDB FTS5 query — done in commit `95ca6f6`. Hand-rolled scoring helpers (`_lexically_ranked`, `_lexical_signal_score`, `_significant_query_terms`, `_normalize_ranking_text`, `_contains_rank_term`, `_GENERIC_QUERY_TERMS`, `_QUERY_STOPWORDS`) all deleted; `import re` dropped.
[~] S2.3–S2.4  [#14] **STRUCK 2026-05-05.** Remote-rerank work removed permanently; this is a local-only project. The local-code observation about `rerank.py` auto-spawning `llama-server` from inside a query path is parked in Tier 7 as `B-LOCAL-1` (an issue to raise, not a silent swap).
[x] S2.5  Commit message: "[#1] LanceDB native FTS5 BM25 replaces hand-rolled lexical scoring" (`95ca6f6`).

[x] S3.1  [#3] New `wiki_relations.py` with slug-to-canonical-id mapping — done in `94308d3`. 4-form normalisation (kebab/snake × upper/lower) covers either ontology convention.
[x] S3.2  [#3] Emit `wiki_references` relations during ingest — done in `94308d3`. `extraction_model="wiki_metadata"` distinguishes from LLM-extracted edges. The original plan listed `wiki_cites` (Sources → entities) too; not implemented because source filenames do not correspond to ontology entities — the chunk-row `cited_sources_json` column already carries citations through to retrieval/synthesis without a synthetic relation layer.
[x] S3.3  [#3] Dangling-slug + pages-without-subject diagnostics — done in `94308d3` as a structlog `wiki_relation_derivation_diagnostics` event at end of run (truncated to first 20 each, sorted). Adding the same surface to `pixi run status` is a polish task — left for B-DOCS or a follow-up.
[x] S3.4  Commit: `94308d3` — "[#3] Wiki [[slug]] cross-refs become wiki_references KG edges".

[ ] S4.1  [#2] Add `central_entity_score`, `community_ids` to `RankedChunk`
[ ] S4.2  [#2] Load centrality + community in `_dense_ranked_candidates`
[ ] S4.3  [#2] Implement community-aware MMR (`_diversify_by_community`)
[ ] S4.4  [#2] Add centrality-weighted RRF lane
[ ] S4.5  Commit: "Use centrality + community signals in retrieval"

[ ] S5.1  [#4] Replace synthesis Ollama-only call with `call_with_fallback_async`; default OpenAI primary
[ ] S5.2  [#5] Stream `synthesise_answer`; surface via FastMCP iterator
[ ] S5.3  Commit: "Synthesis: OpenAI primary + streaming"

[ ] S6.1  [#11] New `IngestBudget` config + threshold checks
[ ] S6.2  [#12] Implement OTel via `configure_tracing`, span wrappers in `run_tool`, spans at ingest+retrieval boundaries (or delete the unused `otel_*`/`prometheus_*` settings outright per no-tech-debt — pick one, ship one)
[ ] S6.3  Commit: "Production guardrails: budget + observability"

[ ] S7.1  [#10] Add `circuit_breaker_state` table + migration v7
[ ] S7.2  [#10] Implement `PersistentCircuitBreaker`; **delete** the in-memory `SystemicErrorCircuitBreaker` (no parallel implementations)
[ ] S7.3  [#7] Split `stores/sqlite.py` into a subpackage. Callers update their imports — no `__init__.py` re-export façade.
[ ] S7.4  Commit: "Persistent breaker + sqlite.py split"

[ ] S8.1  [#8] Define backend discriminated unions in `settings/models.py`
[ ] S8.2  [#8] Replace if-chains with `match`/`case` in `relations.py`, `claims.py`, `llm_client.py`
[ ] S8.3  Commit: "Backend dispatch: discriminated unions + match/case"

[ ] S9.1  [#9] Implement contextual retrieval chunker (chunk-summary prepend before embed)
[ ] S9.2  [#17] Implement HyDE pre-retrieval step
[ ] S9.3  Commit: "Contextual retrieval + HyDE"

[ ] S10.1 [#13] LanceDB native hybrid query — **delete** the parallel two-lane code path
[ ] S10.2 [#16] Property-based tests for RRF / lexical / chunking
[ ] S10.3 [#18] `asyncio.TaskGroup` in `llm_client.py`
[ ] S10.4 [#19] `@override` / `Self` / PEP 695 `type` aliases sweep
[ ] S10.5 [#20] PEP 750 t-string prompts in synthesis + extraction
[ ] S10.6 Commit: "Native hybrid query + polish + Python 3.14 idioms"
```

---

## Verification (overall)

Each session ends with the same gate:

```bash
pixi run lint && pixi run typecheck && pixi run test
```

Refactor sessions (#7) additionally:

1. After the `stores/sqlite.py` split, `git grep "from lxd.stores.sqlite import"` shows callers updated to the new submodule paths — there is no `__init__.py` re-export façade preserving the old import shape.
2. Each new module under `src/lxd/stores/sqlite/` is under 300 LOC.

End-to-end smoke (any session that touches retrieval/synthesis):

```bash
pixi run mcp &
# In another shell:
mcp-client search_knowledge "What does ADDIE owe to backward design?"
mcp-client search_corpus "cognitive load theory" --limit 5
mcp-client get_community_context --entity_id "addie-model"
```

**No `pixi run eval` gate.** Measurement is the user's call, run when *they* choose. The plan does not impose recall-delta verification or baseline-refresh chores.

---

## Critical files to be modified (by item)

| Item | Files                                                                                                                                              |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| #1   | `src/lxd/stores/lancedb.py:33-52`, `src/lxd/retrieval/query_pipeline.py:329-373,459-487`                                                           |
| #2   | `src/lxd/retrieval/query_pipeline.py:72-98,329-373,421-456`, `src/lxd/retrieval/graph_routing.py:33-105`                                           |
| #3   | `src/lxd/ingest/wiki_metadata.py:56-113`, `src/lxd/ingest/wiki_relations.py` (new), `src/lxd/stores/sqlite.py:1590-1621`                           |
| #4   | `src/lxd/synthesis/answering.py:60-104`, `src/lxd/settings/models.py`                                                                              |
| #5   | `src/lxd/synthesis/answering.py:60-104`, `src/lxd/mcp/tools.py`                                                                                    |
| #7   | `src/lxd/stores/sqlite.py` → `src/lxd/stores/sqlite/` subpackage (9 modules; callers updated, no re-export façade)                                 |
| #8   | `src/lxd/settings/models.py`, `src/lxd/ingest/relations.py:381-402`, `src/lxd/ingest/claims.py:486`, `src/lxd/ingest/llm_client.py:181`            |
| #9   | `src/lxd/ingest/chunking.py`, `src/lxd/ingest/contextual_chunker.py` (new)                                                                         |
| #10  | `src/lxd/ingest/error_classification.py`, `src/lxd/stores/_base_ddl.py`, `src/lxd/stores/schema.py` (migration 7)                                  |
| #11  | `src/lxd/settings/models.py`, `src/lxd/ingest/pipeline.py`                                                                                         |
| #12  | `src/lxd/observability/logging.py`, `src/lxd/mcp/async_runtime.py:38-72`, new `src/lxd/observability/tracing.py`                                   |
| #13  | `src/lxd/stores/lancedb.py:103-149`, `src/lxd/retrieval/query_pipeline.py:329-373`                                                                 |
| #14  | ❌ struck — see backlog `B-LOCAL-1`                                                                                                                |
| #15  | DELETE `src/lxd/stores/_sqlite_legacy_migrations.py`; edit `src/lxd/stores/sqlite.py:11,111`                                                       |
| #16  | `tests/test_query_pipeline.py`, new `tests/test_chunking_properties.py`                                                                            |
| #17  | `src/lxd/retrieval/query_pipeline.py:206`, new `src/lxd/retrieval/hyde.py`                                                                         |
| #18  | `src/lxd/ingest/llm_client.py:254`                                                                                                                 |
| #19  | `src/lxd/stores/llm_jobs.py:32`, `src/lxd/domain/status.py`, settings models                                                                       |
| #20  | `src/lxd/synthesis/answering.py:_build_prompt`, `src/lxd/ingest/relations.py:_RELATION_BASE_PROMPT`, `src/lxd/ingest/claims.py:_CLAIM_BASE_PROMPT` |

---

## Reusable utilities already in the codebase

The plan deliberately reuses these rather than introducing parallel implementations:

- **`lxd.ingest.llm_client.call_with_fallback_async`** (lines 87+) — used today by relations and claims for OpenAI primary + Ollama fallback. Reused by **#4** (synthesis), **#17** (HyDE).
- **`lxd.net.http`** — pooled `httpx.Client` / `httpx.AsyncClient` factories. Available for any future local- or remote-HTTP client work; **not** used to introduce remote replacements for locally-running components.
- **`lxd.stores.sqlite.replace_canonical_relations`** + **`replace_relation_evidence`** (lines 1590-1708) — bulk-insert targets. Reused by **#3** (wiki-link relations).
- **`lxd.retrieval.graph_routing.build_graph_context`** (lines 33-105) — already loads community reports + entity profiles. The same loader is lifted for retrieval-time community-aware MMR in **#2**.
- **`lxd.mcp.async_runtime.run_tool`** (lines 38-72) — central wrapper for sync→async tool bridging. Natural span insertion point for **#12** (OTel).
- **`lxd.ingest.error_classification.SystemicErrorCircuitBreaker`** — extended to `PersistentCircuitBreaker` in **#10**.

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

| ID       | Finding                                                                 | File / location                                                                       | Note                                                                                                                                                                                                                                                    |
| -------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `B-KG-1` | **Claim verification / contradiction detection**                        | `src/lxd/ingest/claims.py` (extraction); no consumer exists                           | Claims are extracted and stored but never cross-checked. A contradicting pair (claim A: X→Y; claim B: X→¬Y) sits silently in the corpus and amplifies hallucination risk. Would need a post-extraction pass that flags contradictions for human review. |
| `B-KG-2` | **Entity disambiguation is naive**                                      | `src/lxd/ontology/matcher.py` (Aho-Corasick); `src/lxd/ingest/mentions.py`            | Surface form → entity_id is exact-match. No fuzzy matching, no embedding-based mention disambiguation. Acronyms with multiple expansions (e.g. "ID" = instructional design vs identifier) resolve by first-rule-wins.                                   |
| `B-KG-3` | **`entity_embeddings` table built but never queried at retrieval time** | `src/lxd/stores/lancedb.py:232` (table); used only by `get_similar_entities` MCP tool | Could drive query expansion: query embed → top-k entities by cosine → expand with related concepts.                                                                                                                                                     |
| `B-KG-4` | **Graph context has no token budget**                                   | `src/lxd/retrieval/graph_routing.py:_build_graph_context_prompt`                      | Could push synthesis past the model's context window; needs explicit `max_graph_tokens` cap with a tier-based truncation order (entities → communities → claims).                                                                                       |
| `B-KG-5` | **Graph build is en-bloc per phase, not chunk-incremental**             | `src/lxd/cli/graph.py` orchestrator                                                   | A new document still triggers full claim re-extraction for related entities. Should be additive at the chunk level.                                                                                                                                     |

### B-CODE — Code structure (further)

| ID         | Finding                                                                 | File / location                                  | Note                                                                                                                                                                      |
| ---------- | ----------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `B-CODE-1` | **`ingest/pipeline.py` is 1143 lines (21 funcs)**                       | `src/lxd/ingest/pipeline.py`                     | Item #7 splits `sqlite.py`; the same treatment applies here. Natural splits: scan/diff, embed-with-cache, chunk-build, persist, move-detection, clone-records, snapshot.  |
| `B-CODE-2` | **No `Pydantic TypeAdapter` for hot-path validation**                   | `src/lxd/stores/sqlite.py` row → record adapters | `manifest_record_from_row`, `chunk_from_row` etc. construct dataclasses directly. `TypeAdapter[ChunkRecord]` is faster on repeated parsing and gives validation for free. |
| `B-CODE-3` | **No `ComputedField`, `RootModel`, `BeforeValidator`/`AfterValidator`** | `src/lxd/settings/models.py`                     | Settings have one custom validator (`_normalize_query_instruction`); newer Pydantic v2 idioms would tighten the rest.                                                     |

### B-ROBUST — Robustness (further)

| ID           | Finding                                                                     | File / location                                                     | Note                                                                                                                                                                                                    |
| ------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `B-ROBUST-1` | **OpenAI sync client created per-batch**                                    | `src/lxd/ingest/embedder.py:275` (`_openai_embed_texts`)            | `client = openai.OpenAI(api_key=...)` runs inside the function. Bypasses our pooled `httpx` factory in `net/http.py`. Should pass `http_client=` into `openai.OpenAI(...)` referencing the shared pool. |
| `B-ROBUST-2` | **Aho-Corasick matcher rebuilt on every CLI invocation**                    | `src/lxd/ontology/matcher.py` + `src/lxd/retrieval/expansion.py:20` | Pickle to disk keyed on ontology hash. ~1-2s per CLI start; meaningful for short-running commands.                                                                                                      |
| `B-ROBUST-3` | **Empty wiki frontmatter still triggers full mention / relation pipelines** | `src/lxd/ingest/pipeline.py`                                        | No early exit for "this page has no extractable signal." Wasted API calls on edge-case pages.                                                                                                           |

### B-PERF — Performance (further)

| ID         | Finding                                                                        | File / location                                  | Note                                                                                                                       |
| ---------- | ------------------------------------------------------------------------------ | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `B-PERF-1` | **Tokenizer encodes full document text twice**                                 | `src/lxd/ingest/chunking.py:84,188`              | Same text fed through `tiktoken` in both code paths; cache the encoded token list.                                         |
| `B-PERF-2` | **`_unique_source_prefix` re-iterates ranked list on each dense-search retry** | `src/lxd/retrieval/query_pipeline.py:408`        | Use a set early; current pattern is O(N²) in the worst case.                                                               |
| `B-PERF-3` | **SQLite connection opened with WAL + tuned pragmas per call**                 | `src/lxd/stores/sqlite.py:48` (`connect_sqlite`) | A per-thread connection pool would amortise the pragma cost. Matters for the long-lived MCP server, less for one-shot CLI. |
| `B-PERF-4` | **Embedding cache lookup iterates per-chunk in Python**                        | `src/lxd/ingest/embedding_cache.py` (`lookup`)   | Fine for batches of 1k. At 100k+ becomes the bottleneck. Vectorise by returning Arrow and joining in pyarrow.              |

### B-STACK — Tech-stack underutilisation (further)

| ID           | Capability                                                                          | Status  | Note                                                                                                                                |
| ------------ | ----------------------------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `B-STACK-1`  | LanceDB scalar quantisation + IVF_PQ                                                | unused  | At current scale (~25k chunks) not needed; flag once we cross 1M chunks.                                                            |
| `B-STACK-2`  | LanceDB version branching / time-travel queries                                     | unused  | Would let us A/B retrieval changes without rebuilding.                                                                              |
| `B-STACK-3`  | LanceDB secondary indexes on `source_domain`, `source_rel_path`                     | unused  | Would speed up the per-source delete-and-replace path.                                                                              |
| `B-STACK-4`  | FastMCP **Resources** (e.g. `lxd://corpus/{path}`)                                  | unused  | Would let MCP clients fetch raw source files referenced in citations.                                                               |
| `B-STACK-5`  | FastMCP **Prompts** (parameterised templates)                                       | unused  | Could expose `lxd_search_prompt` / `lxd_synthesis_prompt` to clients for transparency.                                              |
| `B-STACK-6`  | FastMCP **Sampling** (server-initiated LLM)                                         | unused  | Could let the server delegate LLM calls back to the client model.                                                                   |
| `B-STACK-7`  | FastMCP **structured tool input schemas**                                           | partial | We accept loose dicts in some tools; tighter Pydantic schemas would improve client autocomplete and validation.                     |
| `B-STACK-8`  | structlog `bind_contextvars` per request                                            | partial | Used at startup; not propagated per MCP-tool-call. Item #12 (OTel) overlaps.                                                        |
| `B-STACK-9`  | structlog sampled logging for high-volume events                                    | unused  | At ingest-scale chunk events flood the log.                                                                                         |
| `B-STACK-10` | `tiktoken` for budget-aware chunking, prompt truncation, pre-flight cost estimation | unused  | Would let `pixi run preflight` show "this ingest will cost ~$X" before running. Pairs naturally with item #11.                      |
| `B-STACK-11` | NetworkX advanced (HITS, TF-IDF weighted paths, k-core, motif detection)            | unused  | Would unlock new graph queries; worth surfacing once item #2 lands and centrality starts paying off.                                |
| `B-STACK-12` | Polars / Arrow-native DataFrames                                                    | unused  | LanceDB returns Arrow natively; some KG analyses currently round-trip through SQLite that Polars-on-Arrow would do in microseconds. |

### B-LOCAL — Local-component code health (issues to RAISE, not silently swap)

These are local-code observations about the existing local stack. They are **not** scheduled. None of them are a license to swap a local component for a remote one — that is permanently forbidden per `feedback_local_only_no_remote_rerank.md`. Before touching any of these, raise the concern with the user, agree on the local fix, then act.

| ID | Finding | File / location | Note |
|---|---|---|---|
| `B-LOCAL-1` | **`rerank.py` auto-spawns `llama-server` from inside a query path** | `src/lxd/retrieval/rerank.py:_ensure_reranker_service`, `_build_llama_server_command`, `_load_running_pid`, `_resolve_ollama_blob_model_path`, `_runtime_paths`, `_write_pid_file`, `_wait_for_reranker_ready`, `_resolve_llama_server_executable`, `_resolve_reranker_model_path`, `_slugify`, `_tail_log`, `_process_is_running` | A search-time call may launch a long-running native process. Possible local-fix space (raise before changing): (a) move launch responsibility to `start.sh` / a new `pixi run reranker` task and have `rerank.py` be a pure HTTP client to a known URL; (b) keep auto-spawn but factor it out of the query path into a singleton/lifecycle hook; (c) leave as-is. **No remote replacement is on the table.** |

### B-DOCS — Documentation refactors (further)

| ID | Finding | File / location | Note |
|---|---|---|---|
| `B-DOCS-1` | **`08_KNOWLEDGE_GRAPH_SPEC.md` internal sub-phase numbering** | `Plans/08_KNOWLEDGE_GRAPH_SPEC.md` | Subsection headings still use `Phase 5.0` … `Phase 5.8` as build-wave identifiers, with ~30 in-doc cross-references and a dependency diagram. Title and external "(Phase 5)" / "Phases 0–4" framings already cleaned (commit `e1d9529`). The full internal rename to descriptive step names is its own dedicated rewrite project. |

### B-TEST — Testing (further)

| ID         | Finding                          | File / location | Note                                                                                                                                        |
| ---------- | -------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `B-TEST-1` | **No synthesis end-to-end test** | `tests/`        | The LLM is mocked everywhere. We don't actually verify the synthesis prompt produces sensible output. Pair with `pixi run eval` end-to-end. |
| `B-TEST-2` | **No mutation testing**          | (n/a)           | Not normally needed at this scale, but `mutmut` against `query_pipeline` would surface dead branches. Low priority.                         |

### Summary

- **B-KG**: 5 items
- **B-CODE**: 3 items
- **B-ROBUST**: 3 items
- **B-PERF**: 4 items
- **B-STACK**: 12 items
- **B-LOCAL**: 1 item
- **B-DOCS**: 1 item
- **B-TEST**: 2 items

**Total backlog: 31 additional items beyond the 18 in the executable plan (item #14 was struck 2026-05-05).**

These are not scheduled. When a session opens with bandwidth, pick a backlog item that complements the just-finished work (e.g. `B-STACK-10` after item #11 because both touch cost estimation; `B-KG-3` after item #2 because both leverage the centrality work). Promote the chosen item into the next session header and update this backlog section with a strikethrough or "promoted to S<N>" annotation.

---

*Plan created: 2026-05-05. Source audit: 2026-05-05 SOTA review. Reshaped 2026-05-05 to drop measurement ceremony (item #6 removed, no eval-gate, no baseline capture, no rebuild prereq) and to bake the user's "no legacy / no back-compat / no tech debt" principle into Tier 1 as a codebase-wide dead-code purge. Backlog (Tier 7) captures audit findings not in the 19-item executable plan.*
