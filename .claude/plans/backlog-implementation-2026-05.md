# LxD Machine — Backlog (Tier 7) Implementation Plan

## Execution status — 2026-05-05

**Sessions 1–7 complete (commits `880c596 → d01b70f`, all on `main`, pushed).** 270 tests passing (+12 net new across the run). Lint and pyright strict both green at every commit.

| Session | Items | Outcome | Commit |
|---|---|---|---|
| S1 | `B-CODE-4` — split `stores/sqlite.py` | shipped (9-module subpackage, 25 callers updated atomically) | `880c596` |
| S2 | `B-DOCS-1` — `08_KNOWLEDGE_GRAPH_SPEC.md` Phase 5 → Step 1–9 rename | shipped | `396f9bf` |
| S2b | `B-CODE-1` — split `ingest/pipeline.py` | shipped (4-module subpackage, 7 callers updated; cross-module helpers de-underscored) | `9119fb0` |
| S3 | `B-PERF-3` + `B-ROBUST-1` — pooled SQLite + pooled OpenAI client | shipped (per-thread pool, content-keyed cache, 6 new unit tests) | `868354f` |
| S4 | `B-PERF-1` + `B-PERF-2` | **closed obsolete-on-survey** — audit line numbers and shape claims did not match current code | `17cc3cb` |
| S5 | `B-CODE-3` — Pydantic v2 field-level validators | shipped (`BeforeValidator` for `query_instruction`, `AfterValidator` for `corpus_id`); `B-CODE-2` and `B-STACK-7` **struck** | `9c151eb` |
| S6 | `B-KG-3` — embedding-based entity expansion | shipped (always-on, no toggle, query-vector reused, 3 new tests); `B-STACK-11` **struck** | `4eb7d2e` |
| S7 | `B-KG-4` — graph context token budget | shipped (`max_graph_context_tokens=1500`, deterministic truncation order, 3 new tests); `B-KG-1` **deferred-pending-design** (audit assumes a `predicate` column on `claims` that does not exist) | `d01b70f` |

### Net delivery so far

- **8 items shipped**: B-CODE-1, B-CODE-3, B-CODE-4, B-DOCS-1, B-KG-3, B-KG-4, B-PERF-3, B-ROBUST-1.
- **5 items struck on survey**: B-CODE-2, B-PERF-1, B-PERF-2, B-STACK-7, B-STACK-11 — each audit description either pointed at code that no longer existed or proposed work that the current architecture already covered.
- **1 item deferred**: B-KG-1 — schema mismatch with the audit's premise; needs user direction between three honest design options (relations-based, claims-redundancy, LLM-adjudicated).

### Remaining sessions (not yet executed)

- **Session 8** — `B-KG-2` (entity disambiguation), `B-KG-5` (chunk-incremental graph build).
- **Session 9** — `B-STACK-4`, `B-STACK-5` (MCP capability surface).
- **Session 10** — `B-STACK-8/9/10`, `B-ROBUST-2/3` (observability + ingest polish).
- **Session 11** — `B-TEST-1` (synthesis e2e test).
- **RAISE-FIRST** (gated on user direction): `B-LOCAL-1`, `B-LOCAL-2`, `B-STACK-6`.
- **SCALE-DEFERRED** (gated on corpus / load growth): `B-PERF-4`, `B-STACK-1`, `B-STACK-2`, `B-STACK-12`, `B-TEST-2`.

The user has paused the run before Session 8 so each remaining session can be re-scoped against current code first — several remaining items have similar "design choice needed" or "audit framing now stale" properties (B-KG-5's own caveat warns of centrality-drift risk; B-STACK-4/5 only deliver value with a concrete consumer; B-STACK-8/9/10 may overlap with already-shipped observability surface).

---

## Context

This is the follow-on plan for the 33 Tier 7 backlog items captured during the 2026-05-05 SOTA review. The executable SOTA plan (`sota-fix-implementation-2026-05.md`) is now complete (12 items shipped, 5 struck, 1 deferred → backlog as `B-CODE-4`); this plan picks up the backlog.

**The same principles apply** as in the parent plan:

- **Local-only.** No remote replacements for local components (Ollama, llama-server, embeddings, synthesis). The local-fix space is open; the remote-replacement space is closed. See `feedback_local_only_no_remote_rerank.md`.
- **No legacy / no back-compat / no tech debt.** Replace outright; no shims, no dual paths, no re-export façades, no compat aliases.
- **Code-only SOTA.** No measurement chores (eval gates, baseline captures) embedded in the plan. The user runs ingest and eval when they choose.
- **No half-implementations.** Each item ships complete or doesn't ship in this session.
- **Strike if it doesn't fit.** Several items are explicitly marked DEFERRED-UNTIL-SCALE-CHANGES or RAISE-FIRST rather than implement-now.

Each item's verification is code-level: lint, typecheck, targeted tests, manual MCP smoke. No `pixi run eval-gate`.

The expected fixed cost: **~50–60 hours over ~10 sessions**, half-day to full-day each. Three items are explicitly flagged as RAISE-FIRST (B-LOCAL-1, B-LOCAL-2) or SCALE-DEFERRED (B-STACK-1, B-STACK-2, B-TEST-2) and don't enter the executable schedule until that gate clears.

---

## Items reordered by ROI

### Tier 1 — Architectural cleanup (refactor surface)

These reduce blast radius for every future change. Do them first so subsequent SOTA patterns land in cleaner modules.

#### `B-CODE-4` Split `stores/sqlite.py` (2085 LOC → 9-module subpackage)

**Why first**: Originally `[#7]` in the parent plan, deferred mid-session because format-hook stripping made an atomic 28-file commit fragile. With a fresh session it's the highest-ROI structural item.

**Files**:

- `src/lxd/stores/sqlite.py` (2085 LOC, ~70 functions) → 9 submodules under `src/lxd/stores/sqlite/`
- ~28 caller files: 20 in `src/`, 8 in `tests/` (every `from lxd.stores.sqlite import X` line updates).

**Module layout** (all caller imports become `from lxd.stores.sqlite._<n> import X`):

| Submodule | Functions | LOC est. |
|---|---|---|
| `_connection.py` | `connect_sqlite`, `build_store_paths`, `assert_no_v2_legacy_tables`, `initialize_schema`, `reset_store`, `_ensure_indexes` | ~120 |
| `_runs.py` | `begin_ingest_run`, `finish_ingest_run`, `update_ingest_run_progress` | ~155 |
| `_manifest.py` | `load_manifest_*`, `upsert_manifest_record`, `upsert_asset_link`, `store_has_committed_state`, `delete_source` | ~225 |
| `_ontology.py` | `replace_ontology_*`, `replace_ingest_config_snapshot`, `list_allowed_domains`, `load_*_snapshot` | ~175 |
| `_chunks.py` | `replace_source_chunks`, `load_chunk_records_*`, `load_mentions_for_source`, `find_chunks_by_entity_mentions`, `load_chunk_centrality_signals`, `load_relation_chunk_ids`, `load_all_extracted_relations`, `load_entity_mention_stats`, `load_chunk_ids_for_entity` | ~415 |
| `_summary.py` | `_summarize_*`, `summarize_store` | ~110 |
| `_claims.py` | claim insert/load/count | ~80 |
| `_kg_profiles.py` | `entity_profiles` + `entity_communities` + `community_reports` | ~270 |
| `_kg_relations.py` | canonical relations + relation_evidence + graph_build_state + graph_metadata | ~300 |

**Actions**:

1. Read the full 2085-line file in one go (no diff-from-context).
2. Use `git mv src/lxd/stores/sqlite.py /tmp/sqlite-original.py` to remove the file from the package namespace cleanly.
3. `mkdir src/lxd/stores/sqlite/` and write each submodule with its own imports, no cross-imports between submodules unless strictly necessary.
4. `__init__.py` is empty (no re-export façade per no-legacy rule). The package marker only.
5. Update all ~28 caller imports atomically in the same commit.
6. Delete `/tmp/sqlite-original.py` after lint+typecheck+tests are green.

**Verify**:

- `pixi run lint && pixi run typecheck && pixi run test` clean.
- `git grep "from lxd.stores.sqlite import"` shows zero — every caller uses the new submodule paths.
- Each new submodule under 300 LOC.

**Effort**: 4–6 h.

---

#### `B-CODE-1` Split `ingest/pipeline.py` (1143 LOC → 7-module subpackage)

**Why second**: Same treatment, smaller scope. Once `B-CODE-4` proves the pattern, `pipeline.py` follows the same recipe.

**Files**:

- `src/lxd/ingest/pipeline.py` → submodules under `src/lxd/ingest/pipeline/`.
- ~6 caller files (mostly `cli/`, `app/bootstrap.py`, tests).

**Natural splits**:

| Submodule | Concerns |
|---|---|
| `_orchestrator.py` | `run_ingest`, `IngestPlan`, `build_ingest_plan` |
| `_sources.py` | `_build_source_records`, `_load_extracted_document` |
| `_embed.py` | `_embed_with_cache`, `_embed_with_contextual_augmentation`, `_embed_with_context_refinement` |
| `_move.py` | move-detection branch, `_clone_source_records` |
| `_diagnostics.py` | wiki-relation diagnostics aggregation, ingest_runs telemetry |
| `_manifest_helpers.py` | `_manifest_record`, `_resolve_document_id`, file-state inspection |
| `_validation.py` | `_validate_ingest_dependencies`, embed-readiness probes |

**Verify**: same checks as B-CODE-4. `git grep "from lxd.ingest.pipeline import"` shows the new submodule paths.

**Effort**: 3–4 h.

---

#### `B-DOCS-1` Refactor `08_KNOWLEDGE_GRAPH_SPEC.md` sub-phase numbering

**Why third**: Doc-only debt. ~30 in-doc cross-references using `Phase 5.0`…`Phase 5.8` as build-wave identifiers, plus a dependency diagram. The earlier session stripped the title and external "(Phase 5)" / "Phases 0–4" framings (commit `e1d9529`) but left the internal numbering.

**Actions**:

1. Rename each subsection from `### Phase 5.X — Y` to `### Step N — Y` (descriptive name only, no wave-anchored identifier).
2. Walk the file and update every internal cross-reference (`see Phase 5.3`, `Dependencies: Phase 5.1`, etc.) to use the new step names.
3. Redraw the dependency diagram with descriptive node labels (Claims → Combined Graph → Communities → ...).
4. Run `markdownlint-cli2 --fix` on the file.

**Verify**: `git grep -E 'Phase 5\.[0-9]' Plans/08_KNOWLEDGE_GRAPH_SPEC.md` returns empty. The doc still parses; cross-references resolve.

**Effort**: 2 h.

---

### Tier 2 — Connection-pool / HTTP discipline

These reduce per-call overhead and remove duplicate client construction.

#### `B-PERF-3` Per-thread SQLite connection pool

**Why**: SQLite's WAL + tuned pragmas (`busy_timeout`, `temp_store`, `cache_size`, `synchronous`) are applied on every `connect_sqlite()` call. For one-shot CLI commands this is fine; for the long-lived MCP server, every tool call constructs a fresh connection and re-applies the pragmas. A per-thread pool amortises the cost.

**Files**:

- `src/lxd/stores/sqlite/_connection.py` — add a thread-local `Lock`-guarded `dict[Path, sqlite3.Connection]` accessed via a `get_pooled_connection(path)` helper.
- `src/lxd/mcp/async_runtime.py` — `run_tool` should pass through the pool rather than calling `connect_sqlite` per invocation.
- Reset behaviour: pool entries are torn down on `reset_store()`.

**Actions**:

1. New `_pool.py` (or in `_connection.py`) with a context manager that yields a pooled connection per `(thread_id, path)` key.
2. Migrate every `connect_sqlite()` call site that's inside the MCP request path.
3. Leave one-shot CLI uses on the unpooled API — they don't benefit.

**Verify**:

- New unit test: 100 sequential `with get_pooled_connection(p):` calls all return the same underlying connection per-thread.
- New test: connections are NOT shared across threads (different IDs, different objects).
- No regression in existing tests.

**Effort**: 3 h.

---

#### `B-ROBUST-1` Reuse pooled `httpx` for OpenAI embedding client

**Why**: `_openai_embed_texts` constructs `openai.OpenAI(api_key=...)` per batch call (`src/lxd/ingest/embedder.py:275`). The shared `lxd.net.http` factory exists exactly for this; bypassing it means each batch handshakes a fresh TLS connection.

**Files**:

- `src/lxd/ingest/embedder.py` — `_openai_embed_texts` should construct the client once outside the per-batch worker, and pass `http_client=` referencing the shared pool.
- `src/lxd/net/http.py` — confirm the pooled factory's signature is right for `openai.OpenAI(http_client=...)`.

**Actions**:

1. Pull the `openai.OpenAI(...)` construction out of `_embed_batch` into module-level lazy initialisation keyed on `(api_key_env, dims)`.
2. Pass `http_client=lxd.net.http.shared_client(...)` for HTTP/2 multiplexing across batches.

**Verify**:

- New test: `_openai_embed_texts` constructs the OpenAI client at most once across N batches in a single call (mocked `httpx`).
- Existing OpenAI embedding tests still pass.

**Effort**: 1.5 h.

---

### Tier 3 — Performance polish

#### `B-PERF-1` Tokenizer encodes full document twice — **OBSOLETE on survey 2026-05-05**

**Status**: Closed. The audit description does not match current code in `src/lxd/ingest/chunking.py`:

- Line 84 builds `full_text` (a `str.join` on text blocks) — **no encoding**.
- Line 90 encodes `full_text` exactly once via `tokenizer.encode(full_text)`.
- Line 196 encodes individual chunker outputs (`configured_tokenizer.encode(normalized_text)`) — those are different texts (chunker outputs, not the source document) and Docling's `HybridChunker` does not expose per-output token counts, so re-encoding is the only correct way to obtain them.

There is no double-encoding of the same text. No code change needed.

---

#### `B-PERF-2` `_unique_source_prefix` set-based dedup — **OBSOLETE on survey 2026-05-05**

**Status**: Closed. `src/lxd/retrieval/query_pipeline.py:403–415` already uses `seen_sources: set[str]` for membership tests and exits early when `len(unique) >= limit`. The helper is O(N) on its input list and short-circuits at the requested prefix size. The "re-run on every retry" framing is a non-issue — each retry iteration in `_dense_ranked_candidates` operates on a *different* `ranked` list (the result of a wider `search_vector_chunks` call), so the prefix computation cannot be cached across iterations. No code change needed.

---

#### `B-PERF-4` Vectorise embedding-cache lookup via Arrow — **PARTIALLY SHIPPED 2026-05-06 (audit framing inverted)**

**Status**: shipped a smaller fix; audit's framing struck.

**What was claimed**: replace per-chunk Python iteration with `pa.FixedSizeListArray.to_pylist()` for "microsecond" Arrow-vectorised conversion at 100k+ chunks.

**What the survey found**: a benchmark on this codebase (1k rows × 1536-dim vectors) showed the Arrow path at **220 ms** vs the existing path at **45 ms** — **5× slower**, not faster. LanceDB's `to_list()` already returns native `list[float]` per row, so the proposed `to_arrow()`-then-`to_pylist()` round-trip allocates the same Python objects via a longer code path. Also benchmarked at N=100 and N=10k; same direction.

**The real fix** (commit, with benchmark recorded inline): the previous `[float(v) for v in vector]` was paying for a per-element Python `float()` coercion that did not change the type — LanceDB's element type is already Python `float`. Replacing it with `list(vector)` keeps the defensive copy (cache returns mutable lists; mutation must not leak back) at **3× the throughput** (~45 ms → ~15 ms at 1k×1536). At 100k chunks that's ~5 s → ~1.5 s.

3 unit tests cover defensive-copy semantics (mutating a returned vector must not corrupt the cache), miss-index ordering with duplicates, and that returned elements remain `float` (not numpy scalars or other types) so downstream `float()` calls do not break.

(Original audit note retained below for reference.)


**Why**: `lookup_summaries` and `lookup` (embedding cache) iterate per-chunk in Python. Fine at 1k batches. At 100k+ chunks the Python overhead dominates.

**SCALE-DEFERRED**: only worthwhile when the corpus crosses ~25k chunks. Today's wiki has ~3k. Re-promote when the user's data dir grows past ~25k chunks.

**Files**: `src/lxd/ingest/embedding_cache.py`, `src/lxd/ingest/contextual_chunker.py`.

**Effort**: 2 h (when promoted).

---

### Tier 4 — Pydantic v2 modernisation

#### `B-CODE-2` `Pydantic TypeAdapter` for hot-path row→record validation — **STRUCK on survey 2026-05-05**

**Status**: Closed. The hand-written row→record adapters in
`src/lxd/stores/_sqlite_rows.py` are minimal `int/str/float` coercions on
a *trusted* internal SQLite schema. Pydantic `TypeAdapter` is for
validating untrusted input shapes; converting our frozen-dataclass
records to Pydantic `BaseModel` would add validation overhead without a
correctness benefit. The audit's own caveat ("benchmark, ship if at
parity") and the user's "no measurement ceremony" rule both point the
same way: skip.

(Original audit note retained below for reference.)


**Why**: `manifest_record_from_row`, `chunk_from_row`, `entity_profile_from_row` etc. construct dataclasses by hand. `TypeAdapter[ChunkRecord]` parses the same dict with field validation in C-implemented Pydantic core, faster on repeated parsing.

**Files**:

- `src/lxd/stores/_sqlite_rows.py` (or wherever it lives post-B-CODE-4) — replace each `*_from_row` with a `TypeAdapter` instance constructed once at module scope.
- May require Pydantic field aliases to map SQLite column names to dataclass field names.

**Actions**:

1. Convert frozen dataclasses to Pydantic `BaseModel` subclasses (or use `pydantic.dataclasses` for the existing dataclass shape).
2. Build `TypeAdapter[ChunkRecord]` etc. at module level (one-time cost amortised).
3. Replace each `*_from_row` with `_CHUNK_TYPE_ADAPTER.validate_python(dict(row))`.

**Caveat**: If the records are used in tight loops (millions of rows), the TypeAdapter overhead may equal or exceed the hand-written constructor. Benchmark the largest hot path (`load_chunk_records_for_source`) before/after.

**Verify**:

- Existing tests still pass (TypeAdapter validation must reject bad rows where the hand path silently coerced).
- New benchmark in `tests/unit/` shows TypeAdapter is at parity or faster on a 10k-row payload.

**Effort**: 3 h.

---

#### `B-CODE-3` Pydantic v2 `ComputedField`, `RootModel`, `BeforeValidator`/`AfterValidator`

**Why**: `settings/models.py` has one custom validator (`_normalize_query_instruction`). Other quirks (e.g. corpus_id slug shape, Tenancy ID validation, OpenAI dims/embed_dims symmetry) are checked in `model_validator(mode="after")`. Newer Pydantic v2 idioms move these into the field definition itself, tightening the schema and making invalid configs fail at field-validation time rather than after-validation time.

**Files**: `src/lxd/settings/models.py`.

**Candidate refactors**:

- `TenancyConfig.corpus_id`: `Annotated[str, AfterValidator(_validate_corpus_id_shape)]` instead of `model_validator`.
- `RuntimeConfig.embed_dims` ↔ `openai.dims`: `model_validator` is correct here (cross-field), but document the why.
- New `ComputedField` for derived values like `embedding_cache_key` if any.

**Verify**: existing config tests pass; new tests for invalid configs (bad corpus_id, missing OpenAI section when backend=openai) confirm the field-level errors.

**Effort**: 2 h.

---

#### `B-STACK-7` FastMCP structured tool input schemas — **CLOSED on survey 2026-05-05**

**Status**: Already done. Every tool in `src/lxd/mcp/tools.py` takes
typed primitives (`str`, `int`, `str | None`) — no tool accepts a
`dict[str, Any]` arg. FastMCP derives input JSON Schema from the typed
signatures automatically. Nothing to refactor.

(Original audit note retained below for reference.)


**Why**: Some MCP tools accept loose `dict[str, Any]` parameters. FastMCP can derive an input JSON Schema from a Pydantic model, which gives clients (Claude.ai, Cursor, etc.) auto-complete and validation.

**Files**: `src/lxd/mcp/tools.py` — every tool whose handler accepts a dict-shape arg.

**Actions**:

1. For each tool, define a `XxxToolArgs(BaseModel)` Pydantic model.
2. Update the handler signature to take the model directly.
3. FastMCP picks up the schema automatically.

**Verify**: golden test `tests/golden/mcp_tool_manifest.json` updates to include the new input schemas; client auto-complete improves (manual smoke).

**Effort**: 2.5 h.

---

### Tier 5 — KG signal lift

#### `B-KG-3` Use `entity_embeddings` for query expansion

**Why**: The `entity_embeddings` LanceDB table is built (per-entity vectors) but currently only consumed by `get_similar_entities` MCP tool. At retrieval time, we could embed the query, look up top-K nearest entities by cosine, expand the query with their canonical IDs / labels.

**Files**:

- `src/lxd/retrieval/expansion.py` — the existing ontology-based expansion is keyword-driven; add an embedding-based expansion lane.
- `src/lxd/stores/lancedb.py` — `search_entities_by_embedding(table, query_vector, limit)`.
- New config flag: `expansion.embedding_expansion_enabled: bool = False` (opt-in like the other LLM-using paths).

**Actions**:

1. After `expand_question` runs, if the new flag is on, embed the query and pull top-K entity vectors.
2. Merge the expanded entity IDs into `expansion.matched_entity_ids` (drives the relation lane in `_fuse_ranked_prefix`).
3. Cache: query embedding is already produced for dense retrieval; reuse it.

**Verify**:

- New tests: embedding expansion adds the right entity IDs given a fixture entity_embeddings table.
- Toggle off → no behaviour change; toggle on → adds related entity IDs.

**Effort**: 3 h.

---

#### `B-STACK-11` Advanced NetworkX graph queries — **STRUCK on survey 2026-05-05**

**Status**: Closed. The KG already exposes 6 centrality metrics
(PageRank, betweenness, closeness, in/out-degree, eigenvector) and a
`find_weighted_path` MCP tool that runs weighted Dijkstra. The audit's
remaining suggestions (HITS authority/hub, k-core decomposition, motif
detection) are graph-theoretic novelties without a concrete consumer in
the retrieval or synthesis paths. Adding metrics in search of a use
case violates the user's "nothing else / no measurement ceremony" rule.
Skip until a concrete consumer is identified.

(Original audit note retained below for reference.)


**Why**: With centrality already shipped (`[#2]`), the `entity_graph` is rich enough to support more sophisticated queries: HITS authority/hub scores, weighted-edge shortest paths, k-core decomposition for "core concepts", motif detection for triangle-completion suggestions.

**Files**:

- `src/lxd/ontology/entity_graph.py` — add HITS, weighted Dijkstra, k-core, motif-counting helpers.
- `src/lxd/mcp/tools.py` — new MCP tools: `get_authority_entities`, `get_hub_entities_hits`, `find_core_concepts`, `find_completing_triangles`.
- Schema: extend `entity_profiles` with `hits_authority` / `hits_hub` columns (migration v8).

**Verify**: new unit tests for each helper; new MCP-tool tests for each surface.

**Effort**: 4 h.

---

### Tier 6 — KG quality

#### `B-KG-4` Graph context token budget

**Why**: `build_graph_context` assembles entity profiles, community reports, and claims into the synthesis prompt. Without a token cap, a query that matches many entities can push the prompt past the model's context window, silently truncating evidence chunks.

**Files**:

- `src/lxd/retrieval/graph_routing.py` — `_build_graph_context_prompt` needs a `max_tokens` cap.
- `src/lxd/settings/models.py` — `KnowledgeGraphConfig.max_graph_context_tokens: int = 1500`.

**Truncation order** (when over budget): drop low-PageRank entities first, then low-modularity-class community reports, then low-confidence claims. Always preserve at least one entity profile per matched entity.

**Verify**:

- New unit test with synthetic high-volume graph context: result truncates deterministically and stays under the cap.
- Tokeniser used for counting: `tiktoken` (already in deps); for non-OpenAI synthesis use char/4 estimate.

**Effort**: 2.5 h.

---

#### `B-KG-1` Claim verification / contradiction detection — **DEFERRED on survey 2026-05-05**

**Status**: Deferred pending design clarification.

The audit's deterministic approach groups claims by
`(subject_entity_id, predicate)` but the actual `claims` schema has no
`predicate` column — only `subject_entity_id`, `object_entity_id`,
`claim_type`, `claim_text`, `confidence`. The audit's worked example
`(addie_model, has_phase_count, "five")` vs `"six"` describes
**relation-shaped triples with literal objects**, which fit neither
`claims` (no predicate) nor `relations` (`object_entity_id` is an
entity, not a literal value).

Three honest options for a future implementation:

1. **Relations-based contradiction**: pairs in canonical `relations`
   sharing `(subject, predicate)` with different `object_entity_id`.
   Real signal but limited to entity-entity relations, not literal
   numeric/textual disagreements like the worked example.
2. **Claims-redundancy detector**: pairs with identical
   `(subject, object, claim_type)` and different `claim_text`. Surfaces
   "possibly redundant or conflicting" pairs but is not actual
   contradiction detection.
3. **LLM-adjudicated pair check**: feed candidate claim pairs to an LLM
   and ask "do these contradict?". Robust but adds API spend per pair;
   needs a concrete consumer (CLI review workflow, MCP tool, or both)
   to justify the cost.

Decision needed from the user before this becomes work: which option
(or which combination) actually answers the LxD use case. Until that
decision is made, building any one of them risks shipping a feature
that does not match the actual need.

(Original audit note retained below for reference.)


**Why**: `claims` table has structured assertions (subject, predicate, object, claim_text, confidence). Two claims with the same subject + predicate but contradictory objects (e.g. `(addie_model, has_phase_count, "five")` vs `(addie_model, has_phase_count, "six")`) sit silently. A contradiction-detection pass flags them for human review.

**Files**:

- New `src/lxd/ontology/claim_verification.py` — `find_contradicting_claim_pairs(connection) -> list[ContradictionPair]`.
- New SQLite table `claim_contradictions(pair_id, claim_a_id, claim_b_id, contradiction_type, surfaced_at)` (migration v9).
- New CLI: `pixi run review-contradictions`.
- New MCP tool: `list_contradictions`.

**Approach**: pairwise comparison over the same `(subject_entity_id, predicate)` tuple is straightforward; LLM-based "is X a contradiction of Y" check is a stretch goal — start with the deterministic pairing.

**Verify**: new unit tests with seeded contradicting/non-contradicting pairs.

**Effort**: 4 h.

---

#### `B-KG-2` Entity disambiguation (fuzzy + embedding-aware)

**Why**: Aho-Corasick is exact-match-only. Acronyms with multiple expansions (e.g. "ID" = instructional design / identifier) resolve first-rule-wins. A contextual disambiguator using surrounding chunk text would resolve ambiguity correctly.

**Files**:

- `src/lxd/ingest/mentions.py` — extend `detect_mentions` to flag ambiguous matches.
- New `src/lxd/ontology/disambiguator.py` — given a match + surrounding window, embed the window, compare against entity_embeddings, pick the highest-cosine canonical_id.

**Actions**:

1. Build a list of "ambiguous surface forms" from the ontology — surface forms that map to >1 canonical_id.
2. For each ambiguous mention, run the embedding-based disambiguator with a ±200-char context window.
3. Tie-break by cosine + length-of-canonical-form.

**Verify**: new tests with seeded ambiguous mentions disambiguating correctly.

**Effort**: 4 h.

---

#### `B-KG-5` Chunk-incremental graph build

**Why**: `pixi run build-graph` re-runs claim extraction for every entity touched by any new document. For a single-page incremental ingest this is wasteful. Should be additive: only extract claims for newly-added or changed chunks; merge into existing entity profiles.

**Files**:

- `src/lxd/cli/graph.py` — orchestrator branch: detect "since last build" via `graph_metadata.last_build_at` and `chunk_rows.last_committed_at`.
- `src/lxd/ingest/claims.py` — add `extract_claims_for_new_chunks_since(connection, since)`.
- `src/lxd/ontology/profiles.py` — incremental profile rebuild for affected entities.

**Caveat**: this is non-trivial. Centrality can change globally when a single chunk adds an edge; profiles are deterministic-from-current-graph, so a partial rebuild risks drift. Make `--full` still the safe default; `--incremental` is the new opt-in.

**Verify**: new integration test with two-page corpus, incremental rebuild only re-extracts the changed page's claims.

**Effort**: 5 h.

---

### Tier 7 — MCP capability surface

#### `B-STACK-4` FastMCP Resources (`lxd://corpus/{path}`) — **SHIPPED 2026-05-06**

**Status**: shipped. `lxd://corpus/{path*}` resource registered in `mcp/server.py` (`{path*}` so nested paths like `Guides/alpha.md` resolve, since FastMCP's default `{path}` matcher does not capture `/`). Allowed text suffixes: `.md`, `.markdown`, `.mdx`, `.txt`, `.json`. Path-traversal protection refuses absolute paths, `..` segments, and symlinks pointing outside the corpus root (verified by `Path.resolve().relative_to(corpus_root.resolve())`). 7 unit tests cover happy path, dotdot, absolute path, missing file, non-text suffix, symlink escape, and empty path.

(Original audit note retained below for reference.)


**Why**: Citations from `search_knowledge` reference chunks by `citation_label`, which is `<source_rel_path>#<chunk_index>`. Clients can't currently fetch the raw source file. A FastMCP Resource for `lxd://corpus/{rel_path}` would let clients (Claude.ai, Cursor) pull the original markdown for display.

**Files**: `src/lxd/mcp/server.py` — register a Resource. New handler reads from `config.paths.corpus_path / rel_path`.

**Actions**:

1. Register `lxd://corpus/{path}` URI scheme.
2. Resolve `path` against `corpus_path`; refuse path traversal.
3. Return text/markdown for `.md`, refuse other types or return file metadata only.

**Verify**: new MCP-tool test fetches a corpus file via Resource URI; path-traversal attempt returns the right error.

**Effort**: 2 h.

---

#### `B-STACK-5` FastMCP Prompts (parameterised templates) — **SHIPPED 2026-05-06**

**Status**: shipped. The synthesis preamble is now a single source of truth at `src/lxd/synthesis/answering.py:synthesis_preamble(...)` with three exported constants (`SYNTHESIS_PREAMBLE_BASE`, `SYNTHESIS_PREAMBLE_TRANSITIVE_SOURCES`, `SYNTHESIS_PREAMBLE_GRAPH_CONTEXT`); the runtime `_build_prompt` and the MCP `lxd_synthesis_preamble` prompt both call the same function. The MCP prompt returns the full text (both sub-sections enabled) so clients see every instruction the system might emit. 6 tests cover the four sub-section permutations plus FastMCP-side prompt registration and render-content equivalence.

(Original audit note retained below for reference.)


**Why**: Currently the synthesis prompt and graph-context prompt are baked into our code. Exposing them as FastMCP Prompts (`lxd_search_prompt`, `lxd_synthesis_prompt`) lets clients render-and-edit them — useful for transparency and for users who want to tweak phrasing without forking the code.

**Files**: `src/lxd/mcp/prompts.py` (new) — register prompt templates. `src/lxd/synthesis/answering.py` — split prompt-building from prompt-string-template so both code-path and Prompt-resource use the same template.

**Effort**: 2 h.

---

#### `B-STACK-6` FastMCP Sampling (server-initiated LLM via client)

**Why**: Sampling lets the MCP server delegate LLM calls back to the client model rather than running its own Ollama. Useful for clients like Claude.ai who already have a strong LLM. **But**: the local-only rule applies here. Sampling routes the call to *whatever the client uses*, which may be remote. Per local-only, this is opt-in via config and clearly labelled.

**Decision**: Add the capability (FastMCP supports it natively), gate it behind `mcp.client_sampling_enabled: bool = False`. Document explicitly that turning it on means the synthesis path no longer runs locally for that client.

**Files**: `src/lxd/synthesis/answering.py` — when sampling is on for the active client connection, dispatch via FastMCP sampling; else local Ollama as today.

**RAISE-FIRST**: this fundamentally changes the local-only guarantee for connected clients. Discuss with user before implementing.

**Effort**: 4 h (when approved).

---

### Tier 8 — Observability

#### `B-STACK-8` `bind_contextvars` per MCP-tool-call — **SHIPPED 2026-05-06**

**Status**: shipped. `mcp/async_runtime.run_tool` now calls `structlog.contextvars.bind_contextvars(tool=name)` at request entry and `reset_contextvars(**bound_keys)` in `finally`, so every log line emitted while the tool body runs is automatically tagged with the originating tool name. Three integration tests verify: tag is present on logs inside the tool, contextvar is cleared after a successful run, and the contextvar is also cleared when the tool body raises.

(Original audit note retained below for reference.)


**Why**: structlog is configured at startup with global contextvars. Per-request fields (tool name, query length, matched entity IDs) are added inline in each log call. Binding them once at request entry makes every log line in that request carry the context automatically.

**Files**:

- `src/lxd/mcp/async_runtime.py` — `run_tool` should `structlog.contextvars.bind_contextvars(tool=...)` at entry; reset on exit.
- `src/lxd/observability/logging.py` — confirm the merger is wired.

**Verify**: new integration test asserts a logged event during a tool call carries `tool=...` automatically.

**Effort**: 1 h.

---

#### `B-STACK-9` Sampled logging for high-volume events — **SHIPPED 2026-05-06**

**Status**: shipped. `make_sampled_processor(rate, high_volume_events)` lives at `src/lxd/observability/logging.py`; counter-based 1-in-N sampling per event name (lock-guarded so threaded callers see exact sampling, not probabilistic). `error` and `critical` level events bypass sampling unconditionally, as do any event names outside the allow-list. Wired through `LoggingConfig.sample_rate` (default 1 — disabled) and `LoggingConfig.sampled_event_names` (default high-volume ingest events: `embedding_cache_hit`, `embedding_cache_miss`, `chunk_processed`, `mention_detected`). Bootstrap passes the new fields into `configure_logging`. 7 unit tests cover rate=1 disable, non-sampled bypass, sampling math, error/critical bypass, per-event counters, and `DropEvent` semantics.

(Original audit note retained below for reference.)


**Why**: Ingest emits per-chunk progress logs (`chunk_processed`, `embedding_cache_hit`, etc.) — at 25k chunks that's 25k log lines per run. Most are redundant. Sample 1 in 100 (configurable); always emit errors and final summaries.

**Files**:

- `src/lxd/observability/logging.py` — new `sampled_processor(rate)` structlog processor.
- Wire to high-volume event names via a config-driven allow-list.

**Effort**: 2 h.

---

### Tier 9 — Pre-flight + ingest polish

#### `B-STACK-10` `tiktoken` pre-flight cost estimation — **SHIPPED 2026-05-06**

**Status**: shipped. `estimate_run_cost(scanned_files, config) -> CostEstimate` lives at `src/lxd/ingest/budget.py`. Embedding tokens are estimated at `ceil(corpus_text_bytes / 4)` (a safe upper bound on natural-prose corpora). LLM cost is bounded by `ingest_budget.max_llm_calls_per_run × (prompt + completion) tokens`, so the user sees a worst-case ceiling rather than an unbounded "infinity"; when no cap is set, the LLM total is reported as 0 and the CLI surfaces a "no cap configured" note. OpenAI list prices are encoded in a hard-coded table commented with the re-check date (2026-05-06): text-embedding-3-small ($0.020/M), text-embedding-3-large ($0.130/M), gpt-4o-mini input/output ($0.150/$0.600 per M). Wired into `pixi run preflight` as a new "Estimated run cost" panel after the existing health checks. 8 unit tests cover token math, image-file exclusion, USD calculation, no-cap fallback, total summation, and unknown-model graceful default.

(Original audit note retained below for reference.)


**Why**: Pairs naturally with `[#11]` ingest budget cap. Today the budget is on **count of LLM calls**; with `tiktoken` we can estimate **token cost** before running, so `pixi run preflight` shows "this ingest will require ~12.5M embedding tokens (~$0.20 at text-embedding-3-small) and ~8M LLM tokens (~$1.60 at gpt-4o-mini)". User can refuse before paying.

**Files**:

- `src/lxd/cli/preflight.py` — new "estimated cost" panel.
- `src/lxd/ingest/budget.py` — add `estimate_run_cost(plan, config) -> CostEstimate`.

**Pricing table**: hard-coded per-model rates as a Python dict; document the source and re-check date in a comment.

**Verify**: new test seeds a small fixture corpus and asserts the estimate matches a hand-calc within 5%.

**Effort**: 2.5 h.

---

#### `B-ROBUST-2` Aho-Corasick matcher disk-pickle — **SHIPPED 2026-05-06**

**Status**: shipped. `build_or_load_automaton(records, *, cache_dir)` lives at `src/lxd/ontology/matcher.py:107`; cache key is `matcher-<matcher_termset_hash>.pkl` under `cache_dir`. Both call sites (`retrieval/expansion.py`, `ingest/pipeline/orchestrator.py`) now use `<data_path>/matcher_cache/`. 4 unit tests cover cold-build, warm-load equivalence, hash-mismatch isolation, and corrupt-cache fall-through. 274 tests passing.

(Original audit note retained below for reference.)


**Why**: Building the matcher from the ontology takes ~1-2s per CLI invocation. For short-running commands (`pixi run status`, MCP tool calls) this dominates the wall-clock. Pickle the matcher to `data/openai/cache/matcher-{ontology_hash}.pkl`; load on startup if the hash matches.

**Files**:

- `src/lxd/ontology/matcher.py` — `build_or_load_automaton(records, *, cache_dir, ontology_hash)`.
- `src/lxd/retrieval/expansion.py:20` — call site uses the cached version.

**Verify**: cold-start build, warm-start load (different code path), assert returned automaton matches identically; cache invalidation on hash mismatch.

**Effort**: 1.5 h.

---

#### `B-ROBUST-3` Empty-frontmatter wiki page early-exit — **OBSOLETE on survey 2026-05-06**

**Status**: closed. The LLM short-circuit the audit asks for *already exists* at the chunk level: `extract_relations_for_chunk` (`src/lxd/ingest/relations.py:79`) returns `[]` immediately when `len(entity_ids) < cfg.min_entity_mentions` (line 95-96) or when `valid_predicates` is empty (line 98-99). `pipeline/sources.py:131-133` mirrors the same gate before recording an LLM call against the budget tracker. An empty-frontmatter wiki page where every chunk has zero entity matches already pays zero LLM calls. The only marginal saving the audit's page-level check would deliver is skipping the cheap Aho-Corasick `detect_mentions` work — sub-millisecond per chunk. Not worth a code change.

(Original audit note retained below for reference.)


**Why**: A wiki page with no `Sources:` line and no `[[slug]]` cross-references still goes through the full mention-detection + relation-extraction pipeline. If the page has no entity matches AND no wiki frontmatter, it can't contribute any KG signal — should skip the LLM relation extraction lane.

**Files**:

- `src/lxd/ingest/pipeline.py` — in `_build_source_records`, if `wiki_metadata.is_empty` AND zero mentions detected, skip the relation-extraction call.

**Verify**: new test with a no-frontmatter no-entity-match page asserts zero LLM calls during ingest.

**Effort**: 1 h.

---

### Tier 10 — Test infrastructure

#### `B-TEST-1` Synthesis end-to-end test — **SHIPPED 2026-05-06**

**Status**: shipped. `tests/integration/test_synthesis_e2e.py` calls real `synthesize_answer` against the local Ollama server with a seeded `EvidenceChunk` and asserts `answer_status == ANSWERED` plus a substring match against expected entity terms. Marked `live` + `integration`; pre-flight reachability probe (`socket.create_connection` with 1 s timeout) returns `pytest.skip` rather than fail when Ollama is not running. Pyproject registers the new `live` marker; `pixi run test` filters it out (`-m 'not live'`); new `pixi run test-live` invokes only `live` tests with verbose output. Default suite: 305 passed, 1 deselected (the new live test) — exactly as designed.

(Original audit note retained below for reference.)


**Why**: LLM is mocked everywhere in the test suite. No test verifies the synthesis prompt produces sensible output against a real local Ollama. Pair with an opt-in "live" pytest marker; runs against a local Ollama instance when `pixi run pytest -m live` is invoked.

**Files**:

- `tests/integration/test_synthesis_e2e.py` — fixture seeds a small corpus, runs `answer_question`, asserts the answer text contains the expected entity name.
- `pyproject.toml` — register `live` marker.

**Caveat**: by default `pixi run test` skips `live` tests; user opts in.

**Verify**: `pixi run pytest -m live` passes against a running Ollama; default `pixi run test` skips silently.

**Effort**: 2 h.

---

### Tier 11 — Polars / Arrow modernisation (low priority)

#### `B-STACK-12` Polars-on-Arrow for KG analyses — **STRUCK on survey 2026-05-06**

**Status**: closed. The audit's framing assumes the source data is already in Arrow ("LanceDB returns Arrow natively"). Survey of the actual aggregation hotspots:

- `load_entity_mention_stats` — `SELECT ... FROM mention_rows JOIN chunk_rows GROUP BY entity_id`
- `load_chunk_ids_for_entity` — `SELECT chunk_id FROM mention_rows WHERE entity_id = ? GROUP BY chunk_id`
- aggregation in `kg_relations.py:94` — `GROUP BY predicate` over `extracted_relations`
- aggregation in `profiles.py:199` — `GROUP BY entity_id, chunk_id` over `mention_rows`

Every one of these reads SQLite tables (`mention_rows`, `chunk_rows`, `extracted_relations`) which have no Arrow counterpart — only chunk *vectors* live in LanceDB, not the mention/relation rows. To use Polars-on-Arrow we would have to:

1. `SELECT * FROM mention_rows` (full table scan into Python),
2. convert to Arrow,
3. run polars `group_by`/`agg`,

which is strictly slower than letting SQLite do the GROUP BY natively over its own b-tree indexes. There is no scale at which this becomes a win until the underlying mention/relation data is migrated into Arrow storage — and that migration would be a much larger change than this audit item, gated on its own consumer requirement.

This is a strike based on the data shape, not a scale defer: the rewrite is a regression at every scale of the current data model.

(Original audit note retained below for reference.)


**Why**: LanceDB returns Arrow natively; some KG analyses currently round-trip through SQLite (`load_entity_mention_stats` etc.) for grouping/aggregation that Polars-on-Arrow does in microseconds.

**Files**: targeted analyses in `src/lxd/ontology/profiles.py` and `src/lxd/cli/graph.py`.

**Approach**:

1. Identify which analyses are aggregation-heavy (>1k rows × multiple group-by passes).
2. Replace per-analysis with `polars.from_arrow(table.to_arrow())` and `.group_by(...).agg(...)`.
3. Keep SQLite-via-sqlite3 for transactional / row-level access.

**SCALE-DEFERRED**: only worthwhile when a specific analysis is observed slow. Profile first; rewrite second.

**Effort**: 3 h (when promoted).

---

## RAISE-FIRST items (discuss with user before any code change)

#### `B-LOCAL-1` `rerank.py` auto-spawns `llama-server` from query path

**Decision space** (raise, agree, then act):

- **(a)** Move launch responsibility to `start.sh` / new `pixi run reranker` task. `rerank.py` becomes a pure HTTP client to a known URL. If unreachable → graceful degrade to "no rerank" (current behaviour). Cleanest separation.
- **(b)** Keep auto-spawn but factor it out of the query path into a singleton/lifecycle hook (start once at MCP server boot, not on every search). Less change to user workflow.
- **(c)** Leave as-is. Document the auto-spawn as a known-fragility, accept it.

**No remote replacement.** Remote rerank is permanently rejected per `feedback_local_only_no_remote_rerank.md`.

**Effort**: 2–4 h depending on choice.

---

#### `B-LOCAL-2` `synthesis/answering.py` is hard-bound to one local model

**Decision space** (raise, agree, then act):

- **(a)** Leave as-is. Switching model = edit `config.models.llm`, restart MCP. Explicit and honest.
- **(b)** Introduce local-only backend dispatch (`LocalSynthesisBackend = Literal["ollama", "llama_cpp", "mlx"]`) so the user can swap engines via config without touching synthesis code. Discriminated union per backend.
- **(c)** Parameterise model choice per-query via an MCP tool argument rather than globally. More flexible; more surface to maintain.

**No remote backends.**

**Effort**: 1 h (a) / 4 h (b) / 6 h (c).

---

## SCALE-DEFERRED (do not promote until the gate clears)

| ID | Gate | Effort when promoted |
|---|---|---|
| `B-PERF-4` Vectorise embedding-cache lookup via Arrow | Corpus crosses ~25k chunks (today: ~3k) | 2 h |
| `B-STACK-1` LanceDB scalar quantisation + IVF_PQ | Corpus crosses ~1M chunks (today: ~3k) | 4 h |
| `B-STACK-2` LanceDB version branching / time-travel queries | Only useful with an A/B framework. Per the no-measurement-ceremony rule, this lands when a concrete user-facing A/B need exists | 3 h |
| `B-STACK-12` Polars-on-Arrow KG analyses | Profile-driven: a specific analysis observed >1s | 3 h |
| `B-TEST-2` Mutation testing | Low-value at current scale; revisit if regressions become a pattern | 4 h |

---

## Multi-Session Schedule

| # | Session | Items | Effort |
|---|---|---|---|
| 1 | **Refactor surface — sqlite split** | `B-CODE-4` | 4–6 h |
| 2 | **Refactor surface — pipeline split + 08 spec rename** | `B-CODE-1`, `B-DOCS-1` | 5–6 h |
| 3 | **Connection-pool / HTTP discipline** | `B-PERF-3`, `B-ROBUST-1` | 4–5 h |
| 4 | **Performance polish** | `B-PERF-1`, `B-PERF-2` | 1.5 h |
| 5 | **Pydantic v2 modernisation** | `B-CODE-2`, `B-CODE-3`, `B-STACK-7` | 7–8 h |
| 6 | **KG signal lift** | `B-KG-3`, `B-STACK-11` | 7 h |
| 7 | **KG quality — token budget + contradictions** | `B-KG-4`, `B-KG-1` | 6.5 h |
| 8 | **KG quality — disambiguation + incremental** | `B-KG-2`, `B-KG-5` | 9 h |
| 9 | **MCP capability surface** | `B-STACK-4`, `B-STACK-5` (`B-STACK-6` is RAISE-FIRST, separate) | 4 h |
| 10 | **Observability + ingest polish** | `B-STACK-8`, `B-STACK-9`, `B-STACK-10`, `B-ROBUST-2`, `B-ROBUST-3` | 8 h |
| 11 | **Test infrastructure** | `B-TEST-1` | 2 h |

**Total scheduled: ~57 h, 11 sessions.** Realistic for ~3 weeks of focused work, ~6 weeks part-time.

**Out-of-band items** (no schedule):

- `B-LOCAL-1`, `B-LOCAL-2`, `B-STACK-6` — RAISE-FIRST. Discuss with user → agree on direction → implement in a one-off session.
- `B-PERF-4`, `B-STACK-1`, `B-STACK-2`, `B-STACK-12`, `B-TEST-2` — SCALE-DEFERRED. Promote only when the listed gate clears.

---

## Verification (overall)

Each session ends with the same gate:

```bash
pixi run lint && pixi run typecheck && pixi run test
```

Refactor sessions (1, 2) additionally:

1. After each module split, `git grep "from <old_module> import"` shows the new submodule paths and zero references to the old path.
2. Each new submodule under 300 LOC.

**No `pixi run eval` gate.** Measurement is the user's call, run when they choose.

---

## To-do list — strictly ordered for execution

```text
[ ] S1.1  B-CODE-4  Read full sqlite.py (2085 LOC) into one Read call
[ ] S1.2  B-CODE-4  git mv sqlite.py to /tmp; mkdir sqlite/ package; write 9 submodules
[ ] S1.3  B-CODE-4  Update ~28 caller imports atomically (src/ + tests/)
[ ] S1.4  B-CODE-4  pixi run lint && typecheck && test green
[ ] S1.5  B-CODE-4  Commit: "B-CODE-4: split stores/sqlite.py into 9-module subpackage"

[ ] S2.1  B-CODE-1  Read full ingest/pipeline.py
[ ] S2.2  B-CODE-1  Move into 7-module subpackage; update ~6 callers
[ ] S2.3  B-DOCS-1  Rename Plans/08 sub-phase headings to descriptive Step-N names
[ ] S2.4  B-DOCS-1  Update ~30 internal cross-references + dependency diagram
[ ] S2.5  Commit: "B-CODE-1 + B-DOCS-1: split ingest/pipeline.py; refactor 08 KG spec"

[ ] S3.1  B-PERF-3   Add per-thread SQLite connection pool in stores/sqlite/_connection.py
[ ] S3.2  B-PERF-3   Migrate MCP request-path call sites to the pool
[ ] S3.3  B-ROBUST-1 Hoist openai.OpenAI() construction out of per-batch worker; pass http_client=
[ ] S3.4  Commit: "B-PERF-3 + B-ROBUST-1: pooled SQLite + pooled OpenAI HTTP client"

[ ] S4.1  B-PERF-1   Cache encoded token list in chunk_document
[ ] S4.2  B-PERF-2   Re-survey _unique_source_prefix; ship if real, skip if stale
[ ] S4.3  Commit: "B-PERF-1 (+B-PERF-2 if applicable): per-doc tokeniser cache"

[ ] S5.1  B-CODE-2   Convert row-adapters to TypeAdapter; benchmark; ship if at parity
[ ] S5.2  B-CODE-3   Field-level validators for TenancyConfig + others
[ ] S5.3  B-STACK-7  Pydantic input-arg models for MCP tools
[ ] S5.4  Commit: "B-CODE-2 + B-CODE-3 + B-STACK-7: Pydantic v2 modernisation"

[ ] S6.1  B-KG-3     embedding-based query expansion (opt-in flag)
[ ] S6.2  B-STACK-11 HITS, weighted Dijkstra, k-core, motif helpers + new MCP tools
[ ] S6.3  Commit: "B-KG-3 + B-STACK-11: KG signal lift"

[ ] S7.1  B-KG-4     Graph context token budget + tier-based truncation
[ ] S7.2  B-KG-1     Claim contradiction detection + new MCP tool
[ ] S7.3  Commit: "B-KG-4 + B-KG-1: graph context budget + claim contradictions"

[ ] S8.1  B-KG-2     Embedding-aware disambiguator for ambiguous mentions
[ ] S8.2  B-KG-5     Chunk-incremental graph build (--incremental flag)
[ ] S8.3  Commit: "B-KG-2 + B-KG-5: disambiguation + incremental graph build"

[ ] S9.1  B-STACK-4  Register lxd://corpus/{path} FastMCP Resource
[ ] S9.2  B-STACK-5  Register lxd_search_prompt + lxd_synthesis_prompt FastMCP Prompts
[ ] S9.3  Commit: "B-STACK-4 + B-STACK-5: MCP Resources + Prompts"

[ ] S10.1 B-STACK-8  bind_contextvars per MCP-tool-call in run_tool
[ ] S10.2 B-STACK-9  Sampled logging processor for high-volume ingest events
[ ] S10.3 B-STACK-10 tiktoken pre-flight cost estimation; pairs with #11 budget
[ ] S10.4 B-ROBUST-2 Aho-Corasick disk-pickle keyed on ontology hash
[ ] S10.5 B-ROBUST-3 Empty-frontmatter early-exit in _build_source_records
[ ] S10.6 Commit: "Observability + ingest polish (B-STACK-8/9/10 + B-ROBUST-2/3)"

[ ] S11.1 B-TEST-1   Synthesis e2e test under @pytest.mark.live
[ ] S11.2 Commit: "B-TEST-1: synthesis e2e test (opt-in via -m live)"

# RAISE-FIRST (out-of-band, no scheduled session)
[?] B-LOCAL-1   rerank.py auto-spawn — pick (a)/(b)/(c) with user
[?] B-LOCAL-2   synthesis model dispatch — pick (a)/(b)/(c) with user
[?] B-STACK-6   FastMCP Sampling — opt-in to client LLM; breaks local-only

# SCALE-DEFERRED (do not promote yet)
[/] B-PERF-4    promote when corpus crosses ~25k chunks
[/] B-STACK-1   promote when corpus crosses ~1M chunks
[/] B-STACK-2   promote when concrete A/B need exists
[/] B-STACK-12  promote when a specific analysis observed >1s
[/] B-TEST-2    promote if regressions become a pattern
```

---

*Plan created: 2026-05-05. Source: Tier 7 backlog from `sota-fix-implementation-2026-05.md`. Total backlog: 33 items; 22 scheduled across 11 sessions; 3 RAISE-FIRST; 5 SCALE-DEFERRED. Same principles as the parent plan: local-only, no half-implementations, no measurement chores, no legacy/back-compat shims.*
