---
name: 2026-05 backlog implementation run
description: Sessions 1–7 of the Tier-7 backlog plan delivered 8 ships, 5 strikes-on-survey, 1 deferral; this captures the *why* behind each strike/defer for future re-litigation.
type: project
---

Tier-7 backlog implementation, 2026-05-05, captured in `.claude/plans/backlog-implementation-2026-05.md`. Seven coding sessions ran end-to-end on `main`; commits visible via `git log 880c596..d01b70f`.

**Ships (8):**

- `B-CODE-4` — `stores/sqlite.py` → `stores/sqlite/` subpackage (9 modules). Commit `880c596`.
- `B-DOCS-1` — `Plans/08_KNOWLEDGE_GRAPH_SPEC.md` Phase 5.X → Step N rename. Commit `396f9bf`.
- `B-CODE-1` — `ingest/pipeline.py` → `ingest/pipeline/` subpackage (4 modules). Commit `9119fb0`.
- `B-PERF-3` — per-thread SQLite connection pool at `stores/sqlite/_pool.py`. Commit `868354f`.
- `B-ROBUST-1` — process-wide cached `openai.OpenAI` instance via `ingest.embedder.get_openai_client`. Commit `868354f`.
- `B-CODE-3` — `BeforeValidator` / `AfterValidator` field annotations in `settings/models.py` for `query_instruction` and `corpus_id`. Commit `9c151eb`.
- `B-KG-3` — embedding-based entity expansion lane via `_augment_with_embedding_neighbours` in `retrieval/query_pipeline.py`; always-on, no toggle. Commit `4eb7d2e`.
- `B-KG-4` — graph context token budget (`knowledge_graph.max_graph_context_tokens`, default 1500). Commit `d01b70f`.

**Strikes (5) — all on survey against current code:**

- `B-CODE-2` (TypeAdapter for row→record adapters): the row adapters are minimal `int/str/float` coercions on a *trusted* internal SQLite schema. Pydantic `TypeAdapter` is for validating *untrusted* input shapes; using it here adds overhead with no correctness benefit. The audit's own caveat ("benchmark, ship if at parity") points the same way.
- `B-PERF-1` (tokenizer encodes full document twice): the line numbers in the audit (84 and 188 of `chunking.py`) do not describe a double-encode in current code. Line 84 builds `full_text` via `str.join` (no encoding); line 90 encodes the document *once*; line 196 encodes chunker outputs (different texts, and Docling's `HybridChunker` does not expose per-output token counts so re-encoding is the only correct path).
- `B-PERF-2` (`_unique_source_prefix` set-based dedup): the helper at `retrieval/query_pipeline.py:403` already uses `seen_sources: set[str]` and exits early. The "re-run on every retry" framing is invalid: each retry iteration in `_dense_ranked_candidates` operates on a *different* `ranked` list, so caching across iterations is impossible.
- `B-STACK-7` (Pydantic input schemas for MCP tools): every tool in `mcp/tools.py` already takes typed primitives (`str`, `int`, `str | None`); FastMCP derives the input JSON schema from the typed signatures automatically. No tool accepts `dict[str, Any]`.
- `B-STACK-11` (advanced NetworkX queries — HITS, k-core, motifs): `find_weighted_path` already implements weighted Dijkstra; six centrality metrics already populate `entity_profiles`. The remaining additions (HITS, k-core, motif detection) are graph-theoretic novelties without a concrete consumer in the retrieval or synthesis path.

**Defer (1):**

- `B-KG-1` (claim contradiction detection): the audit's deterministic plan groups claims by `(subject_entity_id, predicate)` but the actual `claims` schema has no `predicate` column. The worked example `(addie_model, has_phase_count, "five")` vs `"six"` describes relation-shaped triples with literal objects, which fit neither `claims` (no predicate) nor `relations` (`object_entity_id` is an entity, not a literal). Three honest options remain: relations-based, claims-redundancy, or LLM-adjudicated. User direction is needed before implementation.

**Why future-me might want this memory:**

- Before re-litigating a strike, compare the current code at the cited line against the audit framing. If the audit framing still mismatches, the strike stands; if the codebase has drifted in the meantime, re-survey.
- Before resuming `B-KG-1`, ask the user which of the three options they want — implementing any of them without that decision risks a half-correct feature.
- The "no enabled toggles for core KG features" rule (`feedback_mandatory_features.md`) was followed for `B-KG-3` — the embedding entity expansion is always-on and silently no-ops when the entity table doesn't exist. Future KG work should follow the same pattern.

**Sessions remaining (8–11) — not yet executed:**

- S8: `B-KG-2` (entity disambiguation), `B-KG-5` (chunk-incremental graph build).
- S9: `B-STACK-4`, `B-STACK-5` (MCP capability surface).
- S10: `B-STACK-8/9/10`, `B-ROBUST-2/3` (observability + ingest polish).
- S11: `B-TEST-1` (synthesis e2e test).

Re-survey each item against current code before implementing — several of S8–S11 may have audit framing as stale as the items above. RAISE-FIRST gates (`B-LOCAL-1`, `B-LOCAL-2`, `B-STACK-6`) and SCALE-DEFERRED items remain gated on user direction or load growth.
