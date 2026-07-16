# LxD Machine — Query Specification

---

## 1. Query Goal

Answer questions over the real corpus with citations.

The query stack must work even if some optional enrichment is missing.

---

## 2. Minimal Query Pipeline

### Stage 1 — Validate

- ensure input is non-empty
- validate domain if provided
- validate limits

Domain validation rule:

- allowed domains come from committed non-deleted `source_domain` values in `corpus_manifest`
- do not hard-code the allowed domain list in application logic
- if no committed corpus rows exist yet, domain validation may accept `None` only

### Stage 2 — Expand (mandatory)

Runs on every retrieval:

- detect ontology entities in the question via the Aho-Corasick matcher
- widen the matched-entity set with the entities nearest to the query vector in the `entity_embeddings` LanceDB table (adds semantic neighbours, not just surface-form hits)
- expand the widened set over the in-memory ontology graph with `config.expansion.hops` and `config.expansion.max_terms`

Implementation rule:

- expansion is a mandatory feature (no `enabled` toggle) per `.claude/rules/mandatory-features.md`; it degrades gracefully — when the graph isn't built or the question matches no entities, the widening is a no-op
- expansion terms and the matched entity IDs must be surfaced in query metadata so retrieval changes are auditable (`KnowledgeAnswerMetadata.matched_entity_ids` and `.expansion_terms`)

### Stage 3 — Retrieve (hybrid dense + BM25 via LanceDB native fuse)

- dense k-NN + BM25 FTS are always issued and fused inside LanceDB via `Table.search(query_type="hybrid")` fed to `RRFReranker`
- `search_chunks_hybrid` (in `stores/lancedb.py`) returns one ordered stream keyed on `_relevance_score`; per-lane ranks are collapsed inside the engine
- the previous split path (`search_chunks` dense-only + `search_chunks_fts` BM25 + Python-side lexical-lane fuser) is superseded — both helpers remain in the store layer as available APIs but are not called from the pipeline

Implementation choice:

- dense retrieval uses cosine similarity via `distance_type("cosine")`
- if a domain filter is supplied, apply it as a store-level filter on `source_domain` (BTree scalar index present)
- the native FTS index is `create_index(config=FTS(with_position=False))` built by `open_chunk_table` / `refresh_fts_index`

Eval contract:

- `tests/eval/eval_set.json` maps each question to one or more expected `source_rel_path` values (basenames also accepted when unambiguous)
- retrieval quality is measured as `Recall@10` and `MRR@10` over expected `source_rel_path` via `pixi run retrieval-check`; measured on the current 20-question set at Recall@10 = 1.000, MRR@10 = 0.713 after the native-hybrid adoption (commit `e80fd32`; prior 5-lane RRF baseline: MRR@10 = 0.669)

Source-ranking rule:

- query ranking must be source-aware, not chunk-naive
- before reranking, hybrid candidates are diversified to one representative chunk per `source_rel_path`
- if the first fetch does not yield enough unique sources, query fetches more up to `_MAX_LIMIT` rather than silently reranking a duplicate-heavy prefix
- final ordering fuses hybrid rank, rerank rank, relation-membership rank, and centrality (PageRank) rank — a four-lane RRF, with `_rrf_score(rank) = 1.0 / (_RRF_K + rank)`. The lexical lane and its `lexical_fusion_weight` config knob are collapsed into the LanceDB hybrid call and are no longer a fuser input; the config field is vestigial
- after source-aware ordering, any remaining hybrid chunks are appended behind the ranked source prefix

### Stage 4 — Rerank

Baseline:

- shipped profiles enable reranking by default through a dedicated `llama.cpp` server
- the reranker backend is independent from the Ollama embed/synthesis runtime
- if `reranker.launch.auto_start = true`, query may start `llama-server` from the configured local reranker source before the first rerank request
- if the configured reranker is unavailable, query must fall back to dense-only retrieval and surface a warning in query metadata
- alternative rerankers such as `FlashRank` are later optimizations, not the V1 baseline

### Stage 5 — Synthesise

For `search_knowledge` / `search_knowledge_deep` (legacy name: `query_lxd`):

- answer only from retrieved chunks
- cite unique `citation_label` values from chunk sources
- if zero chunks remain after filtering and retrieval, return `answer_status = "no_results"` and no synthesized claim
- if evidence is present but insufficient to ground a claim, return `answer_status = "insufficient_evidence"` and do not fabricate an answer
- if evidence is present but the synthesis model is unavailable or returns an unusable response, return `answer_status = "synthesis_unavailable"` and cite the retrieved evidence without pretending synthesis succeeded
- conflicting evidence must be surfaced in `answer_text` rather than silently collapsed into one claim

Eval normalization rule:

- `eval_set.json` should prefer explicit `source_rel_path` values
- basename-only expectations are allowed only when that basename resolves uniquely across committed searchable sources; ambiguous basenames must fail evaluation setup rather than being guessed

---

## 3. Required Tools At Query Layer

- `search_knowledge` (graph-augmented answer synthesis; was `query_lxd`)
- `search_knowledge_deep` (same plus structured `graph_context` payload)
- `search_corpus`
- `get_entity_types`
- `get_related_concepts`
- `corpus_status`

Optional later:

- `find_documents_for_concept`

That tool becomes far more useful once mention indexing is robust.

`get_related_concepts` must be driven from the real ontology graph, which includes:

- file-level `_meta.relationships`
- per-entity `relates_to`
- hierarchy links such as `parent_entity`
- taxonomy links from `taxonomy_mapping`, `maps_to_taxonomy_types`, and `taxonomy_reference`

`get_related_concepts` response contract (current implementation — `mcp/models.py:EntityNeighbor`):

- each neighbour record is `EntityNeighbor { entity_id: str, relation: str, direction: "outgoing" | "incoming" }`
- the shape is entity-centric: every neighbour is exposed by its canonical entity ID; file, taxonomy, and category nodes reachable via ontology edges are surfaced through the entity IDs they connect to, not as first-class typed nodes
- if a future need surfaces for typed non-entity neighbours (file / taxonomy / category), the model would need extending — this is a known limitation of the current shape rather than a deliberate exclusion

---

## 4. Citation Rule

Only corpus chunks are citable evidence.

Ontology context may guide reasoning, but it must not be treated as source evidence.

---

## 5. Performance Target

The target remains:

- `search_corpus` p95 <= 2.0 seconds on a warm local store
- `search_knowledge` p95 <= 12.0 seconds on a warm local store

MCP runtime:

- every tool runs as `async def`, wrapped by
  `lxd.mcp.async_runtime.run_tool`
- `mcp.tool_timeout_secs` (default `60.0`) is enforced via
  `anyio.fail_after`; breaches surface as `TimeoutError` and a
  `mcp.tool.timeout` structured log event

But the first target is **correctness and durability**, not premature micro-optimization.

If synthesis must be slower than raw retrieval, that is acceptable.

---

## 6. V1 Definition

V1 query is complete when:

- `search_corpus` returns ranked chunk results from the real built store
- `search_knowledge` returns either a cited answer or an explicit no-answer status from the real built store
- ontology lookups work without relying on mention indexing
- failure modes are explicit rather than silent
