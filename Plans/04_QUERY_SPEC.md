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

### Stage 2 — Expand

Deferred by default:

- detect ontology entities in the question
- add a small number of related terms

This stage should stay off unless benchmarking shows it improves retrieval quality on the eval set.

If expansion is weak, noisy, or unproven, query must still work without it.

Implementation rule:

- query-time expansion, when enabled, must use the ontology matcher over the question text, then expand only over the in-memory ontology graph with `config.expansion.hops` and `config.expansion.max_terms`
- expansion terms must be surfaced in query metadata so retrieval changes are auditable

### Stage 3 — Retrieve

- dense search over vectors is required for V1
- sparse text retrieval is optional and must not be assumed until it is implemented
- fusion happens only when both retrieval modes are actually present

Implementation choice:

- dense retrieval uses LanceDB vector search
- dense retrieval must use cosine similarity
- if a domain filter is supplied, apply it as a store-level filter on `source_domain`
- if sparse retrieval is added, it should use LanceDB native FTS rather than a separate search system

Eval contract:

- `eval_set.json` maps each question to one or more expected `source_rel_path` values
- dense retrieval quality is measured as `Recall@10` over expected `source_rel_path`
- rerank quality is measured as `MRR@10` over the same expected `source_rel_path` targets using the configured query candidate set

Source-ranking rule:

- query ranking must be source-aware, not chunk-naive
- before reranking, dense candidates must be diversified to one representative chunk per `source_rel_path`
- if the first dense fetch does not yield enough unique sources to satisfy the configured retrieval window, query must fetch more dense hits up to the query cap rather than silently reranking a duplicate-heavy prefix
- final source ordering may fuse dense rank, lexical rank over source metadata, and rerank rank, but all fused inputs must come from committed corpus data
- after source-aware ordering is produced, any remaining dense chunks may be appended behind the ranked source prefix

### Stage 4 — Rerank

Baseline:

- shipped profiles enable reranking by default through a dedicated `llama.cpp` server
- the reranker backend is independent from the Ollama embed/synthesis runtime
- if `reranker.launch.auto_start = true`, query may start `llama-server` from the configured local reranker source before the first rerank request
- if the configured reranker is unavailable, query must fall back to dense-only retrieval and surface a warning in query metadata
- alternative rerankers such as `FlashRank` are later optimizations, not the V1 baseline

### Stage 5 — Synthesise

For `query_lxd` only:

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

- `query_lxd`
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

`get_related_concepts` response contract:

- each neighbor record must expose both edge data and neighbor node data
- neighbor node data must include at least `neighbor_node_id`, `neighbor_node_type`, `neighbor_entity_id`, `neighbor_label`, and `neighbor_metadata`
- the tool must not pretend every neighbor is an entity; file and taxonomy neighbors are valid first-class graph results

---

## 4. Citation Rule

Only corpus chunks are citable evidence.

Ontology context may guide reasoning, but it must not be treated as source evidence.

---

## 5. Performance Target

The target remains:

- `search_corpus` p95 <= 2.0 seconds on a warm local store
- `query_lxd` p95 <= 12.0 seconds on a warm local store

But the first target is **correctness and durability**, not premature micro-optimization.

If synthesis must be slower than raw retrieval, that is acceptable.

---

## 6. V1 Definition

V1 query is complete when:

- `search_corpus` returns ranked chunk results from the real built store
- `query_lxd` returns either a cited answer or an explicit no-answer status from the real built store
- ontology lookups work without relying on mention indexing
- failure modes are explicit rather than silent
