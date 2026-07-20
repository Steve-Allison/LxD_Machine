# LxD Machine — Knowledge Graph Specification

**Document type:** Implementation specification
**Status:** Implemented
**Version:** 4.1
**Created:** 2026-03-28
**Implemented:** 2026-03-28
**Depends on:** the ingest pipeline complete, 14,461 extracted relations committed
**Reference implementation:** Knowledge Machine (Neo4j-based GraphRAG — architectural patterns adopted, not the database)
**API verification date:** 2026-03-28 (NetworkX 3.6.1, FastMCP 3.1.1, LanceDB 0.21+)

### Implementation Notes

- **Community detection:** Louvain via NetworkX selected as default. graspologic (Leiden) was removed from pixi.toml due to irreconcilable dependency conflict with fastmcp (beartype version clash). The Leiden codepath remains in `communities.py` with ImportError fallback for manual installations.
- **Synthesis integration:** Graph context is prepended to the synthesis prompt in `src/lxd/synthesis/answering.py` (not a separate `prompts.py` — see Step 7 notes).
- **MCP tools:** 14 graph tools implemented (11 from Step 8 + `search_knowledge`, `search_knowledge_deep`, `get_graph_overview`). Total MCP tools: 20.

---

## 1. Rule Zero Compliance

No performance or capacity claim is accepted until benchmarked on the real corpus:

- Community detection algorithm (Leiden vs Louvain) benchmarked on the actual combined entity graph before committing
- Centrality computation time measured on the real graph before committing to all 6 metrics (`pagerank`, `betweenness`, `closeness`, `in_degree`, `out_degree`, `eigenvector`)
- Claim extraction pilot: 50 chunks through GPT-4o-mini before committing to full extraction — quality criteria defined in Step 1 acceptance
- Entity summary deterministic generation verified against 10 entities before full build
- Optional LLM enrichment measured on pilot batch of 10 entities for quality comparison
- Query routing thresholds tuned against the eval set — no threshold is accepted without eval evidence

---

## 2. Tech Stack Additions

- **`graspologic` is optional and not in `pixi.toml`** — removed due to an irreconcilable dependency conflict with `fastmcp` (beartype version clash) per Section 1's Implementation Notes. Community detection defaults to `networkx.algorithms.community.louvain_communities` (zero new deps). The Leiden codepath remains in `ontology/communities.py` behind an `ImportError`-guarded import for operators who install graspologic manually into the pixi env.
- No other new runtime dependencies. NetworkX 3.6+ provides all centrality and path algorithms. LLM calls use the existing shared OpenAI client (via the OpenAI-compat endpoint for Ollama fallback per `ingest/llm_client.py`) with schema-enforced `.parse()` on the sync path.

### Verified API Constraints

| Library | Constraint | Impact |
|---|---|---|
| graspologic `leiden()` | Does NOT support directed graphs (`check_directed=True` raises error) | Must convert to undirected before community detection |
| NetworkX `louvain_communities()` | Supports directed graphs natively (Directed Louvain modularity) | Preferred if Louvain is selected |
| NetworkX `eigenvector_centrality()` | Computes left dominant eigenvector on directed graphs (predecessor-based) | Use `G.reverse()` for successor-based; use `eigenvector_centrality_numpy` as convergence fallback |
| NetworkX `betweenness_centrality()` | "Not guaranteed to be correct if edge weights are floating point numbers" | Use unweighted betweenness or integer-scaled weights |
| NetworkX `degree_centrality()` | On MultiDiGraph, normalised values can exceed 1.0 | Use raw degree (`G.degree()`) instead of normalised `degree_centrality()` |
| LanceDB | Supports multiple tables in the same database | Entity embeddings stored in a dedicated LanceDB table, not SQLite JSON |
| FastMCP 3.1.1 | `@mcp.tool()` API unchanged from 2.x; decorators now keep functions as functions | No migration needed; current code is compatible |

---

## 3. New SQLite Tables

### 3.1 `claims`

```sql
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_rel_path TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    subject_entity_id TEXT,
    object_entity_id TEXT,
    claim_type TEXT NOT NULL DEFAULT 'assertion',
    confidence REAL NOT NULL,
    extraction_model TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunk_rows(chunk_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_claims_object ON claims(object_entity_id);
CREATE INDEX IF NOT EXISTS idx_claims_chunk ON claims(chunk_id);
CREATE INDEX IF NOT EXISTS idx_claims_document ON claims(document_id);
```

**ID generation:** `claim_id = blake3(chunk_id + claim_text)`. Deterministic — reprocessing the same chunk produces the same IDs.

**Claim types** (mapped from instructional design assertion categories):

| `claim_type` | Meaning | Example |
|---|---|---|
| `assertion` | Factual statement | "Bloom's taxonomy has six cognitive levels" |
| `definition` | Defines a concept | "Cognitive load is the mental effort required during learning" |
| `comparison` | Relates two concepts | "Formative assessment is less formal than summative assessment" |
| `causal` | States cause and effect | "Spaced repetition improves long-term retention" |
| `procedural` | Describes a process or step | "The ADDIE model begins with an analysis phase" |

`subject_entity_id` and `object_entity_id` are nullable — claims that state a fact about a single entity have only a subject; claims about the domain in general may have neither. Claims with no entity linkage are still valuable for community summaries where they provide thematic context.

### 3.2 `entity_profiles`

```sql
CREATE TABLE IF NOT EXISTS entity_profiles (
    entity_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT '',
    aliases_json TEXT NOT NULL DEFAULT '[]',
    deterministic_summary TEXT NOT NULL,
    llm_summary TEXT,
    chunk_count INTEGER NOT NULL,
    doc_count INTEGER NOT NULL,
    mention_count INTEGER NOT NULL,
    claim_count INTEGER NOT NULL DEFAULT 0,
    top_predicates_json TEXT NOT NULL DEFAULT '[]',
    top_claims_json TEXT NOT NULL DEFAULT '[]',
    pagerank REAL NOT NULL DEFAULT 0.0,
    betweenness REAL NOT NULL DEFAULT 0.0,
    closeness REAL NOT NULL DEFAULT 0.0,
    in_degree INTEGER NOT NULL DEFAULT 0,
    out_degree INTEGER NOT NULL DEFAULT 0,
    eigenvector REAL NOT NULL DEFAULT 0.0,
    community_id INTEGER,
    source_hash TEXT NOT NULL,
    generated_at TEXT NOT NULL
);
```

- `entity_type`, `domain`: Persisted from the ontology YAML at profile-build time. Avoids MCP tools needing to join the ontology on every call.
- `deterministic_summary`: Always generated from graph structure. No LLM. Reproducible and auditable.
- `llm_summary`: GPT-4o-mini enrichment via `--enrich` flag. NULL if not yet generated.
- `community_id`: Denormalised from `entity_communities` for convenience. `entity_communities` is the authoritative source; this column is updated whenever community detection runs.
- `in_degree` / `out_degree`: Raw degree counts (not normalised `degree_centrality()` which can exceed 1.0 on MultiDiGraph).
- `doc_count`: `COUNT(DISTINCT source_rel_path)` across chunks that mention this entity (joined via `mention_rows`).
- `source_hash`: `blake3(*chunk_ids, str(pr_rank), str(bt_rank), str(cl_rank), str(in_degree), str(out_degree), str(community_id), *claim_ids)`. Uses **rank positions** for the three float-valued centrality metrics (PageRank, betweenness, closeness) rather than the raw floats — this suppresses sub-rank float-noise rebuilds so the profile only regenerates when the entity's centrality actually reordered. Integer metrics (`in_degree`, `out_degree`) go in verbatim. `eigenvector` is intentionally omitted from the hash (it is unstable across small graph edits). If any hashed input changes, the profile is rebuilt.
- Entity embeddings are stored in a dedicated **LanceDB table** (not SQLite JSON) — see Section 4.

### 3.3 `entity_communities`

```sql
CREATE TABLE IF NOT EXISTS entity_communities (
    entity_id TEXT NOT NULL,
    community_id INTEGER NOT NULL,
    community_level INTEGER NOT NULL DEFAULT 0,
    modularity_class TEXT,
    assigned_at TEXT NOT NULL,
    PRIMARY KEY (entity_id, community_level)
);
CREATE INDEX IF NOT EXISTS idx_entity_communities_community_level
ON entity_communities(community_id, community_level);
CREATE INDEX IF NOT EXISTS idx_entity_communities_level
ON entity_communities(community_level);
```

**Authoritative source** for community assignments. `entity_profiles.community_id` is denormalised from this table. Composite PK on `(entity_id, community_level)` supports the hierarchical-community path (Section 5 Step 3): one entity can appear at multiple levels of the community hierarchy.

### 3.4 `community_reports`

```sql
CREATE TABLE IF NOT EXISTS community_reports (
    community_id INTEGER NOT NULL,
    community_level INTEGER NOT NULL DEFAULT 0,
    parent_community_id INTEGER,
    member_count INTEGER NOT NULL,
    member_entity_ids_json TEXT NOT NULL,
    deterministic_summary TEXT NOT NULL,
    llm_summary TEXT,
    top_entities_json TEXT NOT NULL DEFAULT '[]',
    top_claims_json TEXT NOT NULL DEFAULT '[]',
    intra_community_edge_count INTEGER NOT NULL DEFAULT 0,
    source_hash TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    PRIMARY KEY (community_id, community_level)
);
CREATE INDEX IF NOT EXISTS idx_community_reports_level
ON community_reports(community_level);
CREATE INDEX IF NOT EXISTS idx_community_reports_parent
ON community_reports(parent_community_id, community_level);
```

Composite PK on `(community_id, community_level)` and the `parent_community_id` column support the hierarchical-community tree: level 0 is the finest partition; each higher level's report has its immediate-parent `community_id` recorded (`None` at the root).

- `deterministic_summary`: Built from member entity names, top claims, and relationship types. No LLM.
- `llm_summary`: Optional enrichment.
- `top_entities_json`: Top members by PageRank within the community.
- `top_claims_json`: Representative claims from community members.
- `source_hash`: `blake3(sorted(member_entity_ids) + sorted(member_source_hashes))`. Changes when community membership changes or any member's profile inputs change.

**Cleanup on community rerun:** When community detection produces new assignments, stale reports are removed: `DELETE FROM community_reports WHERE community_id NOT IN (SELECT DISTINCT community_id FROM entity_communities)`. This handles resolution changes that alter community count.

### 3.5 `relations` (canonical)

```sql
CREATE TABLE IF NOT EXISTS relations (
    relation_id TEXT PRIMARY KEY,
    subject_entity_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_entity_id TEXT NOT NULL,
    support_count INTEGER NOT NULL DEFAULT 0,
    avg_confidence REAL NOT NULL DEFAULT 0.0,
    min_confidence REAL NOT NULL DEFAULT 0.0,
    max_confidence REAL NOT NULL DEFAULT 0.0,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_spo
ON relations(subject_entity_id, predicate, object_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_entity_id);
```

**`relation_id` contract:** `relation_id = blake3(subject_entity_id + predicate + object_entity_id)`. Canonical per unique triple — NOT per chunk. One row per logical edge in the knowledge graph, with aggregated statistics across all supporting evidence.

**Data flow between `extracted_relations` and `relations`:**

- `extracted_relations` is the **write-path** during ingest. Per-chunk rows, FK CASCADE from `chunk_rows`. Owned by the ingest pipeline.
- `relations` is the **read-path** for graph queries. Canonical per triple, rebuilt from `extracted_relations` during graph build. Owned by the graph build pipeline.
- Both tables coexist. `extracted_relations` is source-of-truth for raw extraction output. `relations` is the consolidated view used by entity profiles, MCP tools, and query routing.

### 3.6 `relation_evidence`

```sql
CREATE TABLE IF NOT EXISTS relation_evidence (
    evidence_id TEXT PRIMARY KEY,
    relation_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    surface_subject TEXT NOT NULL,
    surface_object TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    confidence REAL NOT NULL,
    extraction_model TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    FOREIGN KEY(relation_id) REFERENCES relations(relation_id) ON DELETE CASCADE,
    FOREIGN KEY(chunk_id) REFERENCES chunk_rows(chunk_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_relation_evidence_relation ON relation_evidence(relation_id);
CREATE INDEX IF NOT EXISTS idx_relation_evidence_chunk ON relation_evidence(chunk_id);
```

**ID generation:** `evidence_id = blake3(relation_id + chunk_id)`. One evidence row per (canonical triple × chunk) pair.

**Lifecycle:** Both `relations` and `relation_evidence` are **derived tables** — fully rebuilt from `extracted_relations` during the graph build (Step 4). No `superseded_at` column, no soft-delete: latest-only model means truncate and rebuild. Between builds, FK CASCADE on `chunk_id` cleans up dangling evidence if a chunk is deleted during re-ingest. FK CASCADE on `relation_id` ensures evidence is removed if its parent relation disappears.

### 3.7 `graph_build_state`

```sql
CREATE TABLE IF NOT EXISTS graph_build_state (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    current_phase TEXT NOT NULL DEFAULT 'pending',
    graph_version INTEGER NOT NULL,
    relations_consolidated INTEGER NOT NULL DEFAULT 0,
    evidence_rows_built INTEGER NOT NULL DEFAULT 0,
    claims_extracted INTEGER NOT NULL DEFAULT 0,
    entity_profiles_built INTEGER NOT NULL DEFAULT 0,
    communities_detected INTEGER NOT NULL DEFAULT 0,
    community_reports_built INTEGER NOT NULL DEFAULT 0,
    centrality_computed INTEGER NOT NULL DEFAULT 0,
    entity_embeddings_computed INTEGER NOT NULL DEFAULT 0,
    llm_enrichment_count INTEGER NOT NULL DEFAULT 0,
    notes_json TEXT NOT NULL DEFAULT '[]'
);
```

State machine for resumable graph builds. `current_phase` tracks where to resume from:

`evidence → claims → entity_graph → centrality → communities → entity_profiles → community_reports → [optional: llm_enrichment] → complete`

This is the **default serial execution order**. The dependency graph (Section 6) shows which phases COULD run in parallel — `evidence`, `claims`, and `entity_graph` have no mutual dependencies. The serial order is the implementation default; `--phase <name>` allows advanced users to run individual phases independently (responsibility on the user to ensure dependencies are met).

Entity embeddings are computed as part of `entity_profiles` phase (not a separate phase).

### 3.8 `graph_metadata`

```sql
CREATE TABLE IF NOT EXISTS graph_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Stores `graph_version` (integer, incremented on each full build), algorithm choices, and build parameters.

---

## 4. LanceDB Entity Embeddings Table

Entity embeddings are stored in a dedicated LanceDB table (`entity_embeddings`), not as JSON in SQLite. This is consistent with how chunk embeddings are stored and enables native vector similarity search for entity-level KNN.

```python
# Schema (PyArrow)
entity_embeddings_schema = pa.schema([
    pa.field("entity_id", pa.utf8()),
    pa.field("label", pa.utf8()),
    pa.field("community_id", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), config.models.embed_dims)),
])
```

**API pattern** (verified against LanceDB 0.21+):

```python
# Actual write path lives in stores/lancedb.upsert_entity_embeddings,
# using LanceDB merge_insert for atomic upsert on entity_id:
db = lancedb.connect(store_paths.lancedb_path)
entity_table = open_entity_table(db, vector_size=embed_dims)  # ensures BTree scalar index on entity_id
(entity_table.merge_insert("entity_id")
    .when_matched_update_all()
    .when_not_matched_insert_all()
    .execute(entity_records))

# KNN query — actual path uses cosine similarity on the fully-async path
# via search_similar_entities in stores/lancedb.py:
results = search_similar_entities(entity_table, query_vector=query_vector, limit=k)
```

Each entity embedding is the **L2-normalised** mean of the embeddings of up to `entity_summary_max_chunks` chunks that reference the entity (loaded via `load_chunk_ids_for_entity(..., limit=max_chunks)` in `cli/graph.py`). Chunk vectors are pulled from the existing `chunk_vectors` LanceDB table — zero additional OpenAI API calls. Full-replace uses `merge_insert(on="entity_id").when_matched_update_all().when_not_matched_insert_all().when_not_matched_by_source_delete()` so the full-rebuild path is a single atomic pass with no delete-then-add split-brain window.

---

## 5. Phase Order

### Step 1 — Claim Extraction

Extract fine-grained assertions from chunks. Claims are the evidence layer that makes entity and community summaries meaningful.

**Architecture decision:** Claim extraction runs as a **post-ingest graph build step** (not during ingest). Rationale:

- Relations are lightweight (entity pairs + predicate) and cheap to extract at ingest time.
- Claims require more context and produce more output — better suited to a dedicated build phase.
- Claims depend on entity mentions being complete for the whole corpus, not just a single source file.
- Separating from ingest keeps the ingest pipeline fast and focused.

**New:** `src/lxd/ingest/claims.py`
**Modified:** `src/lxd/stores/sqlite/claims.py`, `src/lxd/stores/models.py`

For each chunk with ≥1 entity mention, extract claims via GPT-4o-mini (with Ollama qwen3:14b fallback):

```text
Given this text and these entities, extract factual assertions.
Each claim should be a single statement that could be true or false.
Link claims to subject/object entities where applicable.
A claim may have only a subject entity, or no entity linkage if it states a general domain fact.

Return JSON (no markdown fences): {"claims": [{
  "claim_text": "...",
  "claim_type": "assertion" | "definition" | "comparison" | "causal" | "procedural",
  "subject": "entity_id or null",
  "object": "entity_id or null",
  "confidence": 0.85    // 0.7-1.0 for well-supported; 0.4-0.69 for weaker
}]}
```

Actual prompt lives in `src/lxd/ingest/claims.py:_CLAIM_BASE_PROMPT` and includes additional LxD role framing plus the confidence-band rule and the no-markdown-fences rule. The persisted row on `claims` carries `claim_type` (defaulting to `assertion` if omitted) alongside the other fields.

- Stored per-chunk with FK cascade on chunk delete
- Per-chunk atomic commits
- Incremental: skip chunks whose content hash hasn't changed since last claim extraction

**Cost estimate:** ~23,000 chunks with ≥1 entity mention (2.2M mentions across 23.6K chunks, average ~93 mentions/chunk — nearly all chunks qualify). At GPT-4o-mini pricing: **~$5–10** for full extraction.

**Acceptance:**

- Claims table populated for all qualifying chunks
- Pilot batch of 50 chunks reviewed before full extraction with criteria:
  - ≥80% of claims are factually grounded in the chunk text (not hallucinated)
  - ≥70% of claims link to at least one entity
  - Average ≥2 claims per chunk
- Extraction is incremental (content-hash-based staleness detection)
- CLI: `pixi run build-graph --phase claims`

---

### Step 2 — Combined Entity Graph + Centrality

Merge ontology edges + corpus relations into one weighted `NetworkX.MultiDiGraph`. Compute all centrality metrics. **Can run in parallel with Step 1.**

**New:** `src/lxd/ontology/entity_graph.py`
**Modified:** `src/lxd/stores/sqlite/kg_profiles.py`, `src/lxd/stores/sqlite/kg_relations.py`, `src/lxd/stores/models.py`

**Entity graph construction:**

- Load ontology `MultiDiGraph` (existing)
- Load `extracted_relations` (per-chunk rows) where `confidence >= config.knowledge_graph.min_relation_confidence`
- Group by `(subject_entity_id, predicate, object_entity_id)` — same grouping key as the canonical `relation_id` in Step 4
- For each unique triple, add a single edge with `weight = max(confidence)` across all supporting rows, `origin_kind="corpus"`, `support_count = len(rows)`
- This in-memory deduplication is consistent with but independent of Step 4's consolidation into the `relations` table (both phases can run in parallel because both read from the same source: `extracted_relations`)

**Centrality computation (6 metrics on the directed MultiDiGraph):**

- **PageRank** (`networkx.pagerank`) — works natively on MultiDiGraph; multigraph edge weights are summed automatically
- **Betweenness** (`networkx.betweenness_centrality`) — use **unweighted** (float weights are documented as unreliable); works on directed graphs
- **Closeness** (`networkx.closeness_centrality`) — uses inward distance on directed graphs (Wasserman-Faust formula handles disconnected components)
- **In-degree / Out-degree** (`G.in_degree()` / `G.out_degree()`) — raw counts, not normalised `degree_centrality()` (which can exceed 1.0 on MultiDiGraph)
- **Eigenvector** (`networkx.eigenvector_centrality_numpy`) — use NumPy/LAPACK solver (guaranteed convergence, no iteration limit issues); computes left dominant eigenvector on directed graphs

**Acceptance:**

- Combined graph loads successfully; benchmark load time (Rule Zero — no hard gate until measured)
- All 325 entities present as nodes
- All 6 centrality metrics computed and stored
- Benchmark centrality computation time on the real graph (Rule Zero — no hard gate until measured)
- Eigenvector centrality uses `_numpy` variant — verify convergence on the real graph

**Dependencies:** the ingest pipeline (extracted relations must exist). Does NOT depend on Step 1 (claims).

---

### Step 3 — Community Detection

Partition entities into communities. Store assignments.

**New:** `src/lxd/ontology/communities.py`
**Modified:** `src/lxd/stores/sqlite/kg_profiles.py`

**Algorithm selection (benchmark both):**

| Algorithm | Library | Directed support | Determinism | Notes |
|---|---|---|---|---|
| Leiden | graspologic `leiden()` | **No** — must convert to undirected via `G.to_undirected()` | Reproducible with `random_seed` | Rust-backed, fast; `trials` param for best-of-N |
| Louvain | NetworkX `louvain_communities()` | **Yes** — natively supports DiGraph/MultiDiGraph | Reproducible with `seed` param | Pure Python; "Directed Louvain" modularity formula |

- Configurable resolution (`config.knowledge_graph.community_resolution`)
- Sweep 3–5 resolution values and report community count / modularity for each
- Store community assignments in `entity_communities`
- Write `community_id` back to `entity_profiles` (denormalised)

**Acceptance:**

- Reproducible for same graph + resolution + seed
- Every entity with ≥1 edge gets a community assignment
- Isolated entities assigned to singleton communities
- Benchmark report records algorithm, resolution, community count, modularity, and runtime
- Expected: 10–40 communities

**Dependencies:** Step 2 (needs combined entity graph).

---

### Step 4 — Relations Consolidation + Evidence Provenance

Consolidate per-chunk `extracted_relations` into canonical `relations` table and populate `relation_evidence` with per-chunk provenance. **Can run in parallel with Steps 1 and 2** (only depends on the ingest pipeline data).

**New:** `src/lxd/ontology/evidence.py`
**Modified:** `src/lxd/ingest/relations.py`, `src/lxd/stores/sqlite/kg_relations.py`

**Step 1 — Consolidate canonical relations:**

- Read all rows from `extracted_relations`
- Group by `(subject_entity_id, predicate, object_entity_id)`
- For each unique triple, compute `relation_id = blake3(subject + predicate + object)`
- Aggregate: `support_count`, `avg_confidence`, `min_confidence`, `max_confidence`, `first_seen_at`, `last_seen_at`
- Truncate `relations` table, insert consolidated rows

**Step 2 — Build evidence records:**

For each row in `extracted_relations`:

- Compute `relation_id` from its triple (same as Step 1)
- Compute `evidence_id = blake3(relation_id + chunk_id)`
- Look up chunk text from `chunk_rows`
- Look up surface forms from `mention_rows` (`surface_form`, `start_char`, `end_char` for each entity mention in this chunk)
- Extract the text window around the subject and object mentions as `evidence_text`
- Record surface forms for subject and object

Truncate `relation_evidence` table, insert all evidence rows.

**Lifecycle:** Both tables are **derived** — fully rebuilt from `extracted_relations` on each graph build. No incremental maintenance, no soft-delete. Between builds, FK CASCADE on `chunk_id` cleans up dangling evidence if a chunk is re-ingested.

**Forward integration (not yet implemented):** the plan of the day was to have `extract_relations_for_chunk` also return surface forms at extraction time so the evidence build could read them directly from `extracted_relations`. Current code (`src/lxd/ingest/relations.py`) does not do this — surface forms continue to be derived from `mention_rows` at evidence-build time (`src/lxd/ontology/evidence.py`). Kept as a possible future optimisation; the current cost is one extra SQLite scan per build, which is negligible against the LLM-call cost of relation extraction itself.

**Acceptance:**

- Every row in `relations` has ≥1 row in `relation_evidence`
- Evidence records include surface forms and source text
- `relations.support_count` matches actual evidence row count for each `relation_id`
- Full rebuild from `extracted_relations` completes without LLM calls (pure SQLite read/write)

**Dependencies:** the ingest pipeline only (`extracted_relations`, `chunk_rows`, `mention_rows`). Does NOT depend on Step 1 (claims) or Step 2 (combined graph).

---

### Step 5 — Entity Profiles (Deterministic)

Build a complete profile for each entity from graph structure. No LLM required. Entity embeddings computed and stored in LanceDB.

**New:** `src/lxd/ontology/profiles.py`
**Modified:** `src/lxd/stores/sqlite/kg_profiles.py`, `src/lxd/stores/lancedb.py`

Per entity, assemble deterministically:

- Label, aliases, `entity_type`, and `domain` (from ontology YAML — persisted on profile)
- Centrality scores (from Step 2)
- Community assignment (from Step 3)
- Top predicates by frequency (from canonical `relations` table — Step 4)
- Top claims (from `claims` table, ranked by confidence — Step 1)
- `chunk_count` (from `mention_rows` — distinct chunk_ids)
- `doc_count` (`COUNT(DISTINCT source_rel_path)` across chunks mentioning this entity, joined via `mention_rows`)
- `mention_count` (from `mention_rows`)
- Deterministic summary template:

```text
{label} is a {entity_type} entity in the {domain} domain.
It has {mention_count} mentions across {chunk_count} chunks from {doc_count} source documents.
Centrality: PageRank {rank}/{total} | Betweenness {rank}/{total} | Closeness {rank}/{total}.
Community: {community_id} ({community_member_count} members).
Key relationships: {top_5_relations_as_subject_predicate_object}.
Key claims: {top_3_claims_by_confidence}.
```

- Compute `source_hash` per the definition in Section 3.2 — incorporates all profile inputs (chunk_ids, centrality scores, community_id, claim_ids). Any upstream change triggers rebuild.
- Incremental: skip entities whose `source_hash` hasn't changed

**Entity embeddings (in LanceDB):**

- For each entity with ≥`entity_embedding_min_mentions` mentions (default 3):
  - Retrieve embeddings of the top N chunks (by mention frequency) from the LanceDB chunks table
  - Compute mean embedding
  - Store in LanceDB `entity_embeddings` table (see Section 4)
- Enables `get_similar_entities` MCP tool via native LanceDB vector search

**Acceptance:**

- All 325 entities have deterministic profiles
- Profiles are fully reproducible (same input = same output, always)
- Entity embeddings computed for all entities with ≥3 mentions, stored in LanceDB
- Incremental rebuild skips unchanged entities
- Zero LLM calls. Zero API cost.

**Dependencies:** Steps 1 (claims for `top_claims`), 2 (centrality), 3 (communities), 4 (canonical `relations` for `top_predicates`).

---

### Step 6 — Optional LLM Summary Enrichment

Generate richer prose summaries via GPT-4o-mini for entities and communities. **Entirely optional.** The system is fully functional with deterministic summaries from Step 5.

**Modified:** `src/lxd/ontology/profiles.py`, `src/lxd/stores/sqlite/kg_profiles.py`

Per entity (where `llm_summary IS NULL` or `source_hash` changed):

- Feed the entity's label + entity_type + domain + deterministic summary + top predicates + top claims to GPT-4o-mini (with Ollama qwen3:14b fallback). Chunk text is not passed in — the deterministic summary already carries the corpus-derived facts and passing chunks would inflate the prompt without measurable gain
- Generate 150–300 word prose summary
- Store in `entity_profiles.llm_summary`

Per community (where `llm_summary IS NULL` or `source_hash` changed):

- Feed the community's member deterministic summaries + top claims to GPT-4o-mini (with Ollama qwen3:14b fallback). Intra-community edges are surfaced through the deterministic summaries rather than passed as a separate prompt input
- Generate 200–400 word community narrative
- Store in `community_reports.llm_summary`

**Cost:** 325 entity calls + 10–40 community calls ≈ ~365 calls, **~$0.45** total. Incremental: only changed entities.

**Acceptance:**

- System works identically with `llm_enrichment: false`
- LLM summaries stored separately from deterministic ones (never overwritten)
- Incremental rebuild skips unchanged entities/communities
- CLI flag: `pixi run build-graph --enrich`

**Dependencies:** Step 5.

---

### Step 7 — Graph-Aware Query Routing

Multi-layer context augmentation for synthesis. Graph context is **additive** — it frames chunk evidence, it does not replace it.

**New:** `src/lxd/retrieval/graph_routing.py`
**Modified:** `query_pipeline.py`, `expansion.py`, `settings/models.py`, `src/lxd/synthesis/answering.py`

**Important architectural note:** Graph context and chunk results are NOT fused via RRF. RRF operates on ranked lists of the same item type (e.g. chunks from vector search + chunks from fulltext search). Entity profiles, community reports, and claims are different item types — they are prepended to the synthesis context as structured framing, not ranked alongside chunks.

**Context layers:**

1. **Chunk layer** (existing, unchanged) — dense search → rerank → lexical/relation fusion → ranked chunks
2. **Entity context** — entity profiles for matched entities, ordered by PageRank
3. **Community context** — community reports for matched entities' communities
4. **Claim context** — claims linked to matched entities, ranked by confidence

**Routing logic:**

All queries execute chunk-level retrieval (unchanged baseline). Graph layers are appended to synthesis context based on entity matching:

- Entity matching identifies entities two ways: (a) Aho-Corasick over the question surface text, then (b) `_augment_with_embedding_neighbours` widens the matched set with entities whose `entity_embeddings` vector is nearest to the query vector. The union is what feeds downstream expansion — semantic neighbours land alongside surface-form hits
- **1+ entities matched:** Include top `max_entity_context` (default 5) entity profiles, ranked by PageRank, in synthesis context
- **Matched entities span 2+ communities:** Also include top `max_community_context` (default 3) community reports, ranked by member PageRank sum
- **Claims:** Include top `max_claim_context` (default 10) claims for matched entities, ranked by confidence
- Multi-hop expansion uses the combined entity graph (up to `multi_hop_max` hops, default 3)
- Edge weight filtering: only traverse corpus edges above `min_relation_confidence`

**Modified `SearchOutcome`:**

```python
graph_context_level: str                    # "none", "entity", "community"
graph_entity_profiles: list[EntityProfile]  # matched entity profiles
graph_community_reports: list[CommunityReport]  # matched community reports
graph_claims: list[ClaimRecord]             # relevant claims
graph_expansion_hops: int                   # actual hops used
```

**Synthesis prompt structure** (when graph context is present):

```text
## Graph Context

### Entity Profiles
{entity_profiles, ordered by PageRank, up to max_entity_context}

### Community Context
{community_reports, if entities span 2+ communities, up to max_community_context}

### Related Claims
{claims for matched entities, ranked by confidence, up to max_claim_context}

### Conflicting Claims
{query-time heuristic pairs — synthesis must surface both sides; never silently pick a winner}

## Source Evidence
{ranked chunks from existing retrieval pipeline — may include claim-boosted graph-lane chunks}
```

Claim-linked chunk IDs also participate in retrieval RRF as a **graph lane** (`retrieval.graph_lane_enabled`) — they are ranked candidates, not only prompt framing. Path-finding tools remain on-demand and are not forced into every deep answer.

The synthesis module (`src/lxd/synthesis/answering.py`) prepends graph context before chunk evidence. `graph_routing.py` builds the context (including `detect_claim_conflicts`); `answering.py._build_prompt()` formats it. When no graph context is present, the prompt is identical to the graph-free baseline.

**Acceptance:**

- Queries with 0 entity matches behave identically to Phase 4
- Graph context is additive (chunks always present as primary evidence)
- Benchmark latency overhead of graph lookups (Rule Zero — target <100ms, all reads are SQLite/LanceDB with no LLM calls at query time)
- No regression on eval set with graph routing active
- Eval set extended with 10 graph-specific questions to validate graph routing quality

**Dependencies:** Steps 1–5.

---

### Step 8 — MCP Tools

11 new tools. **Can run in parallel with Step 7.**

| Tool | Returns |
|---|---|
| `get_entity_summary(entity_id)` | Full entity profile: deterministic + LLM summary, centrality, top relations, top claims, community |
| `get_community_context(entity_id)` | Community report: members, deterministic + LLM summary, top claims, intra-community edges |
| `get_similar_entities(entity_id, limit=10)` | Entity-level KNN via LanceDB vector search on entity embeddings |
| `search_entities(query, limit=20)` | Entity name/alias substring search (SQLite `LIKE` on `label` + `aliases_json`), results ranked by PageRank. For 325 entities, no FTS needed. |
| `inspect_evidence(relation_id)` | Audit trail: all evidence records for a canonical relation, including surface forms and chunk provenance |
| `find_path_between_entities(source, target, max_hops=5)` | Shortest unweighted path + edge records. `max_hops` capped at `config.knowledge_graph.multi_hop_max`. Operates on `plan.ontology.graph` (the ontology-only graph loaded into memory at lifespan start), not the combined ontology+corpus graph. |
| `find_weighted_path(source, target)` | Confidence-weighted Dijkstra shortest path (weight = 1.0 − confidence). Dijkstra naturally terminates at shortest — no hop limit needed. Same graph source as `find_path_between_entities`. |
| `get_hub_entities(limit=20)` | Top entities by PageRank |
| `find_bridge_entities(limit=20)` | Top entities by betweenness centrality |
| `find_foundational_entities(limit=20)` | Top entities by closeness centrality |
| `get_entity_graph_stats()` | Node/edge counts, community count, centrality distributions, graph version, last build time |

All tools degrade gracefully when graph features aren't built (empty results, not errors). All tools read current tables directly (latest-only model — no version filtering needed).

---

### Step 9 — CLI, State Machine & Maintenance

**New:** `src/lxd/cli/graph.py`

**Build state machine:**

```text
evidence → claims → entity_graph → centrality → communities → entity_profiles → community_reports → [optional: llm_enrichment] → complete
```

Each phase writes its progress to `graph_build_state`. If interrupted, `pixi run build-graph` reads the state and resumes from the last incomplete phase. Per-entity/per-community operations within each phase use the same source_hash staleness check for incremental progress.

Note: `entity_profiles` phase includes entity embedding computation (stored in LanceDB). This is not a separate phase in the state machine.

**New pixi tasks:**

- `pixi run build-graph` — full build (all phases, incremental by default)
- `pixi run build-graph --full` — force regeneration of all phases (ignores staleness checks, does NOT wipe data)
- `pixi run build-graph --enrich` — include optional LLM enrichment
- `pixi run build-graph --dry-run` — preview: count qualifying chunks per phase, estimate API calls and cost, report stale entities/communities. Zero writes, zero API calls.
- `pixi run build-graph --phase claims` — run only claim extraction
- `pixi run build-graph --phase communities` — rerun community detection only
- `pixi run build-graph --phase profiles` — regenerate entity profiles only
- `pixi run graph-status` — print graph build state, phase progress, entity/community counts

**None of these require re-ingest or re-embedding.** The graph build pipeline operates on data already in SQLite and LanceDB.

**Acceptance:**

- `pixi run build-graph` completes end-to-end
- Interrupting at any phase leaves previously completed phases intact
- Restarting resumes from the interrupted phase
- `pixi run graph-status` reports current phase, progress counts, and graph version
- Graph version increments on each `--full` build only (not incremental)

---

## 6. Dependency Graph

```text
Step 1 (Claims)  ----------+
                               |
Step 2 (Entity Graph)      |    Step 4 (Relations + Evidence)
    |                          |        [needs the ingest pipeline only]
    +---> Step 3 (Communities)      |
    |                          |       |
    +--------------------------+-------+
    |
    v
Step 5 (Entity Profiles)  [needs Steps 1 + 2 + 3 + 4]
    |
    +---> Step 6 (Optional LLM Enrichment)
    |
    +---> Step 7 (Query Routing)          [needs Steps 1–5]
    +---> Step 8 (MCP Tools)              [needs Steps 1–5]
    +---> Step 9 (CLI & State Machine)    [needs Steps 1–5]
```

- **Steps 1, 2, and 4 all run in parallel** — no mutual dependencies; all only need the ingest pipeline data
- Step 3 runs after Step 2 (needs combined entity graph)
- Step 5 waits for **all of** Steps 1, 2, 3, and 4 (needs claims + centrality + communities + canonical relations)
- Steps 7, 8, 9 run in parallel after Step 5
- Step 6 is optional and can run any time after Step 5
- **Default serial execution order** (state machine): evidence → claims → entity_graph → centrality → communities → entity_profiles → community_reports → complete

---

## 7. API Cost Summary

| Operation | Calls | Est. Cost |
|---|---|---|
| Claim extraction (full, ~23K chunks) | ~23,000 | ~$5–10 |
| Entity profiles (deterministic) | 0 | $0 |
| Entity embeddings (reuse chunk vectors) | 0 | $0 |
| Community reports (deterministic) | 0 | $0 |
| **Full graph build (no LLM enrichment)** | **~23,000** | **~$5–10** |
| Optional LLM enrichment (325 entities + 10–40 communities) | ~365 | ~$0.45 |
| **Full graph build (with LLM enrichment)** | **~23,365** | **~$6–11** |
| Incremental (10% corpus change) | ~2,300 | ~$0.50–1 |

---

## 8. New/Modified Files

| File | Status | Purpose |
|---|---|---|
| `src/lxd/ingest/claims.py` | New | Claim extraction from chunks |
| `src/lxd/ontology/entity_graph.py` | New | Combined graph construction + centrality |
| `src/lxd/ontology/communities.py` | New | Community detection and assignment |
| `src/lxd/ontology/profiles.py` | New | Entity profiles + deterministic summaries |
| `src/lxd/ontology/evidence.py` | New | Relations consolidation + evidence provenance |
| `src/lxd/retrieval/graph_routing.py` | New | Context augmentation for synthesis |
| `src/lxd/cli/graph.py` | New | CLI commands + build state machine |
| `src/lxd/stores/sqlite/` | Modified | 8 new tables, ~30 query functions split across `claims.py`, `kg_profiles.py`, `kg_relations.py` |
| `src/lxd/stores/lancedb.py` | Modified | Entity embeddings table (create, add, search) |
| `src/lxd/stores/models.py` | Modified | 8 new dataclass records |
| `src/lxd/settings/models.py` | Modified | `KnowledgeGraphConfig` |
| `src/lxd/ingest/relations.py` | Modified | Evidence provenance capture |
| `src/lxd/retrieval/query_pipeline.py` | Modified | Graph routing integration |
| `src/lxd/retrieval/expansion.py` | Modified | Multi-hop over combined graph |
| `src/lxd/synthesis/answering.py` | Modified | Graph context sections prepended to synthesis prompt |
| `src/lxd/mcp/tools.py` | Modified | 14 new tool implementations (11 graph + 3 full-pipeline) |
| `src/lxd/mcp/server.py` | Modified | 14 new tool registrations (20 total) |
| `pixi.toml` | Modified | New tasks (`build-graph`, `graph-status`) |

---

## 9. Config Addition

```python
class KnowledgeGraphConfig(BaseModel):
    """Knowledge graph build and query settings."""
    model_config = ConfigDict(extra="forbid")

    min_relation_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Community detection
    community_resolution: float = Field(default=1.0, gt=0.0)
    community_algorithm: Literal["leiden", "louvain"] = "louvain"
    community_seed: int = Field(default=42)

    # Entity profiles
    entity_summary_max_chunks: int = Field(default=20, gt=0)
    entity_embedding_min_mentions: int = Field(default=3, ge=1)

    # Claim extraction
    claim_extraction_backend: Literal["openai", "ollama"] = "openai"
    claim_extraction_model: str = "gpt-4o-mini"
    claim_extraction_fallback_model: str = "qwen3:14b"
    claim_extraction_min_mentions: int = Field(default=1, ge=1)
    claim_max_per_chunk: int = Field(default=10, gt=0)
    claim_extraction_timeout_secs: int = Field(default=90, gt=0)
    claim_extraction_temperature: float = Field(default=0.0, ge=0.0)

    # LLM enrichment
    llm_enrichment_backend: Literal["openai", "ollama"] = "openai"
    llm_enrichment_model: str = "gpt-4o-mini"
    llm_enrichment_fallback_model: str = "qwen3:14b"
    llm_enrichment_temperature: float = Field(default=0.1, ge=0.0)
    llm_enrichment_timeout_secs: int = Field(default=30, gt=0)

    # Query routing
    multi_hop_max: int = Field(default=3, ge=1, le=5)
    max_entity_context: int = Field(default=5, gt=0)
    max_community_context: int = Field(default=3, gt=0)
    max_claim_context: int = Field(default=10, gt=0)
```

Added to `RuntimeConfig` as `knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)`.

---

## 10. Graph Versioning

**V1 contract: latest-only, no version filtering, no cross-version retention.**

All graph tables store exactly one row per entity/community/relation. Rebuilds overwrite in place. No historical versions are retained. MCP tools and query routing read current tables directly — no `WHERE graph_version = X` clauses anywhere.

**Where `graph_version` lives and why:**

| Location | Role | Queried by |
|---|---|---|
| `graph_metadata` (key=`graph_version`) | Authoritative current version counter | `graph-status` CLI, `get_entity_graph_stats()` MCP tool |
| `graph_build_state.graph_version` | Operational: which version a build run is creating/resuming | State machine resume logic |
| Data table columns (`claims`, `entity_profiles`, etc.) | **Removed in V1.** Latest-only means one row per entity — the current version is always `graph_metadata.graph_version`. A per-row stamp adds no information that `generated_at` + `graph_metadata` don't already provide. | N/A — column removed |
| LanceDB `entity_embeddings` | **No `graph_version` column.** Table is fully rebuilt during profile phase. | N/A — column absent |

**Version lifecycle:**

- Incremented on each `--full` graph build only (not incremental)
- Read by `graph-status` and `get_entity_graph_stats()` for observability
- Stored in `graph_build_state` for resume logic
- Never used for query filtering on data tables

---

## 11. Known Limitations

| Limitation | Status |
|---|---|
| **Entity disambiguation**: Aho-Corasick matches exact strings from ontology aliases. Variant forms not in the ontology are missed. | Accepted for V1. Mitigation: expand ontology aliases. Future: add fuzzy matching or NER. |
| **Entity embeddings are mean-pooled**: L2-normalised mean of the top-N chunk embeddings referencing the entity. | Accepted. Mean pooling is the standard baseline. Future: TF-IDF or attention-weighted aggregation. |
| **Fulltext search now landed**: previously deferred; the current retrieval path fuses dense k-NN and BM25 FTS inside LanceDB via `Table.search(query_type="hybrid").rerank(RRFReranker())` (commit `e80fd32`). Graph routing continues to add structured context on top of the hybrid retrieval, not in place of FTS. | Superseded — no longer a limitation. |
| **Betweenness uses unweighted edges**: NetworkX docs warn float weights are unreliable for betweenness. | Accepted. Unweighted betweenness still identifies topological bridges correctly. |

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Degenerate communities (1 giant or all singletons) | Sweep 3–5 resolution values in Step 3; select based on modularity |
| Claim extraction noise / hallucinated claims | Confidence threshold; pilot batch with quality criteria (≥80% grounded, ≥70% entity-linked, avg ≥2/chunk) |
| Claim extraction cost higher than estimated | Incremental after first build; can reduce by raising `claim_extraction_min_mentions` |
| `graspologic` heavy deps | Fall back to `networkx.algorithms.community.louvain_communities` (zero new deps, supports directed graphs) |
| Corpus relation noise degrades communities | Filter by `min_relation_confidence`; aggregate parallel edges |
| Graph build interrupted mid-phase | State machine resumes from last incomplete phase; per-entity atomic commits within phases |
| Deterministic summaries too terse | LLM enrichment available as optional overlay |
| Query routing adds latency for no benefit | Disabled by default; all graph lookups are SQLite/LanceDB reads (benchmark actual overhead — Rule Zero) |
| Eigenvector centrality fails to converge | Use `eigenvector_centrality_numpy` (LAPACK solver, guaranteed convergence) |

---

## 13. Testing Strategy

- **Unit tests:** `tests/ingest/test_claims.py`, `tests/ontology/test_entity_graph.py`, `tests/ontology/test_communities.py`, `tests/ontology/test_profiles.py`, `tests/retrieval/test_graph_routing.py`
- **Integration tests:** `tests/stores/test_sqlite_graph_tables.py` — verify all new SQLite operations including FK cascades; `tests/stores/test_lancedb_entities.py` — verify entity embeddings table creation, insert, KNN search
- **MCP tool tests:** `tests/mcp/test_graph_tools.py` — verify all 11 new tools return correct shapes and degrade gracefully when graph data is absent
- **Relations consolidation:** Verify `relations` table has one row per unique triple; verify `support_count` matches `relation_evidence` row count; verify full rebuild from `extracted_relations` is idempotent
- **Evidence provenance:** Verify every canonical relation has ≥1 evidence record; verify FK CASCADE on `chunk_id` cleans up evidence when chunk is re-ingested; verify FK CASCADE on `relation_id` cleans up evidence when relation is removed
- **Resumability:** Kill graph build mid-phase; restart; verify completion without data loss
- **Centrality:** Verify eigenvector_centrality_numpy converges on the real graph; verify betweenness is computed unweighted; verify degree counts match graph inspection
- **Eval regression:** Run existing eval set with graph routing enabled; verify no regression in `Recall@10` or `MRR@10`
- **Graph-specific eval:** Add 10 questions to eval set that specifically test graph routing (entity lookups, cross-community queries, multi-hop reasoning)
