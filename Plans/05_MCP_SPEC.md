# LxD Machine — MCP Specification

**Note:** This document was the original Phase 0–4 MCP specification. Phase 5 added 14 knowledge graph tools and renamed `query_lxd` to `search_knowledge`. See `08_KNOWLEDGE_GRAPH_SPEC.md` Section 5.7 for the full Phase 5 tool specification.

---

## 1. MCP Role

MCP is the only external interface.

There is no REST API and no web UI.

Implementation choice:

- use `fastmcp` as the server/runtime library
- run exclusively over `stdio` transport
- do not expose HTTP or SSE endpoints

---

## 2. Tools

### Current tool inventory (20 tools)

**Corpus tools (Phase 0–4):**

| Tool | Parameters | Purpose |
|---|---|---|
| `corpus_status()` | — | Health snapshot: counts, hashes, drift warnings |
| `get_entity_types()` | — | Sorted list of canonical ontology entity IDs |
| `get_related_concepts(entity_id)` | entity_id: str | Direct ontology graph neighbours |
| `search_corpus(terms, domain?, limit?)` | terms: str, domain: str \| None, limit: int | Semantic chunk search with ranked results |
| `find_documents_for_concept(entity_id, hops?, limit?)` | entity_id: str, hops: int, limit: int | Chunks mentioning entity + graph neighbours |
| `get_corpus_relations(entity_id, limit?)` | entity_id: str, limit: int | Corpus-extracted relations for an entity |

**Knowledge graph tools (Phase 5):**

| Tool | Parameters | Purpose |
|---|---|---|
| `get_entity_summary(entity_id)` | entity_id: str | Full entity profile: centrality, claims, community |
| `get_community_context(entity_id)` | entity_id: str | Community report for entity's community |
| `get_similar_entities(entity_id, limit?)` | entity_id: str, limit: int | Entity KNN via LanceDB vector search |
| `search_entities(query, limit?)` | query: str, limit: int | Entity name/alias search, ranked by PageRank |
| `inspect_evidence(relation_id)` | relation_id: str | Audit trail for a canonical relation |
| `find_path_between_entities(source, target, max_hops?)` | source: str, target: str, max_hops: int | Shortest unweighted path |
| `find_weighted_path(source, target)` | source: str, target: str | Confidence-weighted Dijkstra path |
| `get_hub_entities(limit?)` | limit: int | Top entities by PageRank |
| `find_bridge_entities(limit?)` | limit: int | Top entities by betweenness centrality |
| `find_foundational_entities(limit?)` | limit: int | Top entities by closeness centrality |
| `get_entity_graph_stats()` | — | KG statistics: counts, version, build time |

**Full answer pipeline (Phase 5):**

| Tool | Parameters | Purpose |
|---|---|---|
| `search_knowledge(question, domain?)` | question: str, domain: str \| None | Graph-augmented answer synthesis |
| `search_knowledge_deep(question, domain?)` | question: str, domain: str \| None | Same + structured graph context returned |
| `get_graph_overview()` | — | KG health: version, build timestamp, all counts |

### Return shapes

**`search_knowledge(question, domain?)`** returns:

- `answer_status`: `answered`, `no_results`, `insufficient_evidence`, or `synthesis_unavailable`
- `answer_text`
- source `citations`
- query `metadata` including `reranking_applied`, `expansion_applied`, `matched_entity_ids`, `expansion_terms`, `graph_context_applied`, and `result_count`
- `warnings`

**`search_knowledge_deep(question, domain?)`** returns everything above plus:

- `graph_context`: structured data with `level`, `entity_profiles` (with centrality scores), `community_reports`, and `claims`

**`search_corpus(terms, domain?, limit?)`** returns:

- ranked raw chunks with `chunk_id`, `document_id`, `citation_label`, `source_rel_path`, `score`, `text`, `metadata_json`

Validation/source-of-truth rule:

- `domain` must validate against committed non-deleted `source_domain` values in `corpus_manifest`
- do not hard-code domain names in the MCP layer

---

## 3. Server Rules

- load ontology once at startup
- hold LanceDB table handle
- open SQLite per request
- keep tool logic thin
- use shared lower-level query/store functions
- configure SQLite connections for concurrent read/write workloads
- document client launch using `stdio` with `pixi run mcp` from the repo root
- do not rely on inherited shell environment for `stdio` clients; required runtime settings must come from `config.yaml`, `config.{profile}.yaml`, or explicit `--config` / `--profile` launch arguments

---

## 4. Validation Rules

Every tool must validate:

- non-empty required strings
- allowed domains
- sane limits

Errors must be explicit and user-facing.

---

## 5. Operational Rule

The MCP server must remain usable even if:

- the store is partially built
- mention indexing is incomplete
- reranker is unavailable
- knowledge graph has not been built

The system should degrade, not collapse.
