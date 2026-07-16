# LxD Machine — MCP Specification

This is the canonical specification for the MCP surface. The full
inventory of 20 tools plus 3 resources and 2 prompts is in Section 2;
the knowledge-graph build pipeline behind the graph tools is described
in `08_KNOWLEDGE_GRAPH_SPEC.md`.

---

## 1. MCP Role

MCP is the only external interface.

There is no REST API and no web UI.

Implementation choice:

- use `fastmcp` (>=3.0) as the server/runtime library
- default to `stdio` transport for LLM-client integrations; the
  `--transport {stdio,http,sse}` CLI flag on `python -m lxd.mcp.server`
  (bound as `pixi run mcp`) selects one of the three, with `--host`
  and `--port` for the HTTP / SSE variants. `stdio` is the recommended
  and documented shape; HTTP / SSE exist for headless deployments and
  bench tooling, not for browser-facing clients

---

## 2. Tools

### Current tool inventory (20 tools)

**Corpus tools:**

| Tool | Parameters | Purpose |
|---|---|---|
| `corpus_status()` | — | Health snapshot: counts, hashes, drift warnings |
| `get_entity_types()` | — | Sorted list of canonical ontology entity IDs |
| `get_related_concepts(entity_id)` | entity_id: str | Direct ontology graph neighbours |
| `search_corpus(terms, domain?, limit?)` | terms: str, domain: str \| None, limit: int | Semantic chunk search with ranked results |
| `find_documents_for_concept(entity_id, hops?, limit?)` | entity_id: str, hops: int, limit: int | Chunks mentioning entity + graph neighbours |
| `get_corpus_relations(entity_id, limit?)` | entity_id: str, limit: int | Corpus-extracted relations for an entity |

**Knowledge graph tools:**

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

**Full answer pipeline:**

| Tool | Parameters | Purpose |
|---|---|---|
| `search_knowledge(question, domain?)` | question: str, domain: str \| None | Graph-augmented answer synthesis |
| `search_knowledge_deep(question, domain?)` | question: str, domain: str \| None | Same + structured graph context returned |
| `get_graph_overview()` | — | KG health: version, build timestamp, all counts |

### Return shapes

**`search_knowledge(question, domain?)`** returns:

- `answer_status`: one of `answered`, `no_results`, `insufficient_evidence`, `synthesis_unavailable`, or `no_retrieval_needed` (adaptive router short-circuits meta / out-of-scope questions — see `router_*` fields in `metadata` for the route rationale)
- `answer_text`
- `citations` — flat list of citation labels the answer references
- `sentence_citations` — per-sentence attribution parsed from inline `[citation_label]` markers; empty `citation_labels` on a sentence signals unattributed claim (hallucination risk)
- `metadata` — typed `KnowledgeAnswerMetadata`, not `dict[str, Any]`: `router_retrieve` (bool), `router_breadth` (`narrow` | `standard` | `broad`), `router_rationale` (str), `router_routed` (bool) always populated; `reranking_applied`, `expansion_applied`, `matched_entity_ids`, `expansion_terms`, `result_count`, `graph_context_applied`, `dense_top_k` populated only when retrieval ran
- `warnings` — buffered list of degradation notices returned in the payload; the same notices also stream live during the call over the MCP `notifications/message` channel via `Context.warning` (see Section 3 async runtime)

**`search_knowledge_deep(question, domain?)`** returns everything above plus:

- `graph_context`: structured `GraphContextData` with `level`, `entity_profiles` (with centrality scores), `community_reports`, and `claims`

**`search_corpus(terms, domain?, limit?)`** returns:

- ranked raw chunks with `chunk_id`, `document_id`, `citation_label`, `source_rel_path`, `score`, `text`, `metadata_json`

**`get_entity_summary(entity_id)`** and **`get_community_context(entity_id)`** return typed models with named sub-fields for previously JSON-encoded blobs: `aliases: list[str]`, `member_entity_ids: list[str]`, `top_predicates: list[PredicateCount]`, `top_claims: list[TopClaim]`, `top_entities: list[TopEntity]`. No opaque JSON strings inside the typed envelope.

Validation/source-of-truth rule:

- `domain` must validate against committed non-deleted `source_domain` values in `corpus_manifest`
- do not hard-code domain names in the MCP layer

### Resources (3)

Registered via `@mcp.resource` in `mcp/server.py`:

| URI template | MIME type | Purpose |
|---|---|---|
| `lxd://corpus/{path*}` | text/markdown | Raw text of a corpus file; path-traversal guarded |
| `lxd://entity/{entity_id}` | application/json | Same payload as `get_entity_summary`, exposed as a stable resource URI |
| `lxd://community/{entity_id}` | application/json | Same payload as `get_community_context`; resolves community via the entity's `community_id` |

### Prompts (2)

Registered via `@mcp.prompt` in `mcp/server.py`:

| Name | Purpose |
|---|---|
| `lxd_synthesis_preamble` | Returns the exact static preamble prepended to every synthesis prompt (transparency surface for clients auditing what the LLM sees) |
| `lxd_query_refinement(question)` | Returns a one-shot prompt clients can run against their own LLM to sharpen an ambiguous query before `search_knowledge` |

---

## 3. Server Rules

- load ontology once at startup
- compute `config_digest` and reconcile `<data_path>/config.lock` at bootstrap
- hold LanceDB table handle
- open SQLite per request through `lxd.stores.connection` (WAL + PRAGMAs)
- keep tool logic thin
- use shared lower-level query/store functions
- configure SQLite connections for concurrent read/write workloads
- document client launch using `stdio` with `pixi run mcp` from the repo root
- do not rely on inherited shell environment for `stdio` clients; required runtime settings must come from `config.yaml`, `config.{profile}.yaml`, or explicit `--config` / `--profile` launch arguments

Async tool runtime:

- every registered tool is declared `async def`
- synchronous bodies execute inside a worker thread via
  `lxd.mcp.async_runtime.run_tool(name, func, timeout_secs=...)`, which
  wraps `anyio.to_thread.run_sync(..., abandon_on_cancel=True)` in
  `anyio.fail_after`
- the timeout comes from `mcp.tool_timeout_secs` (default `60.0`);
  setting it to `0` disables the hard cap
- timeouts and exceptions emit `mcp.tool.timeout` /
  `mcp.tool.error` structured log events before propagating

API-surface stability:

- `tests/golden/mcp_tool_manifest.json` captures the current tool names,
  parameter lists, and required fields; any change must be reviewed and
  the golden file refreshed with `pytest --update-golden`

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
