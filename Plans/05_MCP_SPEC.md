# LxD Machine — MCP Specification

---

## 1. MCP Role

MCP is the only external interface.

There is no REST API and no web UI.

Implementation choice:

- use `fastmcp` as the server/runtime library
- run exclusively over `stdio` transport
- do not expose HTTP or SSE endpoints

---

## 2. Required Tools

### `query_lxd(question: str)`

Returns:

- `answer_status`: `answered`, `no_results`, `insufficient_evidence`, or `synthesis_unavailable`
- `answer_text`
- source citations
- query metadata including whether reranking was applied, whether ontology expansion was applied, which entity IDs were matched, which expansion terms were added, and any fallback warnings

### `search_corpus(terms: str, domain: str | None, limit: int)`

Returns:

- ranked raw chunks

Validation/source-of-truth rule:

- `domain` must validate against committed non-deleted `source_domain` values in `corpus_manifest`
- do not hard-code domain names in the MCP layer

### `get_entity_types()`

Returns:

- ontology entity list derived from the loaded `entity_types` mappings

### `get_related_concepts(entity_id: str)`

Returns:

- ontology-defined relations from the in-memory graph assembled from `_meta.relationships`, `relates_to`, and hierarchy links
- only direct one-hop neighbors; no transitive closure in V1
- relation records including `direction`, `relation_type`, `target_entity_id`, `origin_kind`, and `source_file_rel_path`

The graph must be built from the actual repo schema, not from an assumed flat list of entities.

### `corpus_status()`

Returns:

- committed corpus counts by source type
- committed corpus counts by retrieval role
- chunk count
- ontology file count
- current ontology snapshot hash
- current matcher termset hash
- ontology coverage path count
- ontology graph relation count
- ontology validation issue count and issue samples when present
- config drift warnings when present

---

## 3. Deferred Tool

### `find_documents_for_concept(entity_id: str, limit: int)`

Allowed, but deferred until mention indexing is enabled and benchmarked against the eval set.

---

## 4. Server Rules

- load ontology once at startup
- hold LanceDB table handle
- open SQLite per request
- keep tool logic thin
- use shared lower-level query/store functions
- configure SQLite connections for concurrent read/write workloads
- document client launch using `stdio` with `pixi run mcp` from the repo root
- do not rely on inherited shell environment for `stdio` clients; required runtime settings must come from `config.yaml`, `config.{profile}.yaml`, or explicit `--config` / `--profile` launch arguments

---

## 5. Validation Rules

Every tool must validate:

- non-empty required strings
- allowed domains
- sane limits

Errors must be explicit and user-facing.

---

## 6. Operational Rule

The MCP server must remain usable even if:

- the store is partially built
- mention indexing is incomplete
- reranker is unavailable

The system should degrade, not collapse.
