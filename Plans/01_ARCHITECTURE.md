# LxD Machine - Architecture

## 1. Architectural Principle

The architecture must match the real repository:

- mixed-format corpus, not markdown-only
- long-running embedding builds on local hardware
- ontology data distributed across the full `Yamls` tree

The system therefore optimizes for:

- complete corpus coverage
- incremental committed progress
- resumability
- low operational complexity
- explicit provenance

Runtime behavior must be configuration-driven.

Implementation structure and module-boundary rules are defined in `01b_CODEBASE_STRUCTURE.md`.

Libraries may be fixed by architecture, but file coverage, chunking parameters, tokenization, model choice, batch sizes, and limits must come from config.

## 2. Top-Level Shape

The system has four parts:

1. corpus inventory and ingest
2. ontology load
3. query pipeline
4. MCP interface

### 2.1 Corpus Inventory And Ingest

Responsibilities:

- scan every file under the configured `paths.corpus_path` (default: the curated wiki at `~/AI_Projects+Code/knowledge/wiki/`)
- classify each file as `markdown`, `docling_md`, `docling_json`, or `image_png`
- assign durable manifest state to every file
- chunk and embed text-bearing sources
- register PNG assets with provenance links
- commit progress incrementally

File handling rules:

- `markdown` (`.md`) files are loaded via `ingest/markdown.py`'s frontmatter-aware Markdown reader (Docling's `DocumentConverter` is not on the markdown path — the reader parses the `**Sources**:` and `[[slug]]` conventions and streams the body into the hybrid chunker directly)
- `docling_md` (`.docling.md`) files use the same markdown reader
- `docling_json` (`.docling.json`) files are loaded as structured Docling documents and chunked with the Docling `HybridChunker`
- `image_png` files are registered as corpus assets; they are not embedded in V1, but they remain durable first-class corpus members

### 2.2 Ontology Load

Responsibilities:

- load ontology YAML from the full `Yamls` tree
- resolve `!include` references
- inventory every resolved YAML key path and classify it
- build an in-memory `networkx.MultiDiGraph`
- build an Aho-Corasick matcher from entity definitions

Ontology shape rules:

- ontology inputs are every YAML file matched by `settings.ontology.include_globs` under `config.paths.ontology_path`
- every resolved YAML key path is classified as one of three categories by `ontology/inventory.classify_key_path`: `graph_input`, `matcher_input`, or `metadata_input`
- unrecognised paths default to `metadata_input` — the classifier never fails a load on an unknown key; the default preserves structural data, at the cost of accepting new fields silently until they are explicitly promoted to `graph_input` or `matcher_input`
- non-relational YAML data is preserved in structured metadata records via the `metadata_input` default; no resolved YAML field is silently dropped
- graph nodes include entity nodes, ontology file nodes, taxonomy-derived nodes, and explicit unresolved-reference nodes when source data cannot be resolved to a known target
- graph edges come from file-level `_meta.relationships`, per-entity `relates_to`, per-entity `parent_entity`, `taxonomy_mapping`, `maps_to_taxonomy_types`, and `taxonomy_reference` / `validate_against_taxonomy`
- relation definitions in `file_relationships`, `entity_relations`, and `entity_relation_weights` must be consumed as validation schema for loaded graph edges
- ontology change detection must cover every YAML file that participates in the resolved ontology snapshot, not just `*_entities.yaml`
- entity node identifiers are canonical entity IDs; non-entity nodes use typed stable IDs such as `file:{rel_path}` and `taxonomy_value:{taxonomy}:{dimension}:{value}`
- graph edge keys are deterministic and stable: `blake3(origin_kind + 0x00 + source_file_rel_path + 0x00 + source_node_id + 0x00 + relation_type + 0x00 + target_node_id)`
- every edge stores at least `relation_type`, `origin_kind`, `origin_path`, `source_file_rel_path`, `source_node_id`, `source_node_type`, `source_entity_id`, `target_node_id`, `target_node_type`, `target_entity_id`, and structured relation metadata
- unresolved relation targets must be preserved as explicit graph nodes plus validation issues; they must not be silently discarded

### 2.3 Query Pipeline

Responsibilities:

- validate input
- route the query via the adaptive router (`retrieval/router.py`) — decide `retrieve?` and `breadth` (`narrow` | `standard` | `broad`); short-circuit meta / out-of-scope questions with `no_retrieval_needed`
- expand the question with ontology entities (Aho-Corasick) plus entity-embedding nearest neighbours (mandatory feature, degrades gracefully when the graph isn't built)
- retrieve via LanceDB native hybrid search (`Table.search(query_type="hybrid").rerank(RRFReranker())`) — engine issues one query, runs dense k-NN and BM25 FTS in parallel, fuses via RRF; the previous two-query split + Python-side lexical-lane fuse is superseded
- rerank retrieved candidates through the configured backend (`llama_cpp` cross-encoder over HTTP or in-process ColBERT); when the reranker is unavailable the pipeline surfaces a live warning via `ctx.warning` AND records it in the buffered `warnings` list, then continues with the hybrid ranking
- fuse rerank, relation-membership, and centrality lanes on top of the hybrid ranking via source-aware RRF; optionally community-diversify
- build graph context (entity profiles + community reports + claims) additively for the synthesis prompt
- synthesise a cited answer either against the server's Ollama model or, when `mcp.synthesis_backend=client_sampling`, via MCP `ctx.sample` (with server-LLM fallback on `SamplerFailure`)

Query scope rule:

- V1 search and answer generation operate on text-bearing chunk sources only
- PNG assets influence provenance and inspection, not core retrieval scoring

### 2.4 MCP Interface

Responsibilities:

- expose 20 read-only tools spanning corpus operations, ontology lookup, knowledge-graph operations, status, and the full answer pipeline
- expose 3 URI-templated resources (`lxd://corpus/{path*}`, `lxd://entity/{entity_id}`, `lxd://community/{entity_id}`) and 2 prompts (`lxd_synthesis_preamble`, `lxd_query_refinement`)
- keep tool bodies thin: server-layer tools call lower-level query and store modules and add no business logic
- use the per-thread SQLite connection pool (`stores/sqlite/_pool.pooled_connection`) instead of opening a fresh connection per request — the pool is initialised once per worker thread with the required PRAGMAs and reused for every subsequent tool call on that thread

MCP runtime rules:

- every registered tool is an `async def`; synchronous tool bodies run inside worker threads via `lxd.mcp.async_runtime.run_tool`, which enforces a per-tool hard timeout (`mcp.tool_timeout_secs`)
- the lifespan bundle (`_LxDLifespan`) owns exactly two things: the `AppContext` (config, resolved paths, digests) and the `IngestPlan` (ontology graph, matcher, plan metadata). LanceDB table handles and HTTP client factories are NOT held on the lifespan; the two long-running LLM tools (`search_knowledge`, `search_knowledge_deep`) additionally take a phased-progress callback, a `Context.warning` streaming-notice callback, and (when `mcp.synthesis_backend=client_sampling`) an `anyio.from_thread.run`-bridged sampler that dispatches synthesis to `ctx.sample`

## 3. Stores

### 3.1 LanceDB

Three tables under `<paths.data_path>/lancedb/`:

- **`chunk_vectors`** — searchable chunk text, dense embeddings, citation labels, chunk-level provenance including transitive `cited_sources_json` (wiki `**Sources**:`) and `wiki_links_json` (`[[slug]]` cross-references). Native FTS index via `create_index(config=FTS(with_position=False))`. BTree scalar indexes on `source_rel_path`, `chunk_id`, `source_domain` for O(log N) filter lookups on the hot ingest/retrieval paths.
- **`entity_embeddings`** — per-entity L2-normalised mean-pooled vectors written by the `build-graph` entity-embedding phase; used by `search_similar_entities` and by the query pipeline's `_augment_with_embedding_neighbours` (widens matched-entity set with semantic neighbours). BTree scalar index on `entity_id`.
- **`embedding_cache`** — content-addressed cache keyed on `"{chunk_hash}|{embedding_model}|{embedding_dims}"`. Survives full rebuilds because identical text + identical model = identical vector. BTree scalar index on `cache_key`.

LanceDB is the single source of truth for vectors. The legacy `chunk_rows.vector_json` column in SQLite was dropped by schema migration `0002_drop_chunk_vector_json`; all vector reads must go through the LanceDB helpers in `lxd.stores.lancedb`.

Writes to `entity_embeddings` and `embedding_cache` go through `merge_insert(...).when_matched_update_all().when_not_matched_insert_all()` — a single atomic upsert per key, no delete-then-add split-brain window. `replace_entity_embeddings` additionally uses `.when_not_matched_by_source_delete()` for full-replace semantics.

All LanceDB filter expressions must be constructed through the helpers in `lxd.stores.lance_sql` (`eq_clause`, `in_clause`, `escape_string_literal`), which reject NUL/newline characters and enforce SQL-identifier column names. LanceDB's Python API does not support parameter binding — string-composition is unavoidable and the helpers are the safe boundary.

### 3.2 SQLite

Used for:

- `corpus_manifest` — document metadata and content hashes
- `chunk_rows` — chunked text (embeddings live in LanceDB) with provenance
- `mention_rows` — entity mentions detected per chunk
- `extracted_relations` — LLM-extracted relations per chunk
- `asset_links` — PNG/asset registration and parent inference
- `ontology_sources` — ontology file tracking
- `ontology_snapshot` — ontology state hash for drift detection
- `ingest_config` — persisted ingest config snapshot
- `ingest_runs` — per-run bookkeeping (files, chunks, tokens, cost, cache hit-rate)
- `relations` — canonical deduplicated relations (knowledge graph)
- `relation_evidence` — provenance linking canonical relations to source chunks (knowledge graph)
- `claims` — LLM-extracted factual claims per chunk (knowledge graph)
- `entity_profiles` — deterministic entity summaries with centrality (knowledge graph)
- `entity_communities` — community assignments per entity, composite PK `(entity_id, community_level)` supporting multi-level (hierarchical) communities
- `community_reports` — deterministic community summaries, composite PK `(community_id, community_level)`, `parent_community_id` for hierarchy
- `graph_metadata` — knowledge graph version and build timestamps
- `graph_build_state` — resumable knowledge graph build state machine
- `circuit_breaker_state` — systemic-error circuit-breaker persistence per scope
- `entity_embedding_state` — per-entity source_hash + embedding_model bookkeeping for incremental entity-embedding rebuilds
- `llm_jobs` — persistent LLM job queue (status, payload, result, attempts)

SQLite is the source of truth for ingest state, recovery, asset registration, ontology snapshot tracking, the full knowledge graph, and persistent LLM job state.

Schema evolution is tracked by SQLite's built-in `PRAGMA user_version` and runs at startup from `lxd.app.bootstrap`. The authoritative baseline DDL lives in `lxd.stores._base_ddl.BASE_SCHEMA_DDL` and is applied on every `ensure_schema` call (idempotent — every statement uses `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`). Numbered migrations in `lxd.stores.schema` then lift older stores up to `CURRENT_SCHEMA_VERSION = 9`. Every destructive migration writes a `*.pre-migration-vN-to-vM-*.sqlite3.bak` backup before altering DDL. `ensure_schema` then verifies `foreign_key_check` and required tables/columns; a half-migrated DB raises `SchemaIntegrityError` and refuses writes. There is no `_sqlite_legacy_migrations` module — the pre-flight guard is `stores/sqlite/connection.assert_no_v2_legacy_tables` (asserts no leftover pre-numbered tables). See `02_DATA_SCHEMA.md §1b` for the full migration list (0001 through 0009).

WAL mode is mandatory because ingest writes and MCP reads are expected to overlap.

Connection PRAGMAs applied at every connect through `stores/sqlite/connection.connect_sqlite`: `journal_mode=WAL`, `synchronous=NORMAL`, `foreign_keys=ON`, `busy_timeout=5000`, `temp_store=MEMORY`, `cache_size=-65536`. No close-time PRAGMA hook is installed.

### 3.3 In-Memory

Used for:

- ontology graph
- entity matcher

These are rebuilt on process start from the resolved ontology snapshot.

## 4. Identity Model

The architecture distinguishes four identities:

- file identity: corpus-relative path (`source_rel_path`) — portable across machines
- logical document identity: `document_id` for text-bearing sources
- content identity: file hash
- chunk identity: stable per logical document and chunk text occurrence

All PKs and FKs in SQLite use corpus-relative paths, making the `data/` folder portable between machines. The `absolute_path` column in `corpus_manifest` stores the machine-local path for file I/O and is refreshed by `pixi run ingest` on each machine.

`document_id` rules:

- each searchable text source has one `document_id`
- when a file stays at the same path, its `document_id` persists across content edits
- when a file moves or renames without content change, move detection transfers the existing `document_id`
- duplicate live files with identical content still keep distinct `document_id` values

`chunk_id` rule:

- `chunk_id = blake3(utf8(document_id) + 0x00 + utf8(chunk_hash) + 0x00 + utf8(chunk_occurrence))`

This keeps unchanged chunks stable within the same logical document while allowing move-safe reuse.

## 5. Ingest Durability Model

The ingest must commit file state incrementally.

For each text-bearing source:

1. mark manifest row `processing`
2. parse and chunk deterministically
3. compare the new chunk set against the committed chunk set for the same `document_id`
4. delete stale chunk rows
5. embed only new or changed chunks
6. write replacement chunk rows
7. verify the committed chunk set
8. mark manifest row `complete`

For each PNG asset:

1. mark manifest row `processing`
2. extract stable metadata and parent linkage if available
3. upsert `asset_links`
4. mark manifest row `complete`

If the process dies halfway through:

- previously completed rows remain complete
- rows left in `processing` are visible and recoverable
- restart reconciliation reprocesses incomplete rows

LanceDB writes are not the commit boundary.

SQLite committed state is the commit boundary.

## 6. Query Architecture

The working query path (single-source-of-truth: `retrieval/query_pipeline.py`) is:

1. validate input (`_validate_question`, `_validate_domain`, `_validate_limit`)
2. route via the adaptive router — decide `retrieve?` and `breadth`; short-circuit meta questions with `no_retrieval_needed`
3. expand: Aho-Corasick over the question + entity-embedding nearest neighbours (mandatory feature; degrades gracefully)
4. embed the (possibly HyDE-rewritten) question
5. retrieve via LanceDB native hybrid (`Table.search(query_type="hybrid").rerank(RRFReranker())`) — dense k-NN + BM25 FTS fused inside the engine
6. attach centrality signals from `entity_profiles` per chunk
7. diversify to one representative chunk per `source_rel_path`; rerank the representative prefix
8. fuse the hybrid + rerank + relation-membership + centrality lanes via RRF; optionally community-diversify
9. build additive graph context (entity profiles + community reports + claims) for the synthesis prompt
10. synthesise a cited answer either via the server's Ollama model or, when `mcp.synthesis_backend=client_sampling`, via `ctx.sample` (with server-LLM fallback on `SamplerFailure`)

Citation rules:

- cite the chunk source's `citation_label`
- markdown `citation_label = <source_rel_path>#<chunk_index>`
- Docling `citation_label = <source_rel_path>#<chunk_index>` (page numbers are carried in `metadata_json.page_no`, not in the citation label)
- heading text may be returned separately as display metadata, but not inside canonical `citation_label`
- PNG assets are never direct evidence in V1

## 7. MCP Architecture

The MCP server:

- loads settings once and computes a Blake3 `config_digest`; reconciles against `<data_path>/config.lock` (seeds on first run, warns on drift without overwriting)
- loads the ontology + matcher once at lifespan startup, holds them on `_LxDLifespan.ingest_plan`
- opens SQLite connections through a per-thread pool (`stores/sqlite/_pool.pooled_connection`) rather than per-request — the pool is initialised once per worker thread with the required PRAGMAs and reused across every subsequent tool call on that thread
- opens LanceDB tables per request inside tool bodies via `connect_lancedb` + `open_chunk_table` / `open_entity_table` — the table handles are cheap to open and are NOT held on the lifespan bundle
- calls lower-level query and store modules; tool bodies remain thin

The MCP layer should not own ingest, graph, or retrieval policy.

All tools are `async def` and wrapped through `lxd.mcp.async_runtime.run_tool`,
which runs synchronous bodies in a worker thread, applies a per-call timeout
sourced from `mcp.tool_timeout_secs`, and logs `mcp.tool.timeout` /
`mcp.tool.error` events on failure. This keeps the FastMCP event loop
responsive under concurrent client load.

## 8. Cross-Cutting Infrastructure

- **Multi-tenancy hook.** `RuntimeConfig.tenancy.corpus_id` (default
  `"default"`) marks every persisted job and is available to future
  schema migrations. `corpus_id` is validated to match
  `^[a-z0-9][a-z0-9_-]{0,62}$`.
- **Observability.** Structured logging via `structlog` with UTC
  timestamps, `contextvars`-propagated run IDs, a `log_duration` context
  manager for stage timing, and a `scrub_secrets` processor that redacts
  keys containing `api_key`, `token`, `authorization`, `password`, etc.
  Metrics-exporter integrations (OpenTelemetry, Prometheus) are not
  currently wired — logging is the sole observability surface.
- **Persistent LLM jobs.** Long-running LLM workloads (OpenAI Batch,
  background claim/relation extraction) are queued in `llm_jobs` via the
  idempotent helpers in `lxd.stores.llm_jobs`. Each job carries a stable
  caller-chosen `job_id`, an opaque JSON payload, and a
  `queued → running → succeeded|failed|cancelled` lifecycle.
- **Ontology validation.** `lxd.ontology.schema_models.OntologyFileModel`
  (Pydantic v2, `extra="allow"`) offers opt-in structural validation of
  ontology YAML files. The existing hand-rolled loader remains the source
  of truth; the model is a complementary early-warning layer.
