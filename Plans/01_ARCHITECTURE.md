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

- scan every file under `Knowledge_Base`
- classify each file as `markdown`, `docling_json`, or `image_png`
- assign durable manifest state to every file
- chunk and embed text-bearing sources
- register PNG assets with provenance links
- commit progress incrementally

File handling rules:

- `markdown` files are converted into a `DoclingDocument` using Docling's supported conversion path, then chunked, embedded, and made searchable
- `docling_json` files are loaded as structured Docling documents, chunked, embedded, and made searchable
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
- every resolved YAML key path must be explicitly classified as `graph_input`, `matcher_input`, `metadata_input`, `audit_only`, or `explicitly_ignored`
- the loader must fail if any resolved YAML key path is unclassified
- non-relational YAML data must be preserved in structured metadata records; no resolved YAML field may be silently dropped
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
- retrieve from searchable chunk rows
- optionally expand with ontology context
- rerank retrieved candidates through the configured `llama.cpp` reranker backend when available
- fall back to dense-only retrieval with an explicit warning when the reranker is unavailable
- optionally synthesize a cited answer

Query scope rule:

- V1 search and answer generation operate on text-bearing chunk sources only
- PNG assets influence provenance and inspection, not core retrieval scoring

### 2.4 MCP Interface

Responsibilities:

- expose thin tools for query, ontology lookup, status, and asset inspection
- open SQLite per request
- avoid embedding business logic in the server layer

## 3. Stores

### 3.1 LanceDB

Used for:

- searchable chunk text
- vector embeddings
- citation labels
- chunk-level provenance

LanceDB holds only text-bearing chunk rows.

### 3.2 SQLite

Used for:

- `corpus_manifest`
- `asset_links`
- `ontology_sources`
- `ontology_snapshot`
- `mentions`
- `ingest_config`
- `ingest_runs`

SQLite is the source of truth for ingest state, recovery, asset registration, and ontology snapshot tracking.

WAL mode is mandatory because ingest writes and MCP reads are expected to overlap.

### 3.3 In-Memory

Used for:

- ontology graph
- entity matcher

These are rebuilt on process start from the resolved ontology snapshot.

## 4. Identity Model

The architecture distinguishes four identities:

- file identity: current `file_path` in the corpus tree
- logical document identity: `document_id` for text-bearing sources
- content identity: file hash
- chunk identity: stable per logical document and chunk text occurrence

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

The minimal working query path is:

1. validate input
2. retrieve dense candidates from searchable chunk rows
3. optionally expand or rerank if enabled
4. synthesize from chunk evidence if requested

Citation rules:

- cite the chunk source's `citation_label`
- markdown `citation_label = source_rel_path`
- Docling `citation_label = source_rel_path#page=<page_no>` when `page_no` is available, otherwise `source_rel_path`
- heading text may be returned separately as display metadata, but not inside canonical `citation_label`
- PNG assets are never direct evidence in V1

## 7. MCP Architecture

The MCP server should:

- load settings once
- load ontology once at startup
- hold the LanceDB table handle
- open SQLite connections per request
- call lower-level query and store modules

The MCP layer should not own ingest, graph, or retrieval policy.
