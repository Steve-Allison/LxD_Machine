# LxD Machine - Data Schema

## 1. Principles

- represent every in-scope corpus file
- separate searchable text payloads from durable ingest state
- make interrupted ingest recoverable
- keep ontology diff tracking separate from ontology snapshot state
- preserve provenance for both text chunks and binary assets
- LanceDB is canonical for vectors; SQLite never duplicates vector bytes

## 1b. Schema Versioning

SQLite schema evolution is tracked by the built-in `PRAGMA user_version`.
The authoritative baseline DDL for every table + index in this schema
lives in `lxd.stores._base_ddl.BASE_SCHEMA_DDL` and is applied on every
`ensure_schema` call (idempotent — every statement uses
`CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`). Numbered
migrations in `lxd.stores.schema` then run in order to lift older stores
up to `CURRENT_SCHEMA_VERSION`. Both are called from
`lxd.app.bootstrap`.

Current schema version: **9**.

| Version | Migration | Purpose |
|---|---|---|
| 1 | `0001_baseline` | Alignment marker — no-op body (the tables it once created are now the baseline DDL applied on every open) |
| 2 | `0002_drop_chunk_vector_json` | Drops `chunk_rows.vector_json`; LanceDB becomes canonical for vectors |
| 3 | `0003_llm_jobs` | Creates the persistent LLM job queue (`llm_jobs` + `idx_llm_jobs_status`) |
| 4 | `0004_repair_ghost_fks` | One-shot cleanup of orphan FK rows left by earlier partial migrations |
| 5 | `0005_ingest_run_telemetry` | Adds `unchanged_files_skipped`, `failed_files`, `embedding_tokens`, `llm_tokens`, `estimated_cost_usd`, `embedding_cache_hits`, `embedding_cache_misses` to `ingest_runs` |
| 6 | `0006_chunk_rows_wiki_metadata` | Adds `cited_sources_json` and `wiki_links_json` on `chunk_rows` |
| 7 | `0007_circuit_breaker_state` | Creates `circuit_breaker_state` for systemic-error circuit breaker persistence |
| 8 | `0008_hierarchical_communities` | Multi-level community support — makes PK composite on `entity_communities` (`entity_id, community_level`) and on `community_reports` (`community_id, community_level`); adds `parent_community_id` and level-scoped indexes |
| 9 | `0009_entity_embedding_state` | Creates `entity_embedding_state` for incremental entity-embedding rebuild bookkeeping |

Every destructive migration writes a `*.pre-migration-vN-to-vM-<timestamp>.sqlite3.bak` backup alongside the DB before altering the schema (see `_run_pending_migration_with_backup` in `lxd.stores.schema`). `ensure_schema` also enforces `PRAGMA foreign_key_check` and required-table/column presence after migrations; a half-migrated DB raises `SchemaIntegrityError` and refuses writes. `pixi run preflight` exposes this check for operators.

There is no `_sqlite_legacy_migrations` module — the pre-numbered-migration cleanups were folded into either the baseline DDL or the numbered migrations, with the sole remaining pre-flight guard being `assert_no_v2_legacy_tables` in `stores/sqlite/connection.py` (asserts no leftover pre-numbered tables exist before opening for writes).

## 2. LanceDB Schema

LanceDB holds three tables under `<paths.data_path>/lancedb/`:

### 2.1 `chunk_vectors` (canonical vector store for chunks)

Fields:

- `chunk_id`: stable chunk identifier
- `document_id`: logical document identifier for the parent text source
- `source_type`: `markdown`, `docling_json`, or `docling_md`
- `source_rel_path`: path relative to the configured corpus root (also FK to `corpus_manifest`)
- `source_filename`: basename of the source file
- `source_domain`: canonical domain slug derived from the first path segment under the corpus root
- `source_hash`: Blake3 of full source file content
- `citation_label`: canonical citation label (`<source_rel_path>#<chunk_index>`)
- `chunk_index`: order within the current chunk list
- `chunk_occurrence`: ordinal for duplicate chunk hashes within the same document
- `token_count`: token count produced by the configured tokenizer
- `text`: chunk text
- `score_hint`: retrieval-hint string
- `metadata_json`: structured provenance metadata (JSON-encoded)
- `cited_sources_json`: JSON list of transitive `**Sources**:` filenames parsed from wiki frontmatter
- `wiki_links_json`: JSON list of `[[slug]]` cross-references parsed from wiki markdown
- `vector`: dense embedding (`float32[embed_dims]`)

Note: `chunk_hash` is a SQLite-only column on `chunk_rows` — not present in the LanceDB row.

Native FTS index on the `text` column via `create_index(config=FTS(with_position=False), name="text_fts_idx", replace=True)` — issued in `refresh_fts_index` and rebuilt on every `open_chunk_table`. The pre-0.25 `create_fts_index` shim is deprecated.

BTree scalar indexes on hot filter columns via `create_index(col, config=BTree(), name=<col>_idx)`, built idempotently on first open: `source_rel_path`, `chunk_id`, `source_domain`. Previously every `where(...)` was an O(N) scan; the indexes turn `delete_source`, `IN (chunk_ids...)` lookups, and domain filters into O(log N).

`metadata_json` carries open-schema Docling chunk metadata; typical fields include `heading_path`, `node_type`, `docling_label`, `page_no`, `bbox`, `charspan`, `content_layer`, `origin_filename`, `linked_asset_paths`.

### 2.2 `entity_embeddings` (per-entity mean-pooled vectors)

Fields:

- `entity_id`: canonical ontology entity ID
- `label`: display label
- `community_id`: nullable community assignment
- `vector`: L2-normalised mean of the top-N embedded chunks that reference the entity (`float32[embed_dims]`)

Written by the `entity_embeddings` phase of `pixi run build-graph`. Read by `search_similar_entities` and by the query pipeline's `_augment_with_embedding_neighbours` (widens the matched-entity set with semantic neighbours in addition to Aho-Corasick surface hits).

Writes go through `merge_insert("entity_id")` — `upsert_entity_embeddings` uses `.when_matched_update_all().when_not_matched_insert_all().execute(records)`; `replace_entity_embeddings` additionally uses `.when_not_matched_by_source_delete()` so the full-replace semantic is one atomic pass. BTree scalar index on `entity_id`.

### 2.3 `embedding_cache` (content-addressed embedding cache)

Fields:

- `cache_key`: `"{chunk_hash}|{embedding_model}|{embedding_dims}"` — content-addressed
- `chunk_hash`: Blake3 of chunk text (denormalised for readability)
- `embedding_model`: model identifier for this batch
- `embedding_dims`: embedding dimensionality
- `vector`: cached embedding (`float32[embedding_dims]`)

Survives full rebuilds because the key is content-addressed: identical text + identical model = identical vector, so re-ingesting the same corpus with the same embedding model re-uses every cached vector. Writes go through `merge_insert("cache_key").when_matched_update_all().when_not_matched_insert_all()` — a single atomic upsert, no delete-then-add split-brain window. BTree scalar index on `cache_key`.

## 3. SQLite Tables

### 3.1 `corpus_manifest`

One row per known corpus file path, including deleted tombstones until the next full rebuild.

Columns:

- `source_rel_path` TEXT PRIMARY KEY — corpus-relative path (portable across machines)
- `absolute_path` TEXT NOT NULL — machine-local absolute path (updated by ingest on each machine)
- `source_type` TEXT NOT NULL
- `source_domain` TEXT NOT NULL
- `document_id` TEXT
- `blake3_hash` TEXT NOT NULL
- `file_size_bytes` INTEGER NOT NULL
- `parent_source_rel_path` TEXT — corpus-relative path to parent source (for PNG assets)
- `lifecycle_status` TEXT NOT NULL
- `retrieval_status` TEXT NOT NULL
- `chunk_count` INTEGER NOT NULL
- `last_seen_at` TEXT NOT NULL
- `last_processed_at` TEXT
- `last_committed_at` TEXT
- `error_message` TEXT

`source_type` values:

- `markdown`
- `docling_json`
- `image_png`

`lifecycle_status` values:

- `pending`
- `processing`
- `complete`
- `failed`
- `deleted`

`retrieval_status` values:

- `searchable`
- `asset_only`
- `not_searchable`

Rules:

- every in-scope corpus file gets a manifest row
- text-bearing files have a `document_id`
- PNG assets set `retrieval_status = 'asset_only'`

### 3.2 `asset_links`

One row per registered PNG asset.

Columns:

- `asset_rel_path` TEXT PRIMARY KEY — corpus-relative path (FK to corpus_manifest)
- `asset_filename` TEXT NOT NULL
- `source_domain` TEXT NOT NULL
- `parent_source_rel_path` TEXT — corpus-relative path to parent text source
- `parent_document_id` TEXT
- `page_no` INTEGER
- `asset_index` INTEGER
- `link_method` TEXT NOT NULL
- `blake3_hash` TEXT NOT NULL
- `last_committed_at` TEXT NOT NULL

This table records binary assets even though they are not searchable in V1.

### 3.3 `ontology_sources`

One row per YAML file that participates in ontology change detection.

Columns:

- `file_rel_path` TEXT PRIMARY KEY — relative path to ontology YAML file
- `blake3_hash` TEXT NOT NULL
- `last_seen_at` TEXT NOT NULL

### 3.4 `ontology_snapshot`

Exactly one current row for the compiled ontology snapshot.

Columns:

- `snapshot_id` TEXT PRIMARY KEY CHECK (`snapshot_id` = 'current')
- `ontology_root` TEXT NOT NULL
- `blake3_hash` TEXT NOT NULL
- `matcher_termset_hash` TEXT NOT NULL
- `matcher_term_count` INTEGER NOT NULL
- `source_file_count` INTEGER NOT NULL
- `entity_file_count` INTEGER NOT NULL
- `entity_count` INTEGER NOT NULL
- `last_loaded_at` TEXT NOT NULL

The snapshot hash must cover the resolved ontology closure, including `!include` fragments.

`matcher_termset_hash` must be the Blake3 hash of the canonical normalized matcher term set:

- one canonical JSON line per normalized matcher term
- fields in fixed key order: `entity_id`, `term_source`, `normalized_term`
- sorted lexicographically by `normalized_term`, then `entity_id`, then `term_source`
- joined with `\\n` and hashed as UTF-8 bytes

### 3.5 `mention_rows`

Non-blocking enrichment table. It may be empty in V1, but when populated it must correspond to the committed `ontology_snapshot`.

Columns:

- `mention_id` TEXT PRIMARY KEY
- `entity_id` TEXT NOT NULL
- `term_source` TEXT NOT NULL — one of `canonical_id`, `alias`, `indicator`
- `source_domain` TEXT NOT NULL
- `source_rel_path` TEXT NOT NULL — FK to `corpus_manifest(source_rel_path)`
- `source_filename` TEXT NOT NULL
- `chunk_id` TEXT NOT NULL — FK to `chunk_rows(chunk_id)` ON DELETE CASCADE
- `surface_form` TEXT NOT NULL
- `start_char` INTEGER NOT NULL
- `end_char` INTEGER NOT NULL

Index: `idx_mention_rows_entity_id(entity_id)`.

### 3.6 `ingest_config`

Key-value snapshot of config values that affect stored data.

Columns:

- `key` TEXT PRIMARY KEY
- `value` TEXT NOT NULL

### 3.7 `llm_jobs`

Persistent LLM job queue. Used by long-running LLM workloads
(OpenAI Batch, background claim/relation extraction) that must survive
process restarts. Status transitions are enforced by a SQLite `CHECK`
constraint; callers choose a stable `job_id` so re-enqueues are
idempotent.

Columns:

- `job_id` TEXT PRIMARY KEY — caller-supplied stable identifier (commonly `blake3(kind + payload + corpus_id)`)
- `kind` TEXT NOT NULL — logical job category (e.g. `claims.openai_batch`)
- `corpus_id` TEXT NOT NULL DEFAULT `'default'` — tenancy marker (`TenancyConfig.corpus_id`)
- `status` TEXT NOT NULL — one of `queued`, `running`, `succeeded`, `failed`, `cancelled`
- `payload_json` TEXT NOT NULL — opaque JSON payload for the executor
- `result_json` TEXT — opaque JSON result once the job succeeds
- `error` TEXT — short human-readable error string on failure
- `attempts` INTEGER NOT NULL DEFAULT 0 — retry counter (monotonic)
- `created_at` TEXT NOT NULL — ISO-8601 UTC
- `updated_at` TEXT NOT NULL — ISO-8601 UTC

Indexes:

- `idx_llm_jobs_status (corpus_id, status, updated_at)` — supports
  multi-tenant scans and "oldest-first" worker pulls.

Access happens exclusively through `lxd.stores.llm_jobs` (`enqueue_job`,
`get_job`, `list_jobs`, `mark_running`, `mark_succeeded`, `mark_failed`,
`mark_cancelled`).

### 3.8 `ingest_runs`

Per-run bookkeeping. One row per `pixi run ingest` invocation.

Columns:

- `run_id` TEXT PRIMARY KEY
- `started_at` TEXT NOT NULL
- `finished_at` TEXT
- `mode` TEXT NOT NULL
- `status` TEXT NOT NULL
- `files_total` INTEGER NOT NULL
- `files_completed` INTEGER NOT NULL
- `searchable_files_rebuilt` INTEGER NOT NULL
- `asset_files_processed` INTEGER NOT NULL
- `unchanged_files_skipped` INTEGER NOT NULL
- `failed_files` INTEGER NOT NULL
- `chunks_written` INTEGER NOT NULL
- `embedding_tokens` INTEGER NOT NULL
- `llm_tokens` INTEGER NOT NULL
- `estimated_cost_usd` REAL NOT NULL
- `embedding_cache_hits` INTEGER NOT NULL
- `embedding_cache_misses` INTEGER NOT NULL
- `notes` TEXT NOT NULL

### 3.9 Knowledge-graph tables

The knowledge-graph build populates a further family of tables (all committed by `pixi run build-graph`). See `08_KNOWLEDGE_GRAPH_SPEC.md` for full column-level detail; the tables are:

- `extracted_relations` — raw per-chunk (subject, predicate, object) tuples with confidence and extraction model, keyed by `relation_id`.
- `relations` — canonical (subject, predicate, object) tuples aggregated from `extracted_relations`, with `support_count`, `avg/min/max_confidence`, and a UNIQUE `(subject_entity_id, predicate, object_entity_id)` index. PK is `relation_id`.
- `relation_evidence` — one row per `(relation_id, chunk_id)` witness with surface forms and evidence text. Cascades on `relations` and `chunk_rows`. PK is `evidence_id = blake3(relation_id + chunk_id)`.
- `claims` — LLM-extracted factual assertions per chunk with `claim_type` (`assertion`, `definition`, `comparison`, `causal`, `procedural`), confidence, subject/object entity IDs. PK is `claim_id`. Indexed on `subject_entity_id`, `object_entity_id`, `chunk_id`, `document_id`.
- `entity_profiles` — per-entity summary + 6 centrality metrics (`pagerank`, `betweenness`, `closeness`, `in_degree`, `out_degree`, `eigenvector`) + community assignment + deterministic and optional LLM summary + JSON blobs (`aliases`, `top_predicates`, `top_claims`) + `source_hash` (composed from rank positions and chunk/claim IDs; see 08_KG spec). PK is `entity_id`.
- `entity_communities` — entity-to-community assignments; PK is composite `(entity_id, community_level)` supporting multi-level (hierarchical) communities. Indexed on `(community_id, community_level)` and `(community_level)`.
- `community_reports` — deterministic and optional LLM summaries per community; composite PK `(community_id, community_level)`; `parent_community_id` supports hierarchy. Indexed on `(community_level)` and `(parent_community_id, community_level)`.
- `graph_build_state` — one row per `build-graph` run tracking phase progression (`current_phase`), counters, and graph version.
- `graph_metadata` — key-value store for durable KG metadata (`graph_version`, `last_build_at`, `community_algorithm`).
- `circuit_breaker_state` — one row per scope tracking consecutive-failure counts, last error class/message/type, tripped-at timestamp; used by the systemic-error circuit breaker in `ingest/error_classification.py`.
- `entity_embedding_state` — one row per entity recording `source_hash`, `chunk_count`, `embedding_model`, `embedding_dims`, `updated_at`; enables incremental entity-embedding rebuilds by comparing hashes.

## 4. Identity Rules

- file identity: corpus-relative path (`source_rel_path` — portable across machines)
- logical text-source identity: `document_id`
- content identity: Blake3 of full file content
- chunk content identity: Blake3 of chunk text
- chunk identity: Blake3 of `utf8(document_id) + 0x00 + utf8(chunk_hash) + 0x00 + utf8(chunk_occurrence)`
- mention identity: Blake3 of `entity_id + chunk_id + start_char`
- asset link identity: corpus-relative `asset_rel_path`

Rules:

- all PKs and FKs use corpus-relative paths, making the `data/` folder portable between machines
- `absolute_path` is stored in `corpus_manifest` for local file I/O but is not a PK or FK; it is refreshed by `pixi run ingest` on each machine
- changing file content does not change `document_id` for the same path
- moving or renaming a text source without content change transfers the existing `document_id`
- unchanged chunks in the same logical document keep the same `chunk_id`
- duplicate live files with the same content remain distinct documents

## 5. Domain Derivation Rule

- derive `source_domain` from the first segment of the file path relative to the configured corpus root
- normalize to lowercase snake_case
- if a file lives directly under the corpus root, use `root`
- store the original human-readable label in metadata when helpful

## 6. Key Design Rule

The manifest state is the truth for ingest completeness.

If vectors or asset metadata are written without the corresponding committed SQLite state, the ingest is not complete.

## 7. Connection PRAGMAs

Every SQLite connection opened through `lxd.stores.sqlite.connection.connect_sqlite` applies (and verifies) these pragmas at connect time:

- `journal_mode=WAL`
- `synchronous=NORMAL`
- `foreign_keys=ON`
- `busy_timeout=5000`
- `temp_store=MEMORY`
- `cache_size=-65536`  (≈ 64 MiB page cache per connection)

These settings are mandatory for concurrent ingest/MCP workloads; tests and CLI commands must go through `connect_sqlite` (or the MCP request path's per-thread pool in `lxd.stores.sqlite._pool.pooled_connection`) rather than calling `sqlite3.connect` directly. There is no `PRAGMA optimize` on close — the connection helpers install no close-time hook.

## 8. Config Lock

`lxd.app.bootstrap.compute_config_digest` produces a stable Blake3 digest
of the resolved `RuntimeConfig` JSON dump. At startup, `reconcile_config_lock`
writes `<paths.data_path>/config.lock` on first run and emits a
`config.lock.mismatch` warning (without overwriting) when the stored
digest disagrees with the current config. The lock file is a plain text
file containing the hex digest; it is safe to delete to reseed.
