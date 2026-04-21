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
Numbered migrations live in `lxd.stores.schema` and run in order at every
startup via `ensure_schema` (called from `lxd.app.bootstrap`):

| Version | Migration | Purpose |
|---|---|---|
| 1 | `0001_baseline` | Creates all primary ingest and KG tables |
| 2 | `0002_drop_chunk_vector_json` | Drops `chunk_rows.vector_json`; LanceDB becomes canonical for vectors |
| 3 | `0003_llm_jobs` | Creates the persistent LLM job queue (`llm_jobs` + `idx_llm_jobs_status`) |

Legacy pre-versioning upgrades (rename of keys, PK migrations to
corpus-relative paths, etc.) live in
`lxd.stores._sqlite_legacy_migrations` and always run **before** the
numbered migrations so that older stores upgrade cleanly.

## 2. LanceDB Schema

Table: `chunks`

Fields:

- `chunk_id`: stable chunk identifier
- `document_id`: logical document identifier for the parent text source
- `source_type`: `markdown` or `docling_json`
- `source_rel_path`: path relative to the configured corpus root (used as FK to corpus_manifest)
- `source_filename`: basename of the source file
- `source_domain`: canonical domain slug derived from the first path segment under the corpus root
- `source_hash`: Blake3 of full source file content
- `citation_label`: canonical citation label using `source_rel_path` or `source_rel_path#page=<page_no>`
- `chunk_index`: order within the current chunk list
- `chunk_occurrence`: ordinal for duplicate chunk hashes within the same document
- `chunk_hash`: Blake3 of chunk text
- `text`: chunk text
- `token_count`: token count produced by the configured tokenizer
- `metadata_json`: structured provenance metadata
- `vector`: dense embedding

`metadata_json` should support fields such as:

- `heading_path`
- `node_type`
- `docling_label`
- `page_no`
- `bbox`
- `charspan`
- `content_layer`
- `origin_filename`
- `linked_asset_paths`

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

### 3.5 `mentions`

Non-blocking enrichment table. It may be empty in V1, but when populated it must correspond to the committed `ontology_snapshot`.

Columns:

- `mention_id` TEXT PRIMARY KEY
- `entity_id` TEXT NOT NULL
- `source_domain` TEXT NOT NULL
- `source_rel_path` TEXT NOT NULL — FK to corpus_manifest(source_rel_path)
- `source_filename` TEXT NOT NULL
- `chunk_id` TEXT NOT NULL
- `surface_form` TEXT NOT NULL
- `start_char` INTEGER NOT NULL
- `end_char` INTEGER NOT NULL

### 3.6 `ingest_config`

Key-value snapshot of config values that affect stored data.

Columns:

- `key` TEXT PRIMARY KEY
- `value` TEXT NOT NULL

### 3.7 `llm_jobs`

Persistent LLM job queue (Wave 11). Used by long-running LLM workloads
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

Recommended run bookkeeping.

Columns:

- `run_id` TEXT PRIMARY KEY
- `started_at` TEXT NOT NULL
- `finished_at` TEXT
- `mode` TEXT NOT NULL
- `status` TEXT NOT NULL
- `files_total` INTEGER NOT NULL
- `files_completed` INTEGER NOT NULL
- `searchable_files_completed` INTEGER NOT NULL
- `asset_files_completed` INTEGER NOT NULL
- `chunks_written` INTEGER NOT NULL
- `notes` TEXT

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

Every SQLite connection opened through `lxd.stores.connection` applies
(and verifies) these pragmas:

- `journal_mode=WAL`
- `synchronous=NORMAL`
- `foreign_keys=ON`
- `busy_timeout=5000`
- `temp_store=MEMORY`
- `cache_size=-65536`  (≈ 64 MiB page cache per connection)

On close, `PRAGMA optimize` is issued to keep query planner statistics
fresh. These settings are mandatory for concurrent ingest/MCP workloads;
tests and CLI commands must go through the shared connection helpers
rather than calling `sqlite3.connect` directly.

## 8. Config Lock

`lxd.app.bootstrap.compute_config_digest` produces a stable Blake3 digest
of the resolved `RuntimeConfig` JSON dump. At startup, `reconcile_config_lock`
writes `<paths.data_path>/config.lock` on first run and emits a
`config.lock.mismatch` warning (without overwriting) when the stored
digest disagrees with the current config. The lock file is a plain text
file containing the hex digest; it is safe to delete to reseed.
