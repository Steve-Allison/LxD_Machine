# LxD Machine - Data Schema

## 1. Principles

- represent every in-scope corpus file
- separate searchable text payloads from durable ingest state
- make interrupted ingest recoverable
- keep ontology diff tracking separate from ontology snapshot state
- preserve provenance for both text chunks and binary assets

## 2. LanceDB Schema

Table: `chunks`

Fields:

- `chunk_id`: stable chunk identifier
- `document_id`: logical document identifier for the parent text source
- `source_type`: `markdown` or `docling_json`
- `source_path`: absolute source file path
- `source_rel_path`: path relative to the configured corpus root
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

- `file_path` TEXT PRIMARY KEY
- `file_rel_path` TEXT NOT NULL
- `source_type` TEXT NOT NULL
- `source_domain` TEXT NOT NULL
- `document_id` TEXT
- `blake3_hash` TEXT NOT NULL
- `file_size_bytes` INTEGER NOT NULL
- `parent_source_path` TEXT
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

- `asset_path` TEXT PRIMARY KEY
- `asset_rel_path` TEXT NOT NULL
- `asset_filename` TEXT NOT NULL
- `source_domain` TEXT NOT NULL
- `parent_source_path` TEXT
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

- `file_path` TEXT PRIMARY KEY
- `file_rel_path` TEXT NOT NULL
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
- `source_path` TEXT NOT NULL
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

### 3.7 `ingest_runs`

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

- file identity: current corpus path
- logical text-source identity: `document_id`
- content identity: Blake3 of full file content
- chunk content identity: Blake3 of chunk text
- chunk identity: Blake3 of `utf8(document_id) + 0x00 + utf8(chunk_hash) + 0x00 + utf8(chunk_occurrence)`
- mention identity: Blake3 of `entity_id + chunk_id + start_char`
- asset link identity: current `asset_path`

Rules:

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
