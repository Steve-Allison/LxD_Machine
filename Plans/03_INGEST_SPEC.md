# LxD Machine - Ingest Specification

## 1. Purpose

This document specifies what happens when ingest runs against the full corpus root.

It covers:

- full corpus scan across all in-scope file types
- ontology diff and load
- text chunking and embedding
- PNG asset registration
- non-blocking mention indexing
- incremental recovery rules

## 2. Entry Points

```bash
pixi run ingest
pixi run ingest --full
pixi run status
```

`ingest` processes changes.

`ingest --full` rebuilds corpus and ontology state from scratch.

`status` is a read-only operational view over committed store state.

`status` should prefer the committed SQLite/LanceDB state and fall back to live ontology or snapshot inspection only when the store is absent, partial, or written by an older schema that does not contain the required status fields.

## 3. Ingest Phases

The pipeline is a **sequential per-source orchestrator**, not eleven discrete cross-corpus phases. The logical shape below reads top-to-bottom, but the actual per-file work (phases 6-8 and 10) is interleaved in one loop in `ingest/pipeline/orchestrator.run_ingest` — each source is scanned, chunked, embedded, mentioned, and persisted end-to-end before the loop advances to the next source. This is why per-file failure isolation works (§15).

Cross-corpus phases (run once per invocation):

1. validate config and dependencies (`_validate_ingest_dependencies`)
2. initialize stores (`ensure_schema` + LanceDB `open_*_table` idempotent DDL / index creation)
3. scan and diff ontology sources
4. load ontology, build graph, build matcher (via `build_or_load_automaton` with pickled cache)
5. scan and classify corpus files (`ingest/scanner.scan_corpus`)
6. diff corpus files and detect moves/renames (`ingest/pipeline/moves`)

Per-source phases (run in one interleaved loop over each source):

- 7-8. process the source (chunk → embed with cache → detect mentions inline → persist LanceDB-first-then-SQLite → mark `retrieval_status`)

Cross-corpus phases (run once per invocation, after the per-source loop):

- 9\. write ontology source rows and `ontology_snapshot`
- 1. mention indexing is done inline in phase 7; no separate rebuild phase
- 1. write ingest config snapshot and report

Each phase surfaces its own failures loudly; per-file failures inside the per-source loop are isolated to the individual `source_rel_path` (§15).

## 4. Phase 1 - Validate Config And Dependencies

Before mutating any store:

- validate settings
- verify `config.paths.corpus_path` and `config.paths.ontology_path`
- probe the configured embedder — `probe_embedder(config)` issues one embedding request against whichever backend `models.embed_backend` names (`ollama` or `openai`). When `embed_backend=openai`, the Ollama endpoint is not probed at all — its availability isn't a precondition for an OpenAI-embed run
- open SQLite with WAL-capable settings

Model readiness probe contract:

- embedder: one embedding request on a fixed probe string with `truncate=false`
- ingest startup must fail if the configured embedder cannot complete its probe within the configured timeout

If the configured embedder cannot complete its probe, ingest halts before any store mutation via a `RuntimeError` from `_validate_ingest_dependencies`. The halt is not backend-selective: even asset-only or ontology-only work is aborted because the current shape treats embedder readiness as a run-level precondition.

## 5. Phase 2 - Initialize Stores

Create or open:

- LanceDB `chunk_vectors` (canonical vector store for chunks; native FTS index on `text`; BTree scalar indexes on `source_rel_path`, `chunk_id`, `source_domain` created idempotently by `open_chunk_table`)
- LanceDB `entity_embeddings` (per-entity mean-pooled vectors for KG similarity search; BTree scalar index on `entity_id`)
- LanceDB `embedding_cache` (content-addressed embedding cache keyed on `chunk_hash|model|dims`; BTree scalar index on `cache_key`)
- SQLite baseline tables via `ensure_schema`: the DDL for every table + index lives in `lxd.stores._base_ddl.BASE_SCHEMA_DDL` and is re-applied on every call (idempotent `CREATE ... IF NOT EXISTS`). Numbered migrations in `lxd.stores.schema` then lift older stores up to `CURRENT_SCHEMA_VERSION = 9`. Tables include `corpus_manifest`, `chunk_rows`, `asset_links`, `mention_rows` (not `mentions`), `ontology_sources`, `ontology_snapshot`, `ingest_config`, `ingest_runs`, the full KG family (`claims`, `entity_profiles`, `entity_communities`, `community_reports`, `relations`, `relation_evidence`, `extracted_relations`, `graph_build_state`, `graph_metadata`, `circuit_breaker_state`, `entity_embedding_state`), and `llm_jobs`.

Store initialization rules:

- there is no `_migrate_legacy_schema` module — the pre-numbered-migration guard is `stores/sqlite/connection.assert_no_v2_legacy_tables`, which asserts no leftover pre-numbered tables exist before opening for writes
- `ensure_schema` is idempotent; repeated calls re-apply the baseline DDL (all `CREATE IF NOT EXISTS`) and no-op the migrations once `user_version` matches `CURRENT_SCHEMA_VERSION`
- every destructive migration writes a `*.pre-migration-vN-to-vM-*.sqlite3.bak` backup before altering DDL, and `ensure_schema` verifies `foreign_key_check` and required tables/columns afterwards; a half-migrated DB raises `SchemaIntegrityError` and refuses writes
- SQLite connections open through `lxd.stores.sqlite.connection.connect_sqlite` (or the MCP request path's per-thread pool at `lxd.stores.sqlite._pool.pooled_connection`), which applies and verifies the mandatory PRAGMAs at connect time:
  - `PRAGMA journal_mode=WAL;`
  - `PRAGMA synchronous=NORMAL;`
  - `PRAGMA foreign_keys=ON;`
  - `PRAGMA busy_timeout=5000;`
  - `PRAGMA temp_store=MEMORY;`
  - `PRAGMA cache_size=-65536;`
- no close-time PRAGMA hook is installed

## 6. Phase 3 - Scan And Diff Ontology Sources

Scan the full ontology root using `settings.ontology.include_globs`.

In-scope files are YAML files only.

Ignored examples:

- `README.md`
- progress markdown files
- `.DS_Store`

Ontology sources are persisted via `replace_ontology_sources` — a full-replace, not a change-tracked upsert. The pipeline does not derive an `ontology_changed` flag; every ingest reloads the ontology from the YAML tree, rebuilds the matcher (via `build_or_load_automaton` with a pickled cache keyed on the matcher term-set hash), and rewrites the `ontology_sources` table wholesale. The `ontology_snapshot` row is then upserted (`snapshot_id='current'`) with the current `matcher_termset_hash` and file counts.

## 7. Phase 4 - Load Ontology, Build Graph, Build Matcher

This phase always runs.

The loader must:

- parse every in-scope YAML source
- resolve `!include` references
- build the ontology `networkx.MultiDiGraph`
- count entity files and entity types
- compute a resolved snapshot hash
- compute the canonical normalized matcher term set, `matcher_term_count`, and `matcher_termset_hash`

The matcher must then be built from the resolved entity definitions.

The ontology snapshot state is not per-file.

Per-file diff tracking lives in `ontology_sources`.

Compiled state lives in `ontology_snapshot`.

## 8. Phase 5 - Scan And Classify Corpus Files

Scan every file under `config.paths.corpus_path` except ignored filesystem noise.

Classification rules (per `ingest/scanner.classify_source_type`):

- `.md` -> `markdown`
- `.docling.md` -> `docling_md`
- `.docling.json` -> `docling_json`
- `.png` -> `image_png`

Other file types are out of scope unless added to config later.

For every in-scope file, collect:

- absolute path (used for local file I/O only, not stored as PK)
- relative path (corpus-relative — used as the portable identity in all stores)
- source type
- file size
- content hash
- source domain

## 9. Phase 6 - Diff Corpus Files And Detect Moves/Renames

Diff against `corpus_manifest`.

For text-bearing sources:

- same path + changed hash -> modified
- exactly one new path + exactly one missing old path + same hash + same source type -> move/rename
- new path with no match -> new
- missing old path with no new hash match -> deleted

If multiple new paths and/or multiple deleted paths share the same hash and source type, do not guess a move mapping. Treat them as new and deleted rows.

For PNG assets:

- use the same file-level diff rules
- parent linkage is recomputed during processing

Move detection transfers the existing `document_id` for text-bearing sources.

Duplicate live files with identical content are not collapsed.

## 10. Phase 7 - Process Text-Bearing Sources

This phase applies to `markdown` and `docling_json` rows only.

### 10.1 Document Identity

`document_id` rules:

- existing row on same path -> reuse stored `document_id`
- move/rename match -> transfer old `document_id`
- otherwise create a new `document_id`

### 10.2 Parsing

`markdown`:

- load via `ingest/markdown.load_markdown_document`, a frontmatter-aware Markdown reader that parses the wiki's `**Sources**:` and `[[slug]]` conventions before streaming the body into the chunker
- the markdown path does NOT route through Docling's `DocumentConverter` — the wiki content is authored as native Markdown, so the extra Docling conversion step would be overhead without benefit
- preserve heading hierarchy, paragraph text, list item order, table cell text, code block text, and image alt text
- do not let parser-only formatting differences change chunk boundaries for semantically unchanged content

`docling_md`:

- same reader as `markdown` — Docling's `.docling.md` export is Markdown; the extension classifies it as a distinct `source_type` so downstream tooling can attribute the origin, but the loader is unchanged

`docling_json`:

- load the Docling JSON document
- use its structured text and metadata directly
- normalize repeated whitespace and padded table-cell spacing before chunk construction so serialization noise does not create artificial token blow-ups

### 10.3 Chunking

Use the configured Docling native chunker.

If `chunking.strategy = hybrid_docling`, use HybridChunker semantics: start from hierarchical chunking and then apply tokenizer-aware split/merge refinements aligned to the configured tokenizer.

The configured chunker output is an initial candidate set, not a final embedder safety guarantee.

Embedding-safety contract:

- embed requests must use `truncate=false`
- the live embedder response is authoritative for oversize detection
- a chunk that is rejected as oversize must be split again on text boundaries and retried
- emergency split boundaries must prefer paragraph, line, sentence, clause, then word boundaries, in that order
- ingest may emit smaller-than-target chunks during emergency refinement
- ingest must fail loudly only when an oversize chunk cannot be further split into two non-empty text spans

Each chunk receives:

- `chunk_hash = blake3(chunk_text)`
- `chunk_occurrence = ordinal of this`chunk_hash`within the document`
- `chunk_id = blake3(utf8(document_id) + 0x00 + utf8(chunk_hash) + 0x00 + utf8(chunk_occurrence))`

### 10.4 Incremental Diff

Compare the new chunk set to the committed chunk set for the same `document_id`.

Delete stale rows.

Embed only new or changed chunks.

Retain unchanged chunk rows and identities.

If emergency refinement creates additional accepted sub-chunks for a changed source, the committed chunk set for that source must be replaced with the refined accepted set before verification.

Embedding dispatch rules:

- all new/changed chunks are embedded through `embed_texts_batched`, which
  uses the backend's native batch API in `embedding.batch_size`-sized
  batches
- on `EmbeddingContextError` (typically Ollama
  `input length exceeds the context length`) the batch path falls back
  to per-chunk embedding with recursive splitting, so a single oversize
  chunk never stalls the whole document
- both OpenAI and Ollama batches are dispatched concurrently via `_run_batches_concurrently` when `len(batches) > 1` and `max_workers > 1`; the serial-Ollama policy that older versions used has been retired
- embedding writes go through the `embedding_cache` LanceDB table via `merge_insert` on `cache_key` (`chunk_hash|model|dims`), a single atomic upsert per key — no delete-then-add split-brain window. Cache lookups are O(log N) via the BTree scalar index on `cache_key`

### 10.5 Write And Verify

For each text source:

1. set manifest row to `processing`
2. write to LanceDB first (`chunk_vectors` add/replace), then commit chunk_rows in SQLite; on SQLite failure the orchestrator runs a compensating `delete_vector_source` so the two stores never diverge
3. detect entity mentions inline via the loaded Aho-Corasick matcher and persist to `mention_rows` for the newly-written chunks (there is no separate cross-corpus mention-rebuild phase — mention detection is fused into the per-source loop)
4. extract relations via the shared LLM client (schema-enforced `.parse()` with `_RelationsPayload` on the sync path)
5. upsert `corpus_manifest` with `retrieval_status = 'searchable'` and `lifecycle_status = 'complete'`

If any step fails, the source is marked `FAILED` on `corpus_manifest`; the systemic-error circuit breaker in `ingest/error_classification.py` aborts the run after 3 consecutive `SYSTEMIC` errors (DATA errors like `sqlite3.IntegrityError` on duplicate rows do NOT advance the breaker counter, so a burst of duplicate-row failures cannot trip the breaker).

## 11. Phase 8 - Process PNG Assets

This phase applies to `image_png` rows only.

For each PNG:

1. set manifest row to `processing`
2. infer parent linkage when possible
3. upsert `asset_links`
4. set `retrieval_status = 'asset_only'`
5. set manifest row to `complete`

Parent linkage rules:

- if the PNG lives under a sibling `*_images/` directory next to a `.docling.json` file stem, link to that Docling source
- if the PNG lives under a sibling `*_images/` directory next to a `.md` file stem, link to that Markdown source
- if a page number can be parsed from the filename it is stored on `asset_links.page_no`; per-image ordinals are not currently parsed and `asset_index` is left `NULL`
- if no parent can be inferred, keep the asset registered with a null parent

PNG assets are never embedded in V1.

## 12. Phase 9 - Write Ontology Source Rows And Ontology Snapshot

After a successful ontology load:

- upsert `ontology_sources` for all scanned YAML files
- delete removed `ontology_sources` rows
- replace the current `ontology_snapshot` row with the new compiled snapshot state using `snapshot_id = 'current'`
- persist `matcher_termset_hash` and `matcher_term_count` with the committed ontology snapshot

## 13. Mention Indexing (fused into the per-source loop)

There is no separate cross-corpus mention-rebuild phase. Mention detection runs inline per chunk during Phase 7 via `ingest/pipeline/sources.detect_mentions_for_chunks`, using the Aho-Corasick matcher built once at run start (`build_or_load_automaton(plan.ontology)`) — not rebuilt from the committed `ontology_snapshot`.

Consequences of the fused shape:

- ontology change → the matcher is rebuilt at the start of the next `pixi run ingest` run (from the freshly-loaded YAMLs, not from the committed snapshot); the run then re-detects mentions inline as it re-processes changed sources
- text-source-only change → only the changed sources get their mentions re-detected (they're the only ones re-processed by the per-source loop)
- asset-only change → no per-source text work runs, so no mention writes
- mention detection failure per source triggers the per-file recoverable-error path in `orchestrator.py`: the whole file is marked `FAILED` (its LanceDB rows are compensated back out); the rest of the run continues. A cross-corpus "mention indexing failure that preserves the searchable build" contract does not exist — mentions are part of the atomic per-file commit
- the matcher term-set hash is written to `ontology_snapshot.matcher_termset_hash` at snapshot-write time (Phase 9) for downstream drift detection, but there is no pre-write matcher-vs-snapshot reproduction guard in the code path

## 14. Phase 11 - Write Ingest Config Snapshot And Report

Always, after a successful ingest:

- snapshot the config sections that affect stored state
- commit the ingest run summary (`ingest_runs` row)
- print aggregate counts (text / asset / chunk / mention). Per-retrieval-role breakdowns are available through `pixi run status` (which reads `corpus_manifest.retrieval_status`) but are not currently included in the end-of-ingest summary.

The `ingest_runs` row records at least: total files, files completed, `searchable_files_rebuilt`, `asset_files_processed`, `unchanged_files_skipped`, `failed_files`, chunks written, `embedding_tokens`, `llm_tokens`, `estimated_cost_usd`, `embedding_cache_hits`, `embedding_cache_misses`.

## 15. Error Handling

| Failure | Behavior |
|---|---|
| YAML parse or include resolution failure | Halt ingest before writing ontology snapshot |
| Embedder probe failure (Ollama unreachable or OpenAI key missing / rate-limited on the probe) | Halt before any store mutation via `RuntimeError` from `_validate_ingest_dependencies` |
| Markdown / Docling JSON parse failure | Mark file `FAILED` (per-file recoverable path) and continue |
| PNG link inference failure | Register asset with null parent and continue |
| Mention detection failure per file | Mark that file `FAILED`; compensating LanceDB delete keeps stores consistent; run continues |
| LanceDB write failure per file | Same per-file recoverable path (compensating delete + FAILED) |
| SQLite failure per file | Same per-file recoverable path — `sqlite3.Error` is in `_RECOVERABLE_SOURCE_ERRORS`; only outer-scope SQLite failures halt the run |
| 3 consecutive SYSTEMIC failures | Circuit breaker aborts the run before more API spend (per `ingest/error_classification.py`); DATA errors do NOT advance the counter |

Partial ingest is safe because file-level state is committed incrementally.

## 16. Full Rebuild

`ingest --full` does NOT physically truncate any table. Instead it sets `existing_by_path = {}` and `existing_by_hash = {}` before the per-source loop, which forces every file to re-enter the "new" path and be re-processed end-to-end. Overwrite semantics:

- `chunk_vectors` (LanceDB): existing rows for each `source_rel_path` are deleted by `replace_source_chunks` before re-add
- `chunk_rows` (SQLite): same — replaced per source
- `mention_rows`: FK CASCADE from `chunk_rows` re-deletes them when their chunks are replaced
- `corpus_manifest`, `asset_links`, `ontology_sources`, `ontology_snapshot`, `ingest_config`: upsert-in-place; stale rows for files no longer present in the scan get marked `deleted` by the diff pass rather than removed

The upsert-with-diff shape means `--full` is a re-processing pass, not a wipe. To actually reset the store, delete `<paths.data_path>/` before running.
