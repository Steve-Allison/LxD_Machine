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

1. validate config and dependencies
2. initialize stores
3. scan and diff ontology sources
4. load ontology, build graph, build matcher
5. scan and classify corpus files
6. diff corpus files and detect moves/renames
7. process text-bearing sources
8. process PNG assets
9. write ontology source rows and ontology snapshot
10. rebuild mention index when enabled and required
11. write ingest config snapshot and report

Each phase is explicit and must fail loudly.

## 4. Phase 1 - Validate Config And Dependencies

Before mutating any store:

- validate settings
- verify `config.paths.corpus_path` and `config.paths.ontology_path`
- verify Ollama is reachable
- verify the configured embedder passes the defined readiness probe
- open SQLite with WAL-capable settings

Model readiness probe contract:

- embedder: one embedding request on a fixed probe string with `truncate=false`
- ingest startup must fail if the configured embedder cannot complete its probe within the configured timeout

If Ollama is unreachable or the configured embedder is missing, ingest must halt before any searchable-store mutation.

Asset-only and ontology-only work may still proceed.

## 5. Phase 2 - Initialize Stores

Create or migrate:

- LanceDB `chunks`
- SQLite `corpus_manifest`
- SQLite `asset_links`
- SQLite `ontology_sources`
- SQLite `ontology_snapshot`
- SQLite `mentions`
- SQLite `ingest_config`
- SQLite `ingest_runs`

SQLite initialization must set:

- `PRAGMA journal_mode=WAL;`
- `PRAGMA foreign_keys=ON;`

## 6. Phase 3 - Scan And Diff Ontology Sources

Scan the full ontology root using `settings.ontology.include_globs`.

In-scope files are YAML files only.

Ignored examples:

- `README.md`
- progress markdown files
- `.DS_Store`

Diff against `ontology_sources` by `file_path` and `blake3_hash`.

`ontology_changed` is true when any scanned YAML is new, changed, or removed.

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

Classification rules:

- `.md` -> `markdown`
- `.docling.json` -> `docling_json`
- `.png` -> `image_png`

Other file types are out of scope unless added to config later.

For every in-scope file, collect:

- absolute path
- relative path
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

- convert the file through Docling's supported document-conversion path to obtain a `DoclingDocument`
- use Docling's native chunking path from `DoclingDocument`, not post-export markdown splitting
- preserve heading hierarchy, paragraph text, list item order, table cell text, code block text, and image alt text
- do not let parser-only formatting differences change chunk boundaries for semantically unchanged content

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
- `chunk_occurrence = ordinal of this `chunk_hash` within the document`
- `chunk_id = blake3(utf8(document_id) + 0x00 + utf8(chunk_hash) + 0x00 + utf8(chunk_occurrence))`

### 10.4 Incremental Diff

Compare the new chunk set to the committed chunk set for the same `document_id`.

Delete stale rows.

Embed only new or changed chunks.

Retain unchanged chunk rows and identities.

If emergency refinement creates additional accepted sub-chunks for a changed source, the committed chunk set for that source must be replaced with the refined accepted set before verification.

### 10.5 Write And Verify

For each text source:

1. set manifest row to `processing`
2. write changed chunk rows
3. verify the final committed chunk set for the `document_id`
4. set `retrieval_status = 'searchable'`
5. set manifest row to `complete`

If verification fails, the row remains incomplete and is retried on the next run.

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
- if page or image ordinals can be parsed from the filename, store them
- if no parent can be inferred, keep the asset registered with a null parent

PNG assets are never embedded in V1.

## 12. Phase 9 - Write Ontology Source Rows And Ontology Snapshot

After a successful ontology load:

- upsert `ontology_sources` for all scanned YAML files
- delete removed `ontology_sources` rows
- replace the current `ontology_snapshot` row with the new compiled snapshot state using `snapshot_id = 'current'`
- persist `matcher_termset_hash` and `matcher_term_count` with the committed ontology snapshot

## 13. Phase 10 - Rebuild Mention Index

Run mention indexing when:

- mention indexing is enabled, and
- ontology changed, or
- any searchable text source changed

Rules:

- ontology change -> full rebuild over all searchable chunks
- only text-source changes -> incremental rebuild over changed `document_id` values
- asset-only changes do not trigger mention rebuild
- mention indexing failure must not invalidate an otherwise usable searchable build
- the matcher must be rebuilt from the committed ontology snapshot before mention indexing begins
- the rebuilt matcher term set must reproduce the committed `matcher_termset_hash` before any mention rows are written

## 14. Phase 11 - Write Ingest Config Snapshot And Report

Always, after a successful ingest:

- snapshot the config sections that affect stored state
- commit the ingest run summary
- print counts by source type and retrieval role

The report must show at least:

- total in-scope corpus files
- searchable files completed
- asset-only files completed
- chunks written
- ontology YAML count
- entity count
- matcher term count
- matcher termset hash

## 15. Error Handling

| Failure | Behavior |
|---|---|
| YAML parse or include resolution failure | Halt ingest before writing ontology snapshot |
| Ollama unreachable or required embed model missing | Halt before searchable-store mutation |
| Markdown parse failure | Mark file failed and continue |
| Docling JSON parse failure | Mark file failed and continue |
| PNG link inference failure | Register asset with null parent and continue |
| Mention indexing failure | Warn, record failure, and keep the searchable build usable |
| LanceDB write failure | Halt current run |
| SQLite failure | Halt current run |

Partial ingest is safe because file-level state is committed incrementally.

## 16. Full Rebuild

`ingest --full` performs:

1. clear LanceDB `chunks`
2. clear `mentions`
3. clear `asset_links`
4. clear `corpus_manifest`
5. clear `ontology_sources`
6. clear `ontology_snapshot`
7. clear `ingest_config`
8. rerun phases 1 through 11
