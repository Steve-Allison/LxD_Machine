---
paths:
  - "src/lxd/stores/**/*.py"
  - "src/lxd/ingest/pipeline/**/*.py"
  - "Plans/02_DATA_SCHEMA.md"
---

# Stores and Paths

Invariants for the storage layer (SQLite + LanceDB). Auto-loads on edits to
`src/lxd/stores/`, the pipeline modules, or the schema spec.

## Keys are corpus-relative — always

Every SQLite PK and FK uses **corpus-relative paths**. The `data/` folder is
designed to be portable between machines; `pixi run ingest` updates
machine-local absolute paths on each run.

- Never use absolute paths as PKs / FKs.
- Never persist `corpus_root` in a row — derive it from config at read time.
- LanceDB row keys (`source_rel_path`) also stay relative.
- The single exception is `corpus_manifest.absolute_path` — a non-PK column
  refreshed by ingest. Reads must tolerate it being stale until the next ingest.

## LanceDB is canonical for vectors

Migration v0002 dropped `chunk_rows.vector_json`. LanceDB's `chunk_vectors`
table is the single source of truth for embeddings.

- Do not reintroduce vector storage in SQLite.
- Reads that need both metadata and vectors join LanceDB→SQLite by
  `chunk_id`, not the reverse.
- `embedding_cache` is a separate LanceDB table keyed by
  `(chunk_hash, embedding_model, embedding_dims)`. It survives full rebuilds.

## Cross-store atomicity: LanceDB first, then SQLite

Persistence order is fixed:

1. Write to LanceDB (`chunk_vectors`, `entity_embeddings`)
2. Write to SQLite (chunk rows, mentions, claims, etc.)
3. On SQLite failure: run a compensating LanceDB `delete_vector_source`

If you find yourself reading "what if SQLite succeeds but LanceDB failed?",
that question shouldn't be possible — the order makes it impossible. If your
change makes it possible, the change is wrong.

## Schema migrations are append-only

`src/lxd/stores/schema.py` runs numbered migrations keyed by
`PRAGMA user_version`. The DDL itself lives in `src/lxd/stores/_base_ddl.py`.

- **Never edit an existing migration.** Migrations are immutable history. Fix
  forward with a new numbered migration.
- Every destructive migration must auto-backup first (the pattern is already
  in `_run_pending_migration_with_backup`).
- After running migrations, `ensure_schema` verifies `PRAGMA foreign_key_check`
  and required tables/columns. A half-migrated DB raises `SchemaIntegrityError`
  and refuses writes.
- Use `pixi run preflight` to surface schema state before assuming anything
  about the store.

## Store APIs are the only mutation path

Code outside `src/lxd/stores/` does not call SQLite or LanceDB directly. It goes
through the store-layer API:

- `src/lxd/stores/sqlite/*` for SQLite (runs, manifest, ontology, chunks,
  claims, kg_profiles, kg_relations, summary)
- `src/lxd/stores/lancedb.py` for LanceDB
- `src/lxd/stores/sqlite/_pool.py` for the per-thread MCP-path connection pool

Direct `sqlite3.connect()` or `lancedb.connect()` outside the store layer is a
violation. The pooling, pragma hardening, and migration gate live in the store
layer for a reason.

## Filter-clause discipline — no string interpolation

LanceDB `where` clauses build through `stores/lance_sql.py`. SQLite
`IN (?, ?, …)` clauses build through `stores/sql_helpers.py`.

- Never construct SQL or LanceDB filter strings via f-strings or `.format()`.
- Always go through the safe helpers; they handle quoting and parameterisation.

## Cross-reference

- `Plans/02_DATA_SCHEMA.md` — canonical schema definitions
- `.claude/rules/ingest-discipline.md` — operational rules for the pipeline
- `~/.claude/CLAUDE.md` §1 — read governance docs end-to-end before editing
  store code
