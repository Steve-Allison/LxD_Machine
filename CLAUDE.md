# LxD Machine

An **Instructional Designer / Learning Experience Designer in RAG format**. The corpus encodes how to teach — pedagogical and delivery best practice grounded in academic theory and industry evidence. Instructional design frameworks, cognitive load theory, assessment design, modality selection, delivery formats — everything needed to design effective learning for any topic in any modality.

Built for Adobe field enablement. The system ingests a mixed-format corpus (Markdown, Docling JSON, PNGs), builds an ontology-driven knowledge graph with entity recognition, relation extraction, community detection, and centrality analysis, and exposes retrieval and graph-augmented synthesis via 20 MCP tools. Everything runs locally; MCP is the only external interface.

## Architecture

```text
src/lxd/
├── cli/                      # Typer CLI entrypoints
│   ├── preflight.py          # Schema-integrity + corpus-readiness gate
│   └── graph.py              # build-graph + graph-status
├── app/                      # Bootstrap + AppContext + config.lock reconciliation
├── domain/                   # Pydantic models (citations, IDs, status enums)
├── net/                      # Shared httpx.Client / httpx.AsyncClient factories
├── ingest/                   # Corpus pipeline (sequential orchestrator + per-phase modules)
│   ├── pipeline.py           # Top-level orchestrator: scan → diff → load → chunk → embed → mention → relation → persist
│   ├── scanner.py            # Filesystem scan + BLAKE3 hashing of corpus files
│   ├── diff.py               # Set-wise scan diff (new / deleted / unchanged)
│   ├── markdown.py           # Markdown → ExtractedDocument (calls wiki_metadata)
│   ├── docling.py            # Docling JSON → ExtractedDocument
│   ├── wiki_metadata.py      # Frontmatter (**Sources**:, [[slug]]) parser
│   ├── chunking.py           # Hybrid Docling chunker; recursive context refinement
│   ├── embedder.py           # Batched Ollama/OpenAI embedding with context-aware retry
│   ├── embedding_cache.py    # Content-addressed LanceDB cache: (chunk_hash, model, dims)
│   ├── error_classification.py # Error → TRANSIENT/DATA/SYSTEMIC; circuit breaker
│   ├── llm_client.py         # Shared async OpenAI/Ollama client; prompt-cache helper
│   ├── relations.py          # LLM-based relation extraction (OpenAI primary, Ollama fallback)
│   ├── claims.py             # LLM-based claim extraction
│   ├── mentions.py           # Aho-Corasick mention detection in chunk text
│   └── assets.py             # Asset (PNG) parent-link inference
├── ontology/                 # YAML ontology loading, graph building, Aho-Corasick matching
│   ├── entity_graph.py       # Combined entity graph + 6 centrality metrics
│   ├── communities.py        # Louvain community detection (Leiden optional)
│   ├── evidence.py           # Canonical relation deduplication + evidence provenance
│   ├── profiles.py           # Entity profiles, community reports, LLM enrichment
│   └── schema_models.py      # Pydantic ontology schema (opt-in validation)
├── stores/                   # SQLite + LanceDB (vectors canonical in LanceDB)
│   ├── schema.py             # Numbered migrations (PRAGMA user_version) + integrity check
│   ├── _base_ddl.py          # Authoritative CREATE TABLE / CREATE INDEX statements
│   ├── connection.py         # Pragma-tight SQLite connect + close hooks
│   ├── sqlite.py             # Query/upsert API (orchestrator over typed records)
│   ├── _sqlite_rows.py       # Row → record adapters
│   ├── lancedb.py            # Canonical vector store + chunk_vectors / embedding_cache tables
│   ├── lance_sql.py          # Safe LanceDB filter builders
│   ├── sql_helpers.py        # Safe SQLite IN (?, ?, …) helpers
│   ├── models.py             # Typed dataclasses for records (frozen, slots=True)
│   └── llm_jobs.py           # Persistent LLM job queue API
├── retrieval/                # Query pipeline: dense → rerank → expansion → graph routing → synthesis
│   └── graph_routing.py      # Graph context augmentation for synthesis
├── synthesis/                # Answer generation with citations + graph + transitive sources
├── mcp/                      # FastMCP server (20 read-only tools)
│   ├── async_runtime.py      # run_tool: async wrapper + hard timeout for tool bodies
│   └── tools.py              # Tool implementations
├── observability/            # structlog: JSON/console, UTC, log_duration, scrub_secrets
└── settings/                 # Pydantic config models + YAML loader
```

Key directories outside `src/`:

- Corpus root — set in `config.yaml` (`paths.corpus_path`). Default: the curated wiki at `~/Documents/_Knowledge/wiki/` (147 markdown pages with frontmatter Sources lines and `[[slug]]` cross-references). The legacy raw research at `Knowledge_Base/` is retained as an archive.
- `Yamls/` — ontology definitions
- `Plans/` — architecture and design specs
- `tests/` — pytest suite
- `data/` — SQLite + LanceDB stores (gitignored, rebuildable). Auto-backups before destructive migrations.
- `start.sh` — interactive launcher: preflight, ingest, build-graph, MCP, status.
- `.env` — API keys (OPENAI_API_KEY, etc.). Loaded by `app/bootstrap.py` via `python-dotenv` at startup. Never commit.

## Common Commands

```bash
pixi run preflight       # Schema-integrity + corpus-readiness gate (run before ingest)
pixi run ingest          # Incremental corpus ingestion
pixi run ingest --full   # Full rebuild (recreates SQLite + LanceDB tables)
pixi run status          # Corpus and ontology status
pixi run eval            # Retrieval evaluation against tests/eval/eval_set.json
pixi run mcp             # Launch MCP server
pixi run build-graph     # Build knowledge graph (incremental, resumable)
pixi run graph-status    # Knowledge graph build state and statistics
pixi run test            # pytest -q
pixi run lint            # ruff check src tests
pixi run fmt             # ruff format src tests
pixi run typecheck       # pyright src
./start.sh               # Interactive launcher (preflight + ingest + build-graph + MCP)
```

## MCP Tools

20 read-only tools exposed via FastMCP (>=3.0) over stdio transport:

**Corpus tools:** `corpus_status`, `get_entity_types`, `get_related_concepts`, `search_corpus`, `find_documents_for_concept`, `get_corpus_relations`

**Knowledge graph tools:** `get_entity_summary`, `get_community_context`, `get_similar_entities`, `search_entities`, `inspect_evidence`, `find_path_between_entities`, `find_weighted_path`, `get_hub_entities`, `find_bridge_entities`, `find_foundational_entities`, `get_entity_graph_stats`

**Full answer pipeline:** `search_knowledge` (graph-augmented synthesis), `search_knowledge_deep` (same + structured graph context), `get_graph_overview` (KG health check)

## Knowledge Graph

The knowledge graph pipeline builds on top of ingested corpus data:

1. **Claim extraction** — LLM-based extraction of assertions, definitions, comparisons, causal, and procedural claims from chunks
2. **Entity graph** — combined graph from ontology edges + corpus-extracted relations
3. **Centrality** — 6 metrics: PageRank, betweenness, closeness, in-degree, out-degree, eigenvector
4. **Community detection** — Louvain via NetworkX (Leiden available via optional graspologic)
5. **Entity profiles** — deterministic summaries with centrality scores, optional LLM enrichment
6. **Community reports** — deterministic summaries per community, optional LLM enrichment
7. **Graph-augmented synthesis** — entity profiles, community reports, and claims prepended to synthesis prompt when entities match the query

The graph build is a resumable state machine (`pixi run build-graph`). Graph context is additive — when the graph is not yet built or no entities match a query, the pipeline degrades gracefully to a graph-free baseline.

## Design Principles

- **Incremental by default**: ingest skips unchanged files (BLAKE3 content hash), detects moves. Graph build resumes from last incomplete phase.
- **Portable stores**: all SQLite PKs and FKs use corpus-relative paths. The `data/` folder can be copied between machines; `pixi run ingest` updates machine-local absolute paths. LanceDB vectors are also keyed by relative path.
- **Safe by default**: `pixi run build-graph --full` requires interactive confirmation before re-extracting claims (costs API calls and time). No command unconditionally destroys the knowledge graph.
- **Rebuildable**: all stores can be rebuilt from source via `pixi run ingest --full` and `pixi run build-graph --full`.
- **Explicit provenance**: every chunk traces back to source document, page, and extraction method. Every claim and relation traces to source chunk. Wiki pages additionally carry transitive `**Sources**:` citations on every chunk so synthesis can attribute back to the original research.
- **Ontology-first**: entity recognition uses Aho-Corasick automaton built from YAML definitions; relations extracted via LLM. Wiki pages also contribute hand-curated `[[slug]]` cross-references parsed at ingest time and persisted on chunk rows.
- **Graceful degradation**: the system remains usable when the knowledge graph is not yet built or the reranker service is unavailable.
- **Async MCP surface**: every tool is `async def`; synchronous bodies run through `lxd.mcp.async_runtime.run_tool` with a hard `tool_timeout_secs` cap.
- **Single source of truth for vectors**: LanceDB is canonical; `chunk_rows.vector_json` was dropped in schema migration 0002.
- **Schema integrity is a hard gate**: `ensure_schema` runs numbered migrations under `PRAGMA user_version`, then verifies `foreign_key_check` + required tables/columns. A half-migrated DB raises `SchemaIntegrityError` and refuses writes. The CLI `pixi run preflight` exposes this check.
- **Migrations are reversible-by-backup**: every pending migration triggers an auto-backup (`*.pre-migration-vN-to-vM-<timestamp>.sqlite3.bak`) before destructive DDL runs.
- **Embedding cache is content-addressed**: `(chunk_hash, embedding_model, embedding_dims)` keys; cache survives full rebuilds since identical text + identical model = identical vector. Stored in a separate LanceDB table (`embedding_cache`) so chunk-rebuilds don't wipe it.
- **Systemic-error circuit breaker**: ingest classifies errors as TRANSIENT / DATA / SYSTEMIC; 3 consecutive SYSTEMIC errors abort the run before further API spend. DATA errors (e.g. `IntegrityError`) don't advance the counter — duplicate-row failures don't trip the breaker.
- **Cross-store atomicity**: chunk persistence is LanceDB-first, then SQLite; a SQLite failure runs a compensating `delete_vector_source` so the two stores never diverge.
- **Config drift is visible**: bootstrap hashes the resolved config (blake3) and reconciles `data/config.lock`; mismatches log a `config.lock.mismatch` warning without overwriting.

## Key Patterns

- IDs are deterministic (BLAKE3 hash of content + context), not random UUIDs.
- Configuration is fully Pydantic v2 validated, loaded from YAML profiles via `settings/loader.py`. Tenancy (`corpus_id`) and observability exporters are first-class config sections.
- MCP tools are read-only with FastMCP (>=3.0) hints for client compatibility and are invoked through the async runtime with per-call timeouts.
- Chunking uses recursive context refinement when embedding dimensions exceed limits; the batched embedder (`ingest/embedder.py`) falls back per-text on Ollama context-length errors.
- Graph context is additive — prepended to synthesis prompts as structured framing, not fused via RRF with chunks.
- Schema evolution runs automatically on bootstrap via numbered migrations keyed by `PRAGMA user_version` (`stores/schema.py`).
- LanceDB `where` clauses are built through `stores/lance_sql.py`; SQLite `IN (?, ?, …)` clauses through `stores/sql_helpers.py`. No raw string interpolation.
- Long-running LLM jobs are persisted via `stores/llm_jobs.py` (`queued → running → succeeded|failed|cancelled`) so executors can crash and resume.
- Structured logs ship through `observability/logging.py` with UTC timestamps, `contextvars` propagation, a `log_duration` context manager, and a `scrub_secrets` processor.

## Test Layout

- `tests/unit/` — pure-logic tests (no disk, no network)
- `tests/integration/` — temp-dir SQLite/LanceDB + local FastMCP
- `tests/golden/` — API-surface regressions (e.g. `mcp_tool_manifest.json`)
- Markers: `unit`, `integration`, `e2e`, `property`, `benchmark`, `slow`
- Refresh a golden file with `pixi run pytest --update-golden`

## Design Specs

Detailed specifications live in `Plans/`:

- `00_PURPOSE_AND_BACKGROUND.md` — scope, outcomes, constraints
- `01_ARCHITECTURE.md` — system architecture and store design
- `01b_CODEBASE_STRUCTURE.md` — module boundaries
- `02_DATA_SCHEMA.md` — SQLite and LanceDB schema definitions
- `02b_CONFIG_SPEC.md` — YAML configuration specification
- `02c_ENTITY_EXTRACTION.md` — ontology entity extraction design
- `03_INGEST_SPEC.md` — ingestion pipeline detail
- `04_QUERY_SPEC.md` — query and retrieval pipeline
- `05_MCP_SPEC.md` — MCP tool interface specification (20 tools)
- `07_USER_GUIDE.md` — end-user operation guide
- `08_KNOWLEDGE_GRAPH_SPEC.md` — knowledge graph pipeline specification

---

*Last reviewed: 2026-05-05. Updated: corpus default switched to curated wiki; ingest module list refreshed (added `pipeline.py`, `wiki_metadata.py`, `embedding_cache.py`, `error_classification.py`, `chunking.py`, `markdown.py`, `docling.py`, `mentions.py`, `assets.py`, `scanner.py`, `diff.py`, `llm_client.py`); resilience principles documented (schema-integrity gate, auto-backup on migration, content-addressed cache, circuit breaker, cross-store atomicity); `pixi run preflight` and `start.sh` added to commands. Removed false claim of Hamilton DAG patterns in `.claude/rules/project-conventions.md` — pipeline is a sequential per-source orchestrator, no DAG framework.*
