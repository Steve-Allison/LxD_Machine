# LxD Machine

A local-only, single-user knowledge system for instructional design content. Ingests a mixed-format corpus (Markdown, Docling JSON, PNGs), builds a searchable index with ontology-driven entity recognition, and exposes retrieval via MCP tools.

## Architecture

```
src/lxd/
├── cli/          # Typer CLI: ingest, status, eval
├── app/          # Bootstrap and status reporting
├── domain/       # Pydantic models (citations, IDs, status enums)
├── ingest/       # Corpus pipeline: scan → chunk → embed → mention → relation → persist
├── ontology/     # YAML ontology loading, graph building, Aho-Corasick matching
├── stores/       # SQLite (metadata/manifest) + LanceDB (vectors)
├── retrieval/    # Query pipeline: dense search → rerank → expansion → synthesis
├── synthesis/    # Answer generation with citations
├── mcp/          # FastMCP server (6 read-only tools)
├── observability/# structlog configuration
└── settings/     # Pydantic config models + YAML loader
```

Key directories outside `src/`:
- `Knowledge_Base/` — corpus root (gitignored)
- `Yamls/` — ontology definitions (159 YAML files, 318 entity types)
- `Plans/` — architecture and design specs
- `tests/` — pytest suite
- `data/` — SQLite + LanceDB stores (gitignored, rebuildable)

## Common Commands

```bash
pixi run ingest          # Incremental corpus ingestion
pixi run ingest --full   # Full rebuild
pixi run status          # Corpus and ontology status
pixi run eval            # Retrieval evaluation against tests/eval/eval_set.json
pixi run mcp             # Launch MCP server
pixi run test            # pytest -q
pixi run lint            # ruff check src tests
pixi run fmt             # ruff format src tests
pixi run typecheck       # pyright src
```

## Design Principles

- **Incremental by default**: ingest skips unchanged files (BLAKE3 content hash), detects moves.
- **Rebuildable**: all stores can be rebuilt from source via `pixi run ingest --full`.
- **Explicit provenance**: every chunk traces back to source document, page, and extraction method.
- **Ontology-first**: entity recognition uses Aho-Corasick automaton built from YAML definitions; relations extracted via LLM.

## Key Patterns

- IDs are deterministic (BLAKE3 hash of content + context), not random UUIDs.
- Configuration is fully Pydantic v2 validated, loaded from YAML profiles via `settings/loader.py`.
- MCP tools are read-only with FastMCP (>=3.0) hints for client compatibility.
- Chunking uses recursive context refinement when embedding dimensions exceed limits.

## Design Specs

Detailed specifications live in `Plans/`:
- `00_PURPOSE_AND_BACKGROUND.md` — scope, outcomes, constraints
- `01_ARCHITECTURE.md` — system architecture and store design
- `01b_CODEBASE_STRUCTURE.md` — module boundaries
- `03_INGEST_SPEC.md` — ingestion pipeline detail
- `05_MCP_SPEC.md` — MCP interface specification
