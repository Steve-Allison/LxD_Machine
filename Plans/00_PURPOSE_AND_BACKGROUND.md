# LxD Machine - Purpose

## 1. Scope

- local-only
- single-user
- rebuildable from a configured corpus and ontology once runtime dependencies and local model artifacts are provisioned
- the corpus boundary is whatever `paths.corpus_path` in `config.yaml` resolves to; the default is the curated wiki at `~/AI_Projects+Code/knowledge/wiki/` (~269 top-level pages), and the legacy raw corpus under `Knowledge_Base/` is retained only as an archive
- every file under the resolved corpus root is scanned; V1 durable handling covers `.md`, `.docling.md`, `.docling.json`, and `.png`
- MCP is the only external interface

## 2. Required Outcomes

- scan every file under the configured `paths.corpus_path`
- durably ingest every V1-supported file type under that root
- index text-bearing sources for retrieval and cited answering
- register binary assets with durable provenance, even when they are not queryable evidence in V1
- load the ontology from the full repo-local `Yamls/` tree
- expose corpus search, ontology lookup, status, knowledge-graph tools, and full answer-synthesis through MCP
- report committed ingest, ontology, and knowledge-graph state

## 3. Corpus And Ontology Inputs

Corpus root (from `config.yaml :: paths.corpus_path`):

- default: `~/AI_Projects+Code/knowledge/wiki/`
- overridable per-machine via a `config.<profile>.yaml`

Ontology root:

- `<project_root>/Yamls`

Entity source subtree:

- `<project_root>/Yamls/entities`

Corpus / ontology counts are inventory-time facts, not spec facts, and drift as the wiki and ontology grow. Run `pixi run status` for the current committed counts on your machine; the wiki has grown from 262 → 269+ top-level pages over the SOTA sweep. The ontology tree carries ~158 YAML files across 27 entity YAMLs.

## 4. File Classes

The corpus contains four durable file classes and the pipeline treats them differently:

- `markdown` (`.md`): primary text source, chunked and searchable
- `docling_json` (`.docling.json`): primary text source, chunked and searchable with structural provenance
- `docling_md` (`.docling.md`): Markdown export from Docling, chunked and searchable
- `image_png` (`.png`): binary corpus asset, durably registered with `retrieval_status=asset_only` and linked to a parent text source when possible

V1 query answers cite only text-bearing chunk sources.

V1 still ingests PNG files by registering them in durable corpus state and exposing their provenance through status and MCP lookup.

## 5. Operational Constraints

- all source content stays local
- all services bind to localhost only
- ingest progress must be committed incrementally
- partial builds must remain inspectable
- all runtime-selectable behavior must be config-driven
- workload and performance claims must be benchmarked

## 6. V1

V1 includes:

- full corpus inventory over all in-scope file types
- durable ingest for markdown and Docling JSON text sources
- durable registration for PNG assets
- ontology load from the full `Yamls` tree with `!include` resolution
- committed status reporting
- corpus search over text-bearing sources
- baseline reranking with explicit dense-only fallback if the reranker is unavailable
- cited answer synthesis over text-bearing sources
- MCP access to required tools

V1 excludes:

- multimodal image embeddings
- image-to-text OCR during ingest
- PNG files as direct cited answer evidence
- required hybrid retrieval

## 7. Success Conditions

- every in-scope file under the configured `paths.corpus_path` is represented in committed ingest state
- every markdown and Docling JSON source is either searchable or explicitly failed with a recorded error
- every PNG asset is durably registered, even when it is not searchable
- interrupted ingest leaves committed usable progress
- `status` reflects committed state by file type and retrieval role
- query tools work against the built store
- MCP tools work from a documented `stdio` MCP client configuration

These documents are the source of truth for the rewrite:

- `01_ARCHITECTURE.md`
- `01b_CODEBASE_STRUCTURE.md`
- `02_DATA_SCHEMA.md`
- `02b_CONFIG_SPEC.md`
- `02c_ENTITY_EXTRACTION.md`
- `03_INGEST_SPEC.md`
- `04_QUERY_SPEC.md`
- `05_MCP_SPEC.md`
- `07_USER_GUIDE.md`
- `08_KNOWLEDGE_GRAPH_SPEC.md`
