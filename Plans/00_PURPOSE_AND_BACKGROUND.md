# LxD Machine - Purpose

## 1. Scope

- local-only
- single-user
- rebuildable from repo-local corpus and ontology inputs once runtime dependencies and local model artifacts are provisioned
- `Knowledge_Base` is the full corpus boundary
- every file under the corpus root is scanned; V1 durable handling covers `.md`, `.docling.json`, and `.png`
- MCP is the only external interface

## 2. Required Outcomes

- scan every file under `/Users/steveallison/AI_Projects+Code/LxD_Machine/Knowledge_Base`
- durably ingest every V1-supported file type under that root
- index text-bearing sources for retrieval and cited answering
- register binary assets with durable provenance, even when they are not queryable evidence in V1
- load the ontology from the full repo-local `Yamls` tree
- expose corpus search, ontology lookup, status, and asset lookup through MCP
- report committed ingest and ontology state

## 3. Corpus And Ontology Inputs

Corpus root:

- `/Users/steveallison/AI_Projects+Code/LxD_Machine/Knowledge_Base`

Ontology root:

- `/Users/steveallison/AI_Projects+Code/LxD_Machine/Yamls`

Entity source subtree:

- `/Users/steveallison/AI_Projects+Code/LxD_Machine/Yamls/entities`

Measured inventory in this repo on 2026-03-10:

- text-bearing corpus files: `348`
- markdown files: `302`
- Docling JSON files: `46`
- PNG asset files: `2160`
- duplicate physical text-bearing files by content hash: `1`
- ontology YAML files: `159`
- entity YAML files: `27`
- entity types: `318`

## 4. File Classes

The corpus contains three file classes and the plan must treat them differently:

- `markdown`: primary text source, chunked and searchable
- `docling_json`: primary text source, chunked and searchable with structural provenance
- `png`: binary corpus asset, durably registered and linked to a parent source when possible

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

- every in-scope file under `Knowledge_Base` is represented in committed ingest state
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
- `06_BUILD_PLAN.md`
- `07_USER_GUIDE.md`
