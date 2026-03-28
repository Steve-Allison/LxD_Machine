---
name: LxD Machine project overview
description: Core architecture, tech stack, and scale of the LxD Machine knowledge retrieval system
type: project
---

LxD Machine is an ontology-first knowledge retrieval system for instructional design content, built for Adobe field enablement.

**Why:** Steve needs a fast, grounded retrieval system over a large corpus of instructional design documents, sales plays, and methodology frameworks — not just vector search but structurally-aware retrieval with entity mentions, ontology graph expansion, and provenance.

**How to apply:** Treat the ontology (Yamls/) as the source of truth for domain concepts. The pipeline is: Docling parse → HybridChunker → embedding → LanceDB + SQLite. Retrieval is: query embedding → dense vector search → optional graph expansion → rerank → synthesis with citations.

Key numbers (as of 2026-03-27):
- 430 text files, 4314 asset files
- 100+ ontology YAML files across entities/, methodology/, reference/, sales/, corpus/
- 9 ontology validation issues (undeclared relation types, unresolved targets)

Tech stack:
- Python 3.14 on osx-arm64, managed via pixi
- LanceDB (vector store), SQLite (metadata/chunks), Docling (parsing)
- FastMCP >=3.0 for MCP server (5 read-only tools)
- llama.cpp as conda dependency for local reranker
- OpenAI for embeddings and relation extraction (cloud path)
