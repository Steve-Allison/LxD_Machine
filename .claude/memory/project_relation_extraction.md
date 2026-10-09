---
name: Relation extraction feature
description: LLM-based relation extraction from chunks using OpenAI chat/completions; mandatory, no enable toggle
type: project
---

Relation extraction uses OpenAI chat/completions to extract structured relations from document chunks during ingest.

**Why:** Relations enrich the ontology graph and feed into query expansion, retrieval, and MCP tools. The feature was wired into expansion, retrieval, and MCP in commit 06ac94f.

**How to apply:**
- Mandatory: there is no `relation_extraction.enabled` toggle in config (toggles removed; see `.claude/rules/mandatory-features.md`)
- Implementation in src/lxd/ingest/relations.py
- System prompt requires 'json' keyword for structured output
- Ollama timeout was increased to handle extraction (commit 9f3b7f6)
- Relations are yielded during pipeline ingest and stored alongside chunks
- Debug logging was added for relation yield (commit 7f206c5)
