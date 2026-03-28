---
name: Embedding backend migration to OpenAI
description: Decision to move from local Ollama embeddings to OpenAI text-embedding-3-small, with dual-backend support
type: project
---

Migrated embedding backend from local Ollama (nomic-embed-text, 768d) to OpenAI (text-embedding-3-small, 1536d) as the primary path, with Ollama retained as a local alternative.

**Why:** Steve asked "why can we not use OpenAI all the way through?" — desire for simpler, cloud-based stack. The Ollama-based local pipeline required running multiple local services. OpenAI embeddings + OpenAI chat/completions for relation extraction makes the system operational without local GPU dependencies.

**How to apply:**
- config.yaml has `embed_backend` field — can be `ollama` or `openai`
- OpenAI embeddings require OPENAI_API_KEY in environment
- Chunk size was tuned to 256 tokens (overlap 50) during this migration
- A `config.m1max.yaml` profile exists for local M1 Max settings
- The reranker still uses llama.cpp locally (model_source: ollama_blob reads weights from Ollama's blob cache but runs via llama-server, not Ollama itself)
- As of 2026-03-27, a full ingest with OpenAI completed successfully: ~3500 embedding batches + ~3500 relation extraction calls, 0 errors
