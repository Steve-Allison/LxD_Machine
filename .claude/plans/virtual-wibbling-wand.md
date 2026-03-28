# Async Concurrency + Batch API via Shared LLM Service

## Context

Four modules make cloud LLM calls, each creating a new client per call with no concurrency, no batching, and no prompt caching. For 23,607 chunks the claim/relation extraction alone takes 6-13 hours. The user wants shared infrastructure for async concurrency (interactive), Batch API (cost-optimised bulk), and prompt caching (50% off cached input tokens).

## Architecture: Shared LLM Service

All cloud LLM chat completion calls go through a single shared module rather than duplicating client management, fallback logic, and concurrency in each consumer.

### New: `src/lxd/ingest/llm_client.py`

Provides:

- **Lazy `AsyncOpenAI` singleton** via `get_async_client(config)` — reuses connection pool across all consumers
- **Lazy `ollama.Client` singleton** via `get_ollama_client(config)` — same
- **`call_openai_async(system_prompt, user_prompt, model, temperature, timeout, max_tokens, response_format)`** — single async OpenAI call
- **`call_ollama_sync(system_prompt, user_prompt, model, temperature, timeout, format_)`** — single sync Ollama call (wrapped in `asyncio.to_thread` when needed)
- **`call_with_fallback_async(system_prompt, user_prompt, config, primary_backend, ...)`** — OpenAI → Ollama fallback, async
- **`run_concurrent_extraction(items, extract_fn, max_concurrent, sub_batch_size, commit_fn)`** — generic semaphore-gated gather with sub-batch commits. `extract_fn` is an async callable per item; `commit_fn` persists each sub-batch to SQLite.
- **`build_cached_system_prompt(base_prompt, entity_vocabulary, predicate_vocabulary?)`** — expands prompt with reference vocabulary to cross the 1,024-token OpenAI prompt caching threshold
- **Batch API**: `prepare_batch_jsonl(items, build_messages_fn, model, temperature, max_tokens, output_path)`, `submit_batch(jsonl_path, metadata)`, `poll_batch(batch_id)`, `collect_batch_results(batch_id, parse_fn)`

### Consumers (4 modules)

#### 1. `src/lxd/ingest/claims.py` — Claim extraction (bulk, async)

- Defines `_CLAIM_BASE_PROMPT`, `_build_claim_user_prompt()`, `_parse_claim_response()`, `_build_claim_records()`
- Pre-fetches all entity IDs per chunk in one SQL query (eliminates N+1)
- `extract_claims_for_chunks()` → `asyncio.run(_extract_claims_async())` → `llm_client.run_concurrent_extraction()`
- Batch API: `prepare_claims_batch_jsonl()`, `submit_claims_batch()`, `collect_claims_batch()` — delegate to `llm_client` with claim-specific builders/parsers

#### 2. `src/lxd/ingest/relations.py` — Relation extraction (bulk, async)

- Defines `_RELATION_BASE_PROMPT`, `_build_relation_user_prompt()`, `_parse_relation_response()`
- New `extract_relations_for_chunks_async()` for concurrent per-file extraction
- Batch API: `prepare_relations_batch_jsonl()`, `submit_relations_batch()`, `collect_relations_batch()`

#### 3. `src/lxd/ontology/profiles.py` — LLM enrichment (bulk, async)

- Currently iterates profiles + community reports sequentially with new client per call
- Replace `_call_openai_enrichment()` / `_call_ollama_enrichment()` / `_call_llm_enrichment()` with `llm_client.call_with_fallback_async()`
- `enrich_entity_profiles_with_llm()` becomes async, uses `run_concurrent_extraction()` over profiles and reports
- No JSON response format needed — plain text summaries

#### 4. `src/lxd/synthesis/answering.py` — Synthesis (single-call, client reuse)

- Currently creates Ollama client per call
- Replace `_client()` with `llm_client.get_ollama_client(config)` for connection pool reuse
- No async/batch needed — single call per query at request time

### Config — `src/lxd/settings/models.py` ✅ DONE

Added to `KnowledgeGraphConfig`:

- `claim_extraction_max_concurrent: int = Field(default=50, gt=0)`
- `claim_extraction_sub_batch_size: int = Field(default=500, gt=0)`

Added to `RelationExtractionConfig`:

- `max_concurrent: int = Field(default=50, gt=0)`
- `sub_batch_size: int = Field(default=500, gt=0)`

### Pipeline — `src/lxd/ingest/pipeline.py`

In `_build_source_records()`: replace sequential per-chunk relation extraction loop with `asyncio.run(extract_relations_for_chunks_async(...))`.

### CLI — `src/lxd/cli/graph.py` + `__main__.py`

- Add `--batch` flag to `build_graph_command`: prepares JSONL, submits to Batch API, prints batch_id, exits
- Add `collect-batch` command: takes batch_id, downloads results, inserts into SQLite
- Register `collect-batch` in `__main__.py`
- Add `pixi run collect-batch` task to `pixi.toml`

### Tests

- `tests/test_llm_client.py` — mock AsyncOpenAI, test concurrent gather, test sub-batch commits, test JSONL output, test prompt caching expansion
- `tests/test_claims_async.py` — test claim-specific wiring
- `tests/test_relations_async.py` — test relation-specific wiring

## Key design decisions

- **Single shared service** — all chat completion calls go through `llm_client.py`; no duplicated client creation, fallback logic, or concurrency mechanics
- **Client singletons** — one `AsyncOpenAI` and one `ollama.Client` per process, reusing connection pools
- **Entity ID pre-fetch** — eliminates 23,607 individual SQL queries in claim extraction
- **Sub-batch commit** (500 chunks): gather → SQLite transaction → log → next batch. Preserves resumability.
- **Semaphore(50)** — caps in-flight requests (configurable per consumer via config)
- **Ollama fallback** — sync calls wrapped in `asyncio.to_thread()` for async compatibility
- **Batch API sidecar** — JSON metadata file alongside JSONL tracks batch_id → chunk mapping
- **Generic orchestrator** — `run_concurrent_extraction()` takes callbacks, not hardcoded domain logic

## Prompt caching optimisation

OpenAI automatically caches prompt prefixes (50% off cached input for gpt-4o-mini). Requirements:

- **1,024+ token prefix** for caching to activate
- **Byte-identical prefix** across calls — system prompt must not vary
- **Static content first, variable content last**

Current system prompts are ~200 tokens — too short for caching. To hit the threshold:

- **`build_cached_system_prompt()`** appends full entity vocabulary (and predicate vocabulary for relations) as a reference section after the instructions
- Makes system prompt ~1,200-2,000 tokens, crossing the 1,024 threshold
- The entity/predicate list is identical for every chunk in a run — stable prefix
- Structure: `[instructions] + [entity vocabulary] + [predicate vocabulary]` → user message with chunk text last
- Claims and relations both call `build_cached_system_prompt()` once at extraction start
- Enrichment prompts (profiles.py) are per-entity with varying context — less amenable to caching, but still benefit from client reuse

## Implementation order

1. ✅ Config fields in `models.py`
2. `llm_client.py` — shared service (core infrastructure)
3. `claims.py` — rewrite as thin consumer with async + batch
4. `relations.py` — rewrite as thin consumer with async + batch
5. `profiles.py` — rewrite enrichment to use shared service
6. `answering.py` — swap to shared Ollama client
7. `pipeline.py` — async relation extraction call
8. CLI — `--batch` flag + `collect-batch` command
9. Tests
10. Lint, typecheck, verify

## Verification

1. `pixi run lint && pixi run typecheck && pixi run test`
2. `pixi run build-graph --dry-run` — verify preview still works
3. `pixi run build-graph` — verify async claim extraction completes in minutes not hours
4. `pixi run build-graph --batch` — verify JSONL created and batch submitted
5. `pixi run collect-batch <batch_id>` — verify results collected and inserted
6. `pixi run graph-status` — verify all counts populated
7. `pixi run ingest --full` on a small test corpus — verify async relation extraction works
