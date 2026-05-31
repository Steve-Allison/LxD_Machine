---
paths:
  - "src/lxd/retrieval/**/*.py"
  - "src/lxd/synthesis/**/*.py"
  - "src/lxd/ingest/claims.py"
  - "src/lxd/ingest/relations.py"
  - "src/lxd/ingest/wiki_metadata.py"
  - "src/lxd/ingest/wiki_relations.py"
---

# Citations and Evidence — Extract, Don't Invent

LxD's contract with its users is **evidence-first**. Every chunk, mention,
claim, relation, and synthesised answer traces back to a real source. Auto-loads
when editing retrieval, synthesis, or any extraction module.

## The four invariants

### 1. Every chunk has provenance

Each `ChunkRecord` carries:

- `source_rel_path` — the file the text came from
- `source_hash` — the BLAKE3 of the source's content at ingest time
- `chunk_index` / `chunk_occurrence` — position within the source
- `citation_label` — the client-facing handle, format
  `<source_rel_path>#<chunk_index>`

Wiki pages additionally carry transitive `**Sources**:` citations parsed at
ingest time (`wiki_metadata.py`). These appear as `cited_sources` on every
chunk derived from a wiki page so synthesis can attribute back to the
original research.

If you write code that constructs a chunk without one of these fields, the
contract is broken.

### 2. Every claim and relation traces to a chunk

`ClaimRecord` and `extracted_relations` rows carry `chunk_id` —
the chunk the LLM was looking at when it produced the claim/relation.
This is the audit trail for `inspect_evidence`.

Never:

- Construct a claim without `chunk_id`.
- Aggregate claims across chunks in a way that loses the per-chunk attribution.
- Synthesise a "summary claim" not literally extracted from any single chunk.

### 3. Synthesis cites every assertion

`synthesis/answering.py` builds prompts that demand cited answers. The
`citations` list in the response references chunks by `citation_label`. If
synthesis emits an answer with content not traceable to a cited chunk, that
is a bug — not "the LLM hallucinated" but "we asked it incorrectly".

If you change the synthesis preamble or the prompt structure, verify that:

- The preamble still requires citations for every claim
- Uncitable answers route to `insufficient_evidence` status, not a fabricated
  answer

### 4. Never invent provenance

If a source field is missing or can't be resolved, the right answer is:

- For ingest: raise a typed exception, log structured error, and let the
  systemic-error circuit breaker decide whether to abort
- For retrieval: skip the chunk; do not synthesise a missing field
- For synthesis: route to `no_results` / `insufficient_evidence`; do not
  fabricate a citation_label that doesn't exist

There is NO acceptable "use a placeholder" or "use 'unknown'" path. Missing
provenance is data integrity loss.

## Anti-pattern signals to recognise

You are violating this rule if you write any of:

- `citation_label="unknown"` / `citation_label=""` defaults
- `source_rel_path or "untracked"` fallbacks
- `citations.append(f"chunk_{i}")` for synthetic labels
- Aggregating claims from multiple chunks without preserving each chunk's
  `chunk_id`
- Synthesis paths that emit `answer_status="answered"` without entries in
  `citations`
- "If we can't find the source, just use the closest match" reasoning

## Cross-reference

- `~/.claude/CLAUDE.md` §1 (Verify, Don't Assume) — citations are the verify
  step made structural
- `~/.claude/CLAUDE.md` §6 (Report Honestly) — same rule, applied to answer
  generation
- `Plans/04_QUERY_SPEC.md` — citation contract in the query pipeline
- `Plans/08_KNOWLEDGE_GRAPH_SPEC.md` — claim/relation provenance contract
- `.claude/rules/stores-and-paths.md` — corpus-relative paths underpin all
  citation labels
