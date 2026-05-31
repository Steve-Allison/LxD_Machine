---
paths:
  - "src/lxd/settings/**/*.py"
  - "config.yaml"
  - "src/lxd/retrieval/**/*.py"
  - "src/lxd/ontology/**/*.py"
---

# Mandatory Features — No `enabled` Toggles

The codebase made a deliberate decision to remove toggles for the knowledge
graph, relation extraction, reranker, expansion, and LLM enrichment. Those
features are **mandatory**. The system degrades gracefully when prerequisite
state is missing (e.g. the graph hasn't been built yet), but the feature is
never *optionally* off.

This rule auto-loads on edits to settings, config, retrieval, or ontology.

## What is forbidden

- Do not introduce `enabled: bool` flags in `config.yaml` for any of these
  feature blocks:
    `knowledge_graph`, `relation_extraction`, `reranker`, `expansion`,
    `synthesis`, `claim_extraction`, `community_detection`,
    `llm_enrichment`, `entity_disambiguation`
- Do not introduce `backend: "none"` / `backend: "noop"` options that disable
  a feature.
- Do not reintroduce removed toggles. Git log will show that they were
  intentionally removed (commits `25b88ff`, `095f6b3` and adjacent).
- Do not add `if config.feature.enabled:` branches around feature code.

## What is allowed

- Backend choice between real options:
  `reranker.backend: llama_cpp | openai | ollama` — yes.
  `reranker.backend: none` — no.
- Graceful degradation when state is missing:
  *"the graph hasn't been built yet → fall back to graph-free baseline"* is
  a runtime data condition, not a configuration toggle.
- Sampling / quota knobs that bound the work but don't disable it
  (`max_concurrent`, `max_relations_per_chunk`, `min_relation_confidence`).

## Why

Toggles invite divergent runtime paths that the test matrix can't realistically
cover. Mandatory features mean one shape — the prod shape — is what every
contributor encounters. Removing the toggle removed an entire class of "works
on my machine" bugs (where one contributor ran with the feature off and another
with it on).

If you genuinely need to gate behaviour, gate it on **state** (is the graph
built? is the reranker reachable?) rather than on **config**.

## Anti-pattern signals to recognise

Stop and re-think if you find yourself writing any of these:

- `if config.knowledge_graph.enabled:`
- `if config.relation_extraction.backend != "none":`
- `--no-relation-extraction` CLI flag
- A new `enabled` field in a settings model
- A backend literal that includes `"none"` / `"noop"` / `"disabled"`

The pattern is wrong even when the implementation is harmless. The shape of
the config surface is what this rule protects.

## Cross-reference

- `~/.claude/rules/no-spec-invention.md` — adding an `enabled` toggle is
  inventing scope the plan didn't ask for
- `~/.claude/rules/no-defensive-coding.md` — gating on `enabled` is a form of
  defensive coding against a contract you can just enforce
- `Plans/00_PURPOSE_AND_BACKGROUND.md` — the mandatory-feature decision is
  upstream of this rule
