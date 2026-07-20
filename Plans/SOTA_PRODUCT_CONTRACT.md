# SOTA Product Contract

Success for the SOTA Full Vision is two complementary outcomes, both cited and provenance-preserving.

## Outcome A — Cited pedagogical answers (baseline RAG+)

Given a natural-language question about instructional design / learning experience design:

1. Retrieve ontology-expanded, optionally multi-query / HyDE-augmented evidence.
2. Rank with dense + rerank + relation + centrality + **graph lane** (claims / community reports as candidates).
3. Surface **disagreements** when claims conflict; never silently pick a winner.
4. Synthesise a cited answer (`[citation_label]` + transitive wiki sources where present).

Acceptance: retrieval-check and eval-quality ≥ frozen baseline in `Plans/SOTA_BASELINE.md`; design subset shows clear lift after Phase 1.

## Outcome B — Multi-step design artefacts (agentic ID layer)

Given a design brief (audience, modality, Bloom target, constraints):

1. Produce structured artefacts with citations: learning objectives → modality/sequence plan → outline → assessment blueprint.
2. Self-critique once against corpus evidence; revise once under hard step/timeout caps.
3. Honour `session_id` learner brief across turns (structured state, not raw chat dump).

Acceptance: integration/golden test for a fixed brief returns typed artefacts with citations; second turn with the same `session_id` respects audience/modality; MCP remains the only external interface.

## Non-goals (unchanged)

- Auto-editing wiki/ontology without human review.
- Cloud multi-tenant SaaS or replacing MCP.
- Replacing ontology-first entity recognition with pure LLM NER.
- Neo4j migration.
