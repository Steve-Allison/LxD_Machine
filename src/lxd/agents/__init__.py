"""Multi-step design-artefact agent (Phase 3b/c SOTA roadmap).

``lxd.agents.design.design_learning`` orchestrates a bounded step machine
(clarify → retrieve pedagogy evidence → draft objectives/modality/outline/
assessment → one critique+revise pass) on top of the same retrieval and
LLM-client primitives the answer pipeline uses. ``lxd.agents.critique``
scores an existing artefact bundle against fresh corpus evidence.
"""
