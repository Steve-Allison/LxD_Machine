---
name: KG features are mandatory not optional
description: Never add enabled toggles for core KG features — relation extraction, claim extraction, LLM enrichment are mandatory
type: feedback
---

Never add `enabled: bool = False` toggles or `"none"` backend options for core pipeline features like relation extraction, claim extraction, knowledge graph building, or LLM enrichment. These are the core purpose of the system, not optional add-ons.

**Why:** User was furious that defensive defaults (`enabled: false`, `backend: "none"`) meant the KG was never built despite the entire system being designed as a knowledge graph. The OpenAI embedding costs were wasted building a non-KG index. Features that define what the system *is* must not have opt-out switches.

**How to apply:** When adding new pipeline phases or capabilities that are integral to the KG, make them always-on. Only use enabled toggles for genuinely optional enhancements (e.g. reranker, which depends on an external service) or features that are independent of the core purpose.
