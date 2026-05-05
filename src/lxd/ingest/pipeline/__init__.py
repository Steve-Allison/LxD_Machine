"""Ingest pipeline split into purpose-scoped submodules.

Per the no-legacy rule, this package does **not** re-export anything.
Callers import from the specific submodule that owns the function:

- :mod:`lxd.ingest.pipeline.orchestrator` — `IngestPlan`, `IngestRunResult`,
  `validate_project_paths`, `build_ingest_plan`, `run_ingest`, `utc_now`.
- :mod:`lxd.ingest.pipeline.sources` — `build_source_records`,
  `build_manifest_record`.
- :mod:`lxd.ingest.pipeline.embed` — `embed_with_cache`,
  `embed_with_contextual_augmentation`, plus context-refinement and chunk
  reindexing helpers.
- :mod:`lxd.ingest.pipeline.moves` — move-detection: `find_move_source`,
  `can_skip_unchanged_source`, `resolve_document_id`, `clone_source_records`.
"""

from __future__ import annotations
