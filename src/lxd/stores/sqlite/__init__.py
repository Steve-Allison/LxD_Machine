"""SQLite store layer split into purpose-scoped submodules.

Per the no-legacy rule, this package does **not** re-export anything.
Callers import from the specific submodule that owns the function:

- :mod:`lxd.stores.sqlite.connection` — connect, store paths, schema init.
- :mod:`lxd.stores.sqlite.runs` — ingest run lifecycle rows.
- :mod:`lxd.stores.sqlite.manifest` — corpus_manifest + asset_links + delete_source.
- :mod:`lxd.stores.sqlite.ontology` — ontology_sources + snapshot + ingest_config + committed-state probe.
- :mod:`lxd.stores.sqlite.chunks` — chunk_rows, mention_rows, extracted_relations, centrality signals.
- :mod:`lxd.stores.sqlite.summary` — corpus / chunk / mention summary counters.
- :mod:`lxd.stores.sqlite.claims` — claims insert/load/count.
- :mod:`lxd.stores.sqlite.kg_profiles` — entity_profiles, entity_communities, community_reports.
- :mod:`lxd.stores.sqlite.kg_relations` — canonical relations, relation_evidence, graph_build_state, graph_metadata.
"""
