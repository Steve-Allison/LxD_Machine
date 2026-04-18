"""Static DDL for the baseline LxD SQLite schema.

Responsibility:
    Holds the authoritative ``CREATE TABLE`` / ``CREATE INDEX`` statements for
    the current schema version. Consumed by :mod:`lxd.stores.schema` to
    populate fresh databases and keep legacy ones converging.

Design boundary:
    Kept as a plain string so migration code can ``executescript`` it without
    parsing or side effects. Do not add Python logic here — runtime branching
    belongs in :mod:`lxd.stores.schema` migrations.
"""

from __future__ import annotations

BASE_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS corpus_manifest (
    source_rel_path TEXT PRIMARY KEY,
    absolute_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_domain TEXT NOT NULL,
    document_id TEXT,
    blake3_hash TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    parent_source_rel_path TEXT,
    lifecycle_status TEXT NOT NULL,
    retrieval_status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    last_seen_at TEXT NOT NULL,
    last_processed_at TEXT,
    last_committed_at TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS chunk_rows (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    source_rel_path TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_domain TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    citation_label TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_occurrence INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    score_hint TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dims INTEGER NOT NULL,
    FOREIGN KEY(source_rel_path) REFERENCES corpus_manifest(source_rel_path) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS asset_links (
    asset_rel_path TEXT PRIMARY KEY,
    asset_filename TEXT NOT NULL,
    source_domain TEXT NOT NULL,
    parent_source_rel_path TEXT,
    parent_document_id TEXT,
    page_no INTEGER,
    asset_index INTEGER,
    link_method TEXT NOT NULL,
    blake3_hash TEXT NOT NULL,
    last_committed_at TEXT NOT NULL,
    FOREIGN KEY(asset_rel_path) REFERENCES corpus_manifest(source_rel_path) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS mention_rows (
    mention_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    term_source TEXT NOT NULL,
    source_domain TEXT NOT NULL,
    source_rel_path TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    surface_form TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunk_rows(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY(source_rel_path) REFERENCES corpus_manifest(source_rel_path) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_mention_rows_entity_id
ON mention_rows(entity_id);

CREATE TABLE IF NOT EXISTS ontology_sources (
    file_rel_path TEXT PRIMARY KEY,
    blake3_hash TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ontology_snapshot (
    snapshot_id TEXT PRIMARY KEY CHECK (snapshot_id = 'current'),
    ontology_root TEXT NOT NULL,
    blake3_hash TEXT NOT NULL,
    matcher_termset_hash TEXT NOT NULL,
    matcher_term_count INTEGER NOT NULL,
    source_file_count INTEGER NOT NULL,
    entity_file_count INTEGER NOT NULL,
    entity_count INTEGER NOT NULL,
    coverage_path_count INTEGER NOT NULL DEFAULT 0,
    graph_relation_count INTEGER NOT NULL DEFAULT 0,
    validation_issue_count INTEGER NOT NULL DEFAULT 0,
    validation_issues_json TEXT NOT NULL DEFAULT '[]',
    last_loaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extracted_relations (
    relation_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_rel_path TEXT NOT NULL,
    subject_entity_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_entity_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    extraction_model TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunk_rows(chunk_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_extracted_relations_subject
ON extracted_relations(subject_entity_id);

CREATE INDEX IF NOT EXISTS idx_extracted_relations_object
ON extracted_relations(object_entity_id);

CREATE TABLE IF NOT EXISTS ingest_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    files_total INTEGER NOT NULL,
    files_completed INTEGER NOT NULL,
    searchable_files_rebuilt INTEGER NOT NULL,
    asset_files_processed INTEGER NOT NULL,
    unchanged_files_skipped INTEGER NOT NULL,
    failed_files INTEGER NOT NULL,
    chunks_written INTEGER NOT NULL,
    notes TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_rel_path TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    subject_entity_id TEXT,
    object_entity_id TEXT,
    claim_type TEXT NOT NULL DEFAULT 'assertion',
    confidence REAL NOT NULL,
    extraction_model TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunk_rows(chunk_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_claims_object ON claims(object_entity_id);
CREATE INDEX IF NOT EXISTS idx_claims_chunk ON claims(chunk_id);
CREATE INDEX IF NOT EXISTS idx_claims_document ON claims(document_id);

CREATE TABLE IF NOT EXISTS entity_profiles (
    entity_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT '',
    aliases_json TEXT NOT NULL DEFAULT '[]',
    deterministic_summary TEXT NOT NULL,
    llm_summary TEXT,
    chunk_count INTEGER NOT NULL,
    doc_count INTEGER NOT NULL,
    mention_count INTEGER NOT NULL,
    claim_count INTEGER NOT NULL DEFAULT 0,
    top_predicates_json TEXT NOT NULL DEFAULT '[]',
    top_claims_json TEXT NOT NULL DEFAULT '[]',
    pagerank REAL NOT NULL DEFAULT 0.0,
    betweenness REAL NOT NULL DEFAULT 0.0,
    closeness REAL NOT NULL DEFAULT 0.0,
    in_degree INTEGER NOT NULL DEFAULT 0,
    out_degree INTEGER NOT NULL DEFAULT 0,
    eigenvector REAL NOT NULL DEFAULT 0.0,
    community_id INTEGER,
    source_hash TEXT NOT NULL,
    generated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_communities (
    entity_id TEXT PRIMARY KEY,
    community_id INTEGER NOT NULL,
    community_level INTEGER NOT NULL DEFAULT 0,
    modularity_class TEXT,
    assigned_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_communities_community_id
ON entity_communities(community_id);

CREATE TABLE IF NOT EXISTS community_reports (
    community_id INTEGER PRIMARY KEY,
    community_level INTEGER NOT NULL DEFAULT 0,
    member_count INTEGER NOT NULL,
    member_entity_ids_json TEXT NOT NULL,
    deterministic_summary TEXT NOT NULL,
    llm_summary TEXT,
    top_entities_json TEXT NOT NULL DEFAULT '[]',
    top_claims_json TEXT NOT NULL DEFAULT '[]',
    intra_community_edge_count INTEGER NOT NULL DEFAULT 0,
    source_hash TEXT NOT NULL,
    generated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relations (
    relation_id TEXT PRIMARY KEY,
    subject_entity_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_entity_id TEXT NOT NULL,
    support_count INTEGER NOT NULL DEFAULT 0,
    avg_confidence REAL NOT NULL DEFAULT 0.0,
    min_confidence REAL NOT NULL DEFAULT 0.0,
    max_confidence REAL NOT NULL DEFAULT 0.0,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_spo
ON relations(subject_entity_id, predicate, object_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_entity_id);

CREATE TABLE IF NOT EXISTS relation_evidence (
    evidence_id TEXT PRIMARY KEY,
    relation_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    surface_subject TEXT NOT NULL,
    surface_object TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    confidence REAL NOT NULL,
    extraction_model TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    FOREIGN KEY(relation_id) REFERENCES relations(relation_id) ON DELETE CASCADE,
    FOREIGN KEY(chunk_id) REFERENCES chunk_rows(chunk_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_relation_evidence_relation
ON relation_evidence(relation_id);
CREATE INDEX IF NOT EXISTS idx_relation_evidence_chunk
ON relation_evidence(chunk_id);

CREATE TABLE IF NOT EXISTS graph_build_state (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    current_phase TEXT NOT NULL DEFAULT 'pending',
    graph_version INTEGER NOT NULL,
    relations_consolidated INTEGER NOT NULL DEFAULT 0,
    evidence_rows_built INTEGER NOT NULL DEFAULT 0,
    claims_extracted INTEGER NOT NULL DEFAULT 0,
    entity_profiles_built INTEGER NOT NULL DEFAULT 0,
    communities_detected INTEGER NOT NULL DEFAULT 0,
    community_reports_built INTEGER NOT NULL DEFAULT 0,
    centrality_computed INTEGER NOT NULL DEFAULT 0,
    entity_embeddings_computed INTEGER NOT NULL DEFAULT 0,
    llm_enrichment_count INTEGER NOT NULL DEFAULT 0,
    notes_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS graph_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""
