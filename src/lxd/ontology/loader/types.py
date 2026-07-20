"""Public and internal dataclasses for ontology loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxd.ontology.graph import RelationRecord
from lxd.ontology.inventory import OntologyCoverageReport
from lxd.ontology.matcher import MatcherTermRecord


@dataclass(frozen=True, slots=True)
class OntologySource:
    """Loaded ontology source file and parsed payload."""

    file_path: Path
    file_rel_path: str
    blake3_hash: str
    data: Any


@dataclass(frozen=True, slots=True)
class OntologyMetadataRecord:
    """Ontology metadata row derived from file or entity payload."""

    record_kind: str
    source_file_rel_path: str
    entity_id: str | None
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OntologyValidationIssue:
    """Validation issue found during ontology loading."""

    issue_kind: str
    source_file_rel_path: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class OntologyLoadResult:
    """All artifacts produced by ontology loading."""

    sources: list[OntologySource]
    entity_definitions: list[dict[str, Any]]
    matcher_records: list[MatcherTermRecord]
    matcher_termset_hash: str
    snapshot_hash: str
    relation_records: list[RelationRecord]
    metadata_records: list[OntologyMetadataRecord]
    coverage_report: OntologyCoverageReport
    validation_issues: list[OntologyValidationIssue]
    graph: Any


@dataclass(frozen=True, slots=True)
class RelationSchema:
    file_relation_types: dict[str, dict[str, Any]]
    entity_relation_types: dict[str, dict[str, Any]]
    entity_relation_weights: set[str]
