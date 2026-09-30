"""Public and internal dataclasses for ontology loading."""

import re
from dataclasses import dataclass, field
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
class RecognitionPattern:
    """Compiled library recognition pattern for one ontology resource."""

    entity_id: str
    compiled: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class AnchorConstraint:
    """Context-anchor rule that a mention must satisfy before it is kept."""

    entity_id: str
    surface: str | None
    anchors: frozenset[str]
    window_tokens: int


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
    recognition_patterns: tuple[RecognitionPattern, ...] = ()
    suppressed_terms: dict[str, frozenset[str]] = field(default_factory=dict)
    anchor_constraints: tuple[AnchorConstraint, ...] = ()


@dataclass(frozen=True, slots=True)
class RelationSchema:
    file_relation_types: dict[str, dict[str, Any]]
    entity_relation_types: dict[str, dict[str, Any]]
    entity_relation_weights: set[str]
