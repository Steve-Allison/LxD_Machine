"""Compute ontology key-path coverage and classification reports."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

OntologyKeyClassification = Literal[
    "graph_input",
    "matcher_input",
    "metadata_input",
]

_GRAPH_PATH_PATTERNS = (
    re.compile(r"^_meta\.relationships(?:\.|$)"),
    re.compile(r"^file_relationships(?:\.|$)"),
    re.compile(r"^entity_relations(?:\.|$)"),
    re.compile(r"^entity_relation_weights(?:\.|$)"),
    re.compile(r"^entity_types\.[^.]+\.parent_entity(?:\.|$)"),
    re.compile(r"^entity_types\.[^.]+\.relates_to(?:\.|$)"),
    re.compile(r"^entity_types\.[^.]+\.taxonomy_mapping(?:\.|$)"),
    re.compile(r"^entity_types\.[^.]+\.maps_to_taxonomy_types(?:\.|$)"),
    re.compile(r"^entity_types\.[^.]+\.taxonomy_reference(?:\.|$)"),
    re.compile(r"^entity_types\.[^.]+\.validate_against_taxonomy(?:\.|$)"),
)

_MATCHER_PATH_PATTERNS = (
    re.compile(r"^entity_types\.[^.]+\.canonical_id(?:\.|$)"),
    re.compile(r"^entity_types\.[^.]+\.aliases(?:\.|$)"),
    re.compile(r"^entity_types\.[^.]+\.indicators(?:\.|$)"),
)


@dataclass(frozen=True)
class OntologyCoverageReport:
    """Coverage report for discovered ontology key paths."""
    path_counts: dict[str, int]
    path_classifications: dict[str, OntologyKeyClassification]
    classification_counts: dict[str, int]
    unclassified_paths: list[str]

    @property
    def discovered_path_count(self) -> int:
        """Return the computed result for this operation.

        Returns:
            Number of distinct discovered key paths.
        """
        return len(self.path_counts)


def discover_key_paths(payload: Any) -> Counter[str]:
    """Discover and count all key paths in a payload.

    Args:
        payload: Ontology payload to traverse.

    Returns:
        Counter of discovered key paths.
    """
    path_counts: Counter[str] = Counter()
    _walk_value(payload, prefix="", path_counts=path_counts)
    return path_counts


def build_coverage_report(path_counts: Mapping[str, int]) -> OntologyCoverageReport:
    """Build a key-path coverage report.

    Args:
        path_counts: Discovered key-path frequency counts.

    Returns:
        Coverage report with classifications and counts.
    """
    path_classifications: dict[str, OntologyKeyClassification] = {}
    classification_counts: Counter[str] = Counter()
    for path, count in sorted(path_counts.items()):
        classification = classify_key_path(path)
        path_classifications[path] = classification
        classification_counts[classification] += count
    return OntologyCoverageReport(
        path_counts=dict(path_counts),
        path_classifications=path_classifications,
        classification_counts=dict(classification_counts),
        unclassified_paths=[],
    )


def classify_key_path(path: str) -> OntologyKeyClassification:
    """Classify an ontology key path by usage type.

    Args:
        path: Path to the source file.

    Returns:
        Classification label for the key path.
    """
    for pattern in _GRAPH_PATH_PATTERNS:
        if pattern.match(path):
            return "graph_input"
    for pattern in _MATCHER_PATH_PATTERNS:
        if pattern.match(path):
            return "matcher_input"
    return "metadata_input"


def _walk_value(value: Any, *, prefix: str, path_counts: Counter[str]) -> None:
    if prefix:
        path_counts[prefix] += 1
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = str(key) if not prefix else f"{prefix}.{key}"
            _walk_value(child, prefix=child_prefix, path_counts=path_counts)
        return
    if isinstance(value, list):
        child_prefix = "*" if not prefix else f"{prefix}.*"
        for child in value:
            _walk_value(child, prefix=child_prefix, path_counts=path_counts)
