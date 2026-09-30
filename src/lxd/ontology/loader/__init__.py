"""Load ontology data and validate source metadata."""

from pathlib import Path

from lxd.ontology.graph import build_graph
from lxd.ontology.loader.entities import (
    build_node_records,
    extract_entity_definitions,
    extract_metadata_records,
)
from lxd.ontology.loader.relations import extract_relation_schema, extract_relations
from lxd.ontology.loader.sources import (
    coverage_report_for_sources,
    load_sources,
)
from lxd.ontology.loader.sources import (
    snapshot_hash as compute_snapshot_hash,
)
from lxd.ontology.loader.types import (
    OntologyLoadResult,
    OntologyMetadataRecord,
    OntologySource,
    OntologyValidationIssue,
)
from lxd.ontology.matcher import canonical_matcher_term_records, matcher_termset_hash
from lxd.settings.models import RuntimeConfig

__all__ = [
    "OntologyLoadResult",
    "OntologyMetadataRecord",
    "OntologySource",
    "OntologyValidationIssue",
    "load_configured_ontology",
    "load_ontology",
]

_ONTOLOGY_CACHE: dict[tuple[str, str, tuple[str, ...], tuple[str, ...]], OntologyLoadResult] = {}


def load_configured_ontology(config: RuntimeConfig) -> OntologyLoadResult:
    """Load the ontology selected by runtime config, caching the result.

    Args:
        config: Validated runtime configuration.

    Returns:
        Loaded ontology. A Central distribution is projected with its library.
    """
    library_path = config.paths.library_path
    cache_key = (
        str(config.paths.ontology_path),
        str(library_path) if library_path is not None else "",
        tuple(config.ontology.include_globs),
        tuple(config.ontology.ignore_names),
    )
    cached = _ONTOLOGY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    loaded = load_ontology(
        config.paths.ontology_path,
        config.ontology.include_globs,
        config.ontology.ignore_names,
        library_root=library_path,
    )
    _ONTOLOGY_CACHE[cache_key] = loaded
    return loaded


def load_ontology(
    root: Path,
    include_globs: list[str],
    ignore_names: list[str],
    library_root: Path | None = None,
) -> OntologyLoadResult:
    """Load ontology sources and derive runtime artifacts.

    Args:
        root: Ontology root directory.
        include_globs: Glob patterns selecting ontology files.
        ignore_names: Filenames to ignore while loading.
        library_root: Central library root used when ``root`` is a distribution.

    Returns:
        Loaded ontology artifacts and derived indexes.
    """
    # Imported here because central_projection imports this package's submodules.
    # A top-level import cycles when central_projection is imported first.
    from lxd.ontology.central_projection import is_central_distribution, load_central_ontology

    if is_central_distribution(root):
        return load_central_ontology(root, library_root, ignore_names)
    sources = load_sources(root, include_globs, ignore_names)
    coverage_report = coverage_report_for_sources(sources)
    entity_definitions = extract_entity_definitions(sources)
    matcher_records = canonical_matcher_term_records(entity_definitions)
    snapshot_hash = compute_snapshot_hash(sources)
    metadata_records = extract_metadata_records(sources, entity_definitions)
    relation_schema = extract_relation_schema(sources)
    relation_records, validation_issues = extract_relations(
        sources, entity_definitions, relation_schema
    )
    graph = build_graph(
        build_node_records(sources, entity_definitions, relation_records),
        relation_records,
    )
    return OntologyLoadResult(
        sources=sources,
        entity_definitions=entity_definitions,
        matcher_records=matcher_records,
        matcher_termset_hash=matcher_termset_hash(matcher_records),
        snapshot_hash=snapshot_hash,
        relation_records=relation_records,
        metadata_records=metadata_records,
        coverage_report=coverage_report,
        validation_issues=validation_issues,
        graph=graph,
    )
