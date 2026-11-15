"""Load ``Central_Configs`` v5.0.0 semantic authority from ``central.duckdb``."""

from functools import lru_cache
from pathlib import Path

from lxd.ontology.central_projection import load_central_ontology, resolve_central_duckdb_path
from lxd.ontology.loader.types import (
    CrossDomainBridgeRecord,
    DiagnosticDetectorRecord,
    EditorialRuleRecord,
    EmpiricalNormRecord,
    LearnerArchetypeRecord,
    NormBandRecord,
    OntologyCoverageReport,
    OntologyLoadResult,
    OntologyMetadataRecord,
    OntologySource,
    OntologyTermRecord,
    OntologyValidationIssue,
    PedagogicalEffectRuleRecord,
    RubricCriterionRecord,
    RubricRecord,
    VoiceRegisterRecord,
)
from lxd.settings.models import RuntimeConfig

__all__ = [
    "CrossDomainBridgeRecord",
    "DiagnosticDetectorRecord",
    "EditorialRuleRecord",
    "EmpiricalNormRecord",
    "LearnerArchetypeRecord",
    "NormBandRecord",
    "OntologyCoverageReport",
    "OntologyLoadResult",
    "OntologyMetadataRecord",
    "OntologySource",
    "OntologyTermRecord",
    "OntologyValidationIssue",
    "PedagogicalEffectRuleRecord",
    "RubricCriterionRecord",
    "RubricRecord",
    "VoiceRegisterRecord",
    "load_configured_ontology",
    "load_ontology",
    "resolve_central_duckdb_path",
]


@lru_cache(maxsize=4)
def _load_ontology_cached(db_path_str: str, mtime_ns: int) -> OntologyLoadResult:
    _ = mtime_ns
    return load_central_ontology(Path(db_path_str))


def load_configured_ontology(config: RuntimeConfig) -> OntologyLoadResult:
    """Load the ``Central_Configs`` v5.0.0 authority selected by ``config.paths``."""
    db_path = config.paths.central_duckdb_path
    mtime_ns = db_path.stat().st_mtime_ns if db_path.is_file() else -1
    return _load_ontology_cached(str(db_path), mtime_ns)


def load_ontology(root_or_db: Path) -> OntologyLoadResult:
    """Load ``Central_Configs`` v5.0.0 from a repository root or ``central.duckdb`` path."""
    return load_central_ontology(root_or_db)
