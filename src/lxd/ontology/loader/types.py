"""Strongly typed dataclasses for ``Central_Configs`` v5.0.0 DuckDB ontology loading."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxd.ontology.graph import OntologyGraph, RelationRecord
from lxd.ontology.matcher import LexiconRule


@dataclass(frozen=True, slots=True)
class OntologySource:
    """Compiled authority store descriptor and BLAKE3 digest."""

    file_path: Path
    file_rel_path: str
    blake3_hash: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OntologyMetadataRecord:
    """Entity or collection metadata row projected from ``central.duckdb``."""

    record_kind: str
    source_file_rel_path: str
    entity_id: str | None
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OntologyValidationIssue:
    """Integrity or referential issue discovered while projecting ``central.duckdb``."""

    issue_kind: str
    source_file_rel_path: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class OntologyCoverageReport:
    """Table-level population summary for ``central.duckdb``."""

    table_counts: dict[str, int]
    unclassified_paths: list[str] = field(default_factory=list)

    @property
    def discovered_path_count(self) -> int:
        """Return count of populated authority tables."""
        return sum(1 for count in self.table_counts.values() if count > 0)


@dataclass(frozen=True, slots=True)
class OntologyTermRecord:
    """Controlled lexical term row from ``all_terms``."""

    resource_id: str
    pref_label: str
    collection_name: str
    written_form: str
    term_kind: str
    term_status: str | None = None
    language: str | None = None
    ipa_phonetic: str | None = None
    spoken_form: str | None = None
    usage_note: str | None = None


@dataclass(frozen=True, slots=True)
class PedagogicalEffectRuleRecord:
    """Conditional pedagogical boundary/reversal rule (``coe:C05301``-``coe:C05308``)."""

    id: str
    pref_label: str
    definition: str
    governed_principle_or_method: str
    conditioned_expertise_tier: str | None = None
    conditioned_language_baseline: str | None = None
    modality_or_task_condition: str = ""
    effect_polarity: str = ""
    adapted_prescription: str = ""
    evidence_grade: str | None = None
    citations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CrossDomainBridgeRecord:
    """Cross-domain operational bridge (``coe:C09001``-``coe:C09018``)."""

    id: str
    pref_label: str
    definition: str
    source_concept: str
    target_concept: str
    bridge_kind: str
    applicability_condition: str = ""
    actionable_guidance: str = ""
    evidence_grade: str | None = None
    citations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NormBandRecord:
    """Calibrated measurement zone inside an ``EmpiricalNormRecord``."""

    zone: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    coaching_feedback: str = ""


@dataclass(frozen=True, slots=True)
class EmpiricalNormRecord:
    """Calibrated quantitative delivery/instructional norm (``empirical_norms``)."""

    id: str
    pref_label: str
    definition: str
    measured_subject: str
    quantity_kind_uri: str = ""
    unit_uri: str = ""
    aggregation_window: str | None = None
    applicable_delivery_genres: tuple[str, ...] = ()
    applicable_language_baseline: str | None = None
    bands: tuple[NormBandRecord, ...] = ()
    evidence_grade: str | None = None
    wiki_slug: str | None = None


@dataclass(frozen=True, slots=True)
class RubricCriterionRecord:
    """Analytic evaluation criterion inside a ``RubricRecord``."""

    pref_label: str
    definition: str
    criterion_weight: float | None = None
    linked_concepts: tuple[str, ...] = ()
    levels: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class RubricRecord:
    """Analytic quality rubric (``coe:C09501``-``coe:C09505``)."""

    id: str
    pref_label: str
    definition: str
    evaluated_artifact_kind: str
    criteria: tuple[RubricCriterionRecord, ...] = ()
    wiki_slug: str | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticDetectorRecord:
    """Failure-mode diagnostic detector and remediation rule (``coe:C09601``-``coe:C09608``)."""

    id: str
    pref_label: str
    definition: str
    linked_concepts: tuple[str, ...] = ()
    detection_signal_description: str = ""
    remediation_instruction: str = ""
    wiki_slug: str | None = None


@dataclass(frozen=True, slots=True)
class VoiceRegisterRecord:
    """Editorial voice register specification (``coe:C08501``-``coe:C08505``)."""

    id: str
    pref_label: str
    definition: str
    default_locale: str = "en-US"
    target_personas: tuple[str, ...] = ()
    register_traits: tuple[str, ...] = ()
    preferred_syntactic_patterns: tuple[str, ...] = ()
    prohibited_anti_patterns: tuple[str, ...] = ()
    wiki_slug: str | None = None


@dataclass(frozen=True, slots=True)
class LearnerArchetypeRecord:
    """Field/practitioner learner archetype (``coe:C03201``-``coe:C03217``)."""

    id: str
    pref_label: str
    definition: str
    primary_offerings_used: tuple[str, ...] = ()
    primary_goals: tuple[str, ...] = ()
    core_frustrations: tuple[str, ...] = ()
    expertise_tier: str | None = None
    language_baseline: str | None = None
    typical_time_budget_minutes: int | None = None
    wiki_slug: str | None = None


@dataclass(frozen=True, slots=True)
class EditorialRuleRecord:
    """Plain-language and editorial rule (``coe:C08601``-``coe:C08610``)."""

    id: str
    pref_label: str
    definition: str
    editorial_directive: str = ""
    illustrative_before_after: str = ""
    evidence_grade: str | None = None
    wiki_slug: str | None = None


@dataclass(frozen=True, slots=True)
class OntologyLoadResult:
    """Projected runtime semantic authority loaded from ``central.duckdb``."""

    db_path: Path
    sources: list[OntologySource]
    entity_definitions: list[dict[str, Any]]
    terms: tuple[OntologyTermRecord, ...]
    matcher_records: list[LexiconRule]
    matcher_termset_hash: str
    snapshot_hash: str
    relation_records: list[RelationRecord]
    metadata_records: list[OntologyMetadataRecord]
    coverage_report: OntologyCoverageReport
    validation_issues: list[OntologyValidationIssue]
    graph: OntologyGraph
    pedagogical_effect_rules: tuple[PedagogicalEffectRuleRecord, ...] = ()
    cross_domain_bridges: tuple[CrossDomainBridgeRecord, ...] = ()
    empirical_norms: tuple[EmpiricalNormRecord, ...] = ()
    rubrics: tuple[RubricRecord, ...] = ()
    diagnostic_detectors: tuple[DiagnosticDetectorRecord, ...] = ()
    voice_registers: tuple[VoiceRegisterRecord, ...] = ()
    learner_archetypes: tuple[LearnerArchetypeRecord, ...] = ()
    editorial_rules: tuple[EditorialRuleRecord, ...] = ()
