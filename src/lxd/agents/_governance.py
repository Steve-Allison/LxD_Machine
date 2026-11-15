"""Structured ``Central_Configs`` v5.0.0 pedagogical and editorial governance for agents.

Projects authoritative v5 records from ``central.duckdb`` into compact,
deterministic prompt blocks for :mod:`lxd.agents.design` and
:mod:`lxd.agents.critique`:

- ``PedagogicalEffectRule`` (``coe:C05301``-``coe:C05308``)
- ``CrossDomainBridge`` (``coe:C09001``-``coe:C09018``)
- ``EmpiricalNorm`` (``coe:C09102``, ``coe:C09120``, ``coe:C09201``-``coe:C09210``)
- ``Rubric`` (``coe:C09501``, ``coe:C09502``, ``coe:C09505``)
- ``DiagnosticDetector`` (``coe:C09601``-``coe:C09608``)
- ``VoiceRegister`` (``coe:C08504``, ``coe:C08505``)
"""

from pathlib import Path
from typing import Final

import structlog

from lxd.domain.brief import LearnerBrief
from lxd.ontology.loader import (
    CrossDomainBridgeRecord,
    DiagnosticDetectorRecord,
    EmpiricalNormRecord,
    NormBandRecord,
    OntologyLoadResult,
    PedagogicalEffectRuleRecord,
    RubricRecord,
    VoiceRegisterRecord,
    load_ontology,
    resolve_central_duckdb_path,
)
from lxd.settings.models import RuntimeConfig

_log = structlog.get_logger(__name__)

PEDAGOGICAL_EFFECT_RULE_IDS: Final[frozenset[str]] = frozenset(
    f"coe:C0530{i}" for i in range(1, 9)
)
CROSS_DOMAIN_BRIDGE_IDS: Final[frozenset[str]] = frozenset(
    f"coe:C090{i:02d}" for i in range(1, 19)
)
EMPIRICAL_NORM_IDS: Final[frozenset[str]] = frozenset(
    {
        "coe:C09102",
        "coe:C09120",
        *(f"coe:C092{i:02d}" for i in range(1, 11)),
    }
)
RUBRIC_IDS: Final[frozenset[str]] = frozenset({"coe:C09501", "coe:C09502", "coe:C09505"})
DIAGNOSTIC_DETECTOR_IDS: Final[frozenset[str]] = frozenset(
    f"coe:C0960{i}" for i in range(1, 9)
)
VOICE_REGISTER_IDS: Final[frozenset[str]] = frozenset({"coe:C08504", "coe:C08505"})

_EXECUTIVE_HINTS: Final[tuple[str, ...]] = (
    "executive",
    "vp",
    "c-suite",
    "cmo",
    "cio",
    "cdo",
    "director",
    "leadership",
    "advisory",
)


def load_agent_ontology(config: RuntimeConfig) -> OntologyLoadResult | None:
    """Load ``Central_Configs`` v5.0.0 when ``config.paths`` points to a built DuckDB."""
    paths = getattr(config, "paths", None)
    if paths is None:
        return None
    db_path_attr = getattr(paths, "central_duckdb_path", None)
    if db_path_attr is not None:
        db_path = Path(db_path_attr)
    else:
        central_root = getattr(paths, "central_configs_path", None)
        if central_root is None:
            return None
        db_path = resolve_central_duckdb_path(Path(central_root))
    if not db_path.is_file():
        return None
    try:
        return load_ontology(db_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        _log.warning("agent_governance_load_failed", error=str(exc))
        return None


def format_design_governance_block(
    config: RuntimeConfig,
    brief: LearnerBrief | None = None,
) -> str:
    """Format structured ``Central_Configs`` v5.0.0 design rules, norms, bridges, and voice."""
    ontology = load_agent_ontology(config)
    if ontology is None:
        return ""

    sections: list[str] = []
    rules = _select_by_id(ontology.pedagogical_effect_rules, PEDAGOGICAL_EFFECT_RULE_IDS)
    if rules:
        sections.append(_format_pedagogical_rules(rules, brief=brief))

    bridges = _select_by_id(ontology.cross_domain_bridges, CROSS_DOMAIN_BRIDGE_IDS)
    if bridges:
        sections.append(_format_bridges(bridges))

    norms = _select_by_id(ontology.empirical_norms, EMPIRICAL_NORM_IDS)
    if norms:
        sections.append(_format_norms(norms))

    rubrics = _select_by_id(ontology.rubrics, RUBRIC_IDS)
    if rubrics:
        sections.append(_format_rubrics(rubrics))

    detectors = _select_by_id(ontology.diagnostic_detectors, DIAGNOSTIC_DETECTOR_IDS)
    if detectors:
        sections.append(_format_detectors(detectors))

    voices = _select_by_id(ontology.voice_registers, VOICE_REGISTER_IDS)
    if voices:
        sections.append(_format_voices(voices, brief=brief))

    return "\n\n".join(section for section in sections if section)


def format_critique_governance_block(config: RuntimeConfig) -> str:
    """Format structured ``Central_Configs`` v5.0.0 rubrics, detectors, norms, and rules."""
    ontology = load_agent_ontology(config)
    if ontology is None:
        return ""

    sections: list[str] = []
    rubrics = _select_by_id(ontology.rubrics, RUBRIC_IDS)
    if rubrics:
        sections.append(_format_rubrics(rubrics))

    detectors = _select_by_id(ontology.diagnostic_detectors, DIAGNOSTIC_DETECTOR_IDS)
    if detectors:
        sections.append(_format_detectors(detectors))

    rules = _select_by_id(ontology.pedagogical_effect_rules, PEDAGOGICAL_EFFECT_RULE_IDS)
    if rules:
        sections.append(_format_pedagogical_rules(rules, brief=None))

    bridges = _select_by_id(ontology.cross_domain_bridges, CROSS_DOMAIN_BRIDGE_IDS)
    if bridges:
        sections.append(_format_bridges(bridges))

    norms = _select_by_id(ontology.empirical_norms, EMPIRICAL_NORM_IDS)
    if norms:
        sections.append(_format_norms(norms))

    voices = _select_by_id(ontology.voice_registers, VOICE_REGISTER_IDS)
    if voices:
        sections.append(_format_voices(voices, brief=None))

    return "\n\n".join(section for section in sections if section)


def _select_by_id[
    T: (
        PedagogicalEffectRuleRecord,
        CrossDomainBridgeRecord,
        EmpiricalNormRecord,
        RubricRecord,
        DiagnosticDetectorRecord,
        VoiceRegisterRecord,
    )
](records: tuple[T, ...], target_ids: frozenset[str]) -> list[T]:
    selected = [record for record in records if record.id in target_ids]
    return selected if selected else list(records)


def _format_pedagogical_rules(
    rules: list[PedagogicalEffectRuleRecord],
    *,
    brief: LearnerBrief | None,
) -> str:
    brief_text = _brief_haystack(brief)
    lines = ["### Pedagogical Effect & Boundary Rules (Expertise Reversal & L2)"]
    for rule in rules:
        conds: list[str] = []
        if rule.conditioned_expertise_tier:
            conds.append(f"expertise={rule.conditioned_expertise_tier}")
        if rule.conditioned_language_baseline:
            conds.append(f"language={rule.conditioned_language_baseline}")
        if rule.effect_polarity:
            conds.append(f"polarity={rule.effect_polarity}")
        cond_str = f" ({', '.join(conds)})" if conds else ""
        active = _rule_matches_brief(rule, brief_text)
        prefix = "[ACTIVE FOR BRIEF] " if active else ""
        lines.append(
            f"- {prefix}`{rule.id}` **{rule.pref_label}**{cond_str}: {rule.adapted_prescription}"
        )
    return "\n".join(lines)


def _rule_matches_brief(rule: PedagogicalEffectRuleRecord, brief_text: str) -> bool:
    if not brief_text:
        return False
    if rule.conditioned_expertise_tier and rule.conditioned_expertise_tier.lower() in brief_text:
        return True
    baseline = (rule.conditioned_language_baseline or "").lower()
    return bool(
        baseline
        and (
            baseline in brief_text
            or ("l2" in brief_text and "l2" in baseline)
            or ("global" in brief_text and "l2" in baseline)
        )
    )


def _format_bridges(bridges: list[CrossDomainBridgeRecord]) -> str:
    lines = ["### Cross-Domain Operational Bridges"]
    for bridge in bridges:
        lines.append(
            f"- `{bridge.id}` **{bridge.pref_label}** [{bridge.bridge_kind}]: "
            f"{bridge.actionable_guidance}"
        )
    return "\n".join(lines)


def _format_norms(norms: list[EmpiricalNormRecord]) -> str:
    lines = ["### Calibrated Empirical Norms"]
    for norm in norms:
        band_summary = _summarize_bands(norm.bands)
        lines.append(f"- `{norm.id}` **{norm.pref_label}**: {band_summary}")
    return "\n".join(lines)


def _summarize_bands(bands: tuple[NormBandRecord, ...]) -> str:
    parts: list[str] = []
    for band in bands:
        bounds = _format_bounds(band.lower_bound, band.upper_bound)
        segment = f"{band.zone}={bounds}" if bounds else band.zone
        if band.coaching_feedback and ("optimal" in band.zone or "target" in band.zone):
            segment = f"{segment} ({band.coaching_feedback})"
        parts.append(segment)
    return "; ".join(parts)


def _format_bounds(lower: float | None, upper: float | None) -> str:
    if lower is not None and upper is not None:
        return f"{lower:g}–{upper:g}"
    if lower is not None:
        return f"≥{lower:g}"
    if upper is not None:
        return f"≤{upper:g}"
    return ""


def _format_rubrics(rubrics: list[RubricRecord]) -> str:
    lines = ["### Analytic Quality Rubrics"]
    for rubric in rubrics:
        crit_parts = [
            f"{c.pref_label} ({c.criterion_weight:.0%})"
            if c.criterion_weight is not None
            else c.pref_label
            for c in rubric.criteria
        ]
        criteria_text = ", ".join(crit_parts) if crit_parts else rubric.definition
        lines.append(
            f"- `{rubric.id}` **{rubric.pref_label}** [{rubric.evaluated_artifact_kind}]: "
            f"{criteria_text}"
        )
    return "\n".join(lines)


def _format_detectors(detectors: list[DiagnosticDetectorRecord]) -> str:
    lines = ["### Diagnostic Failure-Mode Detectors"]
    for det in detectors:
        lines.append(
            f"- `{det.id}` **{det.pref_label}** — Signal: {det.detection_signal_description} "
            f"Remediation: {det.remediation_instruction}"
        )
    return "\n".join(lines)


def _format_voices(
    voices: list[VoiceRegisterRecord],
    *,
    brief: LearnerBrief | None,
) -> str:
    brief_text = _brief_haystack(brief)
    prefer_exec = any(hint in brief_text for hint in _EXECUTIVE_HINTS)
    ordered = sorted(
        voices,
        key=lambda v: (
            0
            if (prefer_exec and v.id == "coe:C08505")
            or (not prefer_exec and v.id == "coe:C08504")
            else 1,
            v.id,
        ),
    )
    lines = ["### Editorial Voice Registers"]
    for index, voice in enumerate(ordered):
        role_tag = "[PRIMARY VOICE] " if index == 0 and brief is not None else ""
        traits = "; ".join(voice.register_traits[:3])
        avoid = "; ".join(voice.prohibited_anti_patterns[:3])
        lines.append(
            f"- {role_tag}`{voice.id}` **{voice.pref_label}** ({voice.default_locale}): "
            f"Traits: {traits}. Avoid: {avoid}."
        )
    return "\n".join(lines)


def _brief_haystack(brief: LearnerBrief | None) -> str:
    if brief is None:
        return ""
    parts = [
        brief.audience or "",
        brief.modality or "",
        brief.bloom_target or "",
        brief.constraints or "",
    ]
    return " ".join(parts).lower()
