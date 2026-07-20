"""Pydantic models for design-agent artefacts (Phase 3b/c SOTA roadmap).

Each artefact carries the text/items it produced plus the citation labels
(``EvidenceChunk.citation_label`` values) of the corpus evidence that
grounded it. An empty ``citations`` list on a non-empty artefact is a
signal, not an oversight — it means the agent could not find grounding
evidence for that section, which callers should surface rather than
suppress.
"""

from pydantic import BaseModel, ConfigDict, Field


class LearningObjectives(BaseModel):
    """Bloom-aligned learning objective statements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: list[str] = Field(
        default_factory=list, description="Learning objective statements, one per item."
    )
    citations: list[str] = Field(
        default_factory=list, description="Citation labels grounding these objectives."
    )


class ModalityPlan(BaseModel):
    """Recommended delivery modality and rationale."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(default="", description="Recommended modality and the reasoning behind it.")
    citations: list[str] = Field(
        default_factory=list, description="Citation labels grounding this recommendation."
    )


class Outline(BaseModel):
    """Ordered module/section headings for the learning experience."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: list[str] = Field(default_factory=list, description="Ordered outline headings.")
    citations: list[str] = Field(
        default_factory=list, description="Citation labels grounding the sequencing."
    )


class AssessmentBlueprint(BaseModel):
    """Assessment items mapped to the learning objectives."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: list[str] = Field(
        default_factory=list, description="Assessment items, one per line/item."
    )
    citations: list[str] = Field(
        default_factory=list, description="Citation labels grounding the assessment design."
    )


class DesignArtefactBundle(BaseModel):
    """Full output of :func:`lxd.agents.design.design_learning`.

    ``steps_completed`` reports how many of the agent's bounded steps
    actually ran (see :mod:`lxd.agents.design`); it is always
    ``<= max_steps``. ``warnings`` accumulates every degradation the agent
    hit along the way (empty retrieval, LLM failure, max-steps cutoff) so
    a partial bundle is never silently indistinguishable from a complete
    one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str
    objectives: LearningObjectives = Field(default_factory=LearningObjectives)
    modality_plan: ModalityPlan = Field(default_factory=ModalityPlan)
    outline: Outline = Field(default_factory=Outline)
    assessment: AssessmentBlueprint = Field(default_factory=AssessmentBlueprint)
    steps_completed: int = 0
    warnings: list[str] = Field(default_factory=list)


class CritiqueResult(BaseModel):
    """Output of :func:`lxd.agents.critique.critique_design`.

    ``dimension_scores`` keys are free-text dimension labels (e.g.
    ``"objective_alignment"``, ``"evidence_grounding"``,
    ``"assessment_validity"``) mapped to a 0.0-1.0 score; ``overall_score``
    is the agent's own holistic score, not necessarily the mean of the
    dimensions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    feedback: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
