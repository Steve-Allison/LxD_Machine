"""Learner / session brief carried across retrieval, synthesis, and design-agent calls.

A brief captures the optional audience/modality/Bloom-target/constraints
context an instructional designer supplies alongside a question or design
request. Every field is optional so a bare question keeps working
unchanged; when any field is set, the synthesis prompt gains a
``## Learner Brief`` section (see :mod:`lxd.synthesis.answering`) and the
design agent (:mod:`lxd.agents.design`) grounds artefacts against it.

``session_id`` ties the brief to a persisted
:class:`lxd.stores.models.SessionRecord` row so a multi-turn conversation
does not need to repeat audience/modality on every call.
"""

from pydantic import BaseModel, ConfigDict, Field


class LearnerBrief(BaseModel):
    """Optional audience/modality/Bloom-target/constraints context for a request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    audience: str | None = Field(
        default=None,
        description="Target learner audience, e.g. 'new Adobe Analytics admins'.",
    )
    modality: str | None = Field(
        default=None,
        description="Delivery modality, e.g. 'self-paced eLearning', 'ILT workshop'.",
    )
    bloom_target: str | None = Field(
        default=None,
        description="Target Bloom's taxonomy level, e.g. 'apply', 'analyze'.",
    )
    constraints: str | None = Field(
        default=None,
        description="Free-text constraints, e.g. time budget, tooling, accessibility.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID used to load/persist the brief and turn history.",
    )

    def is_empty(self) -> bool:
        """Return ``True`` when every brief field except ``session_id`` is unset."""
        return not any((self.audience, self.modality, self.bloom_target, self.constraints))

    def merge_over(self, stored: LearnerBrief) -> LearnerBrief:
        """Return a brief where this instance's set fields take precedence over ``stored``.

        Used once a ``session_id`` resolves to a persisted brief:
        request-supplied fields always win, and any field the caller left
        unset falls back to what was already on file for the session.
        """
        return LearnerBrief(
            audience=self.audience if self.audience is not None else stored.audience,
            modality=self.modality if self.modality is not None else stored.modality,
            bloom_target=(
                self.bloom_target if self.bloom_target is not None else stored.bloom_target
            ),
            constraints=self.constraints if self.constraints is not None else stored.constraints,
            session_id=self.session_id or stored.session_id,
        )
