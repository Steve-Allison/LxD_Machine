"""Pydantic schema models for ontology YAML files.

Responsibility:
    Provide a Pydantic v2 surface that validates the top-level shape of an
    ontology YAML file. The existing :mod:`lxd.ontology.loader` walks the raw
    payload with hand-rolled type checks because ontology files evolve faster
    than their models; these lightweight models complement that loader by
    catching schema errors earlier (and with clearer messages) when callers
    opt in via :func:`validate_ontology_file`.

Design boundary:
    The models intentionally permit unknown keys (``extra="allow"``) so they
    do not block valid-but-unmodelled ontology shapes during the transition.
    The goal is *observability*, not gatekeeping — validation errors surface
    as :class:`ValueError` and should be logged, not re-raised as fatal.

Key constraints:
    * Pydantic v2 + ``model_config = ConfigDict(extra="allow")``.
    * No I/O: callers are responsible for loading YAML into plain ``dict``
      payloads before invoking the validator.
    * Module must stay import-cheap; no heavy dependencies beyond Pydantic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class OntologyMetaModel(BaseModel):
    """Top-level ``_meta`` block for an ontology file."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    title: str | None = None
    purpose: str | None = None
    domain: str | None = None
    domain_type: str | None = None
    relationships: list[dict[str, Any]] | None = None


class OntologyEntityModel(BaseModel):
    """A single entity_types entry (intentionally permissive)."""

    model_config = ConfigDict(extra="allow")

    label: str | None = None
    gliner_label: str | None = None
    entity_kind: str | None = None
    family: str | None = None
    parent_entity: str | None = None
    relates_to: list[Any] | None = None
    taxonomy_mapping: list[dict[str, Any]] | None = None
    maps_to_taxonomy_types: list[str] | None = None
    taxonomy_reference: str | None = None
    validate_against_taxonomy: bool | None = None


class OntologyFileModel(BaseModel):
    """Validated view of a parsed ontology YAML document.

    Attributes:
        meta: Optional ``_meta`` block (title, domain, relationships...).
        entity_types: Mapping from canonical entity id to its definition.
        file_relationships: Optional top-level relation-type vocabulary.
        entity_relations: Optional entity-relation-type vocabulary.
        entity_relation_weights: Optional set of valid weight names.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    meta: OntologyMetaModel | None = None
    entity_types: dict[str, OntologyEntityModel] | None = None
    file_relationships: dict[str, dict[str, Any]] | None = None
    entity_relations: dict[str, dict[str, Any]] | None = None
    entity_relation_weights: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> OntologyFileModel:
        """Build an :class:`OntologyFileModel` from a raw YAML payload.

        Args:
            payload: Result of ``yaml.safe_load`` for an ontology file.
                Must be a mapping; any other shape is rejected.

        Returns:
            Validated model.

        Raises:
            ValueError: If the payload is not a mapping or fails validation.
        """
        if not isinstance(payload, dict):
            raise ValueError("ontology payload must be a mapping")
        prepared: dict[str, Any] = dict(payload)
        if "_meta" in prepared:
            prepared["meta"] = prepared.pop("_meta")
        try:
            return cls.model_validate(prepared)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc


def validate_ontology_file(payload: Any) -> OntologyFileModel:
    """Validate an ontology file payload and return the model.

    Thin wrapper around :meth:`OntologyFileModel.from_payload` for ergonomic
    use from call sites that do not want to import the class directly.
    """
    return OntologyFileModel.from_payload(payload)
