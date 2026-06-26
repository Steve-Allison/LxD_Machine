"""Tests for the embedding-based entity expansion lane (B-KG-3)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from lxd.retrieval import query_pipeline as _query_pipeline
from lxd.retrieval.expansion import ExpansionOutcome
from lxd.settings.models import RuntimeConfig
from lxd.stores.lancedb import (
    connect_lancedb,
    open_entity_table,
    replace_entity_embeddings,
)
from lxd.stores.models import StorePaths

_augment_with_embedding_neighbours = (
    _query_pipeline._augment_with_embedding_neighbours  # pyright: ignore[reportPrivateUsage]
)


def _config(*, embed_dims: int, max_terms: int) -> RuntimeConfig:
    return cast(
        "RuntimeConfig",
        SimpleNamespace(
            models=SimpleNamespace(embed_dims=embed_dims),
            expansion=SimpleNamespace(max_terms=max_terms),
        ),
    )


def _store_paths(tmp_path: Path) -> StorePaths:
    return StorePaths(sqlite_path=tmp_path / "x.sqlite", lancedb_path=tmp_path / "lance")


def test_augment_no_ops_when_lancedb_store_does_not_exist(tmp_path: Path) -> None:
    """No KG built yet → helper returns the expansion unchanged, no error."""
    expansion = ExpansionOutcome(
        expanded_question="what is X?",
        matched_entity_ids=["surface_match"],
        added_terms=[],
    )

    result = _augment_with_embedding_neighbours(
        expansion=expansion,
        query_vector=[0.0, 0.0, 0.0, 0.0],
        store_paths=_store_paths(tmp_path),
        config=_config(embed_dims=4, max_terms=5),
    )

    assert result is expansion or result.matched_entity_ids == ["surface_match"]


def test_augment_merges_nearest_entities_into_matched_ids(tmp_path: Path) -> None:
    """Helper merges the nearest entity_ids from entity_embeddings into matched_entity_ids."""
    embed_dims = 4
    store_paths = _store_paths(tmp_path)
    database = connect_lancedb(store_paths.lancedb_path)
    table = open_entity_table(database, vector_size=embed_dims)
    replace_entity_embeddings(
        table,
        [
            {
                "entity_id": "alpha",
                "label": "Alpha",
                "community_id": 0,
                "vector": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "entity_id": "beta",
                "label": "Beta",
                "community_id": 0,
                "vector": [0.0, 1.0, 0.0, 0.0],
            },
            {
                "entity_id": "gamma",
                "label": "Gamma",
                "community_id": 1,
                "vector": [0.0, 0.0, 1.0, 0.0],
            },
        ],
    )

    expansion = ExpansionOutcome(
        expanded_question="alpha question",
        matched_entity_ids=["surface_match"],
        added_terms=[],
    )

    result = _augment_with_embedding_neighbours(
        expansion=expansion,
        query_vector=[1.0, 0.0, 0.0, 0.0],
        store_paths=store_paths,
        config=_config(embed_dims=embed_dims, max_terms=2),
    )

    assert "alpha" in result.matched_entity_ids, (
        f"Closest entity 'alpha' should be merged in; saw {result.matched_entity_ids}."
    )
    assert "surface_match" in result.matched_entity_ids, (
        "Surface-form mention from upstream expansion must be preserved."
    )


def test_augment_does_not_duplicate_existing_matched_ids(tmp_path: Path) -> None:
    """If a vector-nearest entity is already in matched_entity_ids, no duplicate is added."""
    embed_dims = 4
    store_paths = _store_paths(tmp_path)
    database = connect_lancedb(store_paths.lancedb_path)
    table = open_entity_table(database, vector_size=embed_dims)
    replace_entity_embeddings(
        table,
        [
            {
                "entity_id": "alpha",
                "label": "Alpha",
                "community_id": 0,
                "vector": [1.0, 0.0, 0.0, 0.0],
            },
        ],
    )

    expansion = ExpansionOutcome(
        expanded_question="alpha question",
        matched_entity_ids=["alpha"],
        added_terms=[],
    )

    result = _augment_with_embedding_neighbours(
        expansion=expansion,
        query_vector=[1.0, 0.0, 0.0, 0.0],
        store_paths=store_paths,
        config=_config(embed_dims=embed_dims, max_terms=5),
    )

    assert result.matched_entity_ids.count("alpha") == 1, (
        f"'alpha' should not be duplicated; saw {result.matched_entity_ids}."
    )
