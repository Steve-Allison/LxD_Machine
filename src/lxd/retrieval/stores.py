"""Shared store handles for one retrieval call — one SQLite open, one Lance open."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from lxd.settings.models import RuntimeConfig
from lxd.stores.lancedb import connect_lancedb, open_chunk_table
from lxd.stores.models import StorePaths
from lxd.stores.sqlite._pool import pooled_connection
from lxd.stores.sqlite.connection import build_store_paths


@dataclass(slots=True)
class RetrievalStores:
    """Open store handles scoped to a single ``search_chunks`` / answer call."""

    paths: StorePaths
    sqlite: sqlite3.Connection
    chunk_table: Any


@contextmanager
def open_retrieval_stores(config: RuntimeConfig) -> Generator[RetrievalStores]:
    """Yield SQLite + LanceDB chunk table using pooled SQLite connection."""
    paths = build_store_paths(config.paths.data_path)
    with pooled_connection(paths.sqlite_path) as connection:
        table = open_chunk_table(
            connect_lancedb(paths.lancedb_path),
            vector_size=config.models.embed_dims,
        )
        yield RetrievalStores(paths=paths, sqlite=connection, chunk_table=table)
