"""Unit tests for migration backup hard-gate behaviour."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from lxd.stores.schema import SchemaIntegrityError, _backup_database_for_migration, ensure_schema

pytestmark = [pytest.mark.unit]


def test_migration_backup_raises_when_copy_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "lxd.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        ensure_schema(connection)
        with (
            patch("lxd.stores.schema.shutil.copy2", side_effect=OSError("disk full")),
            pytest.raises(SchemaIntegrityError, match="could not write backup"),
        ):
            _backup_database_for_migration(connection, from_version=1, to_version=2)
    finally:
        connection.close()


def test_migration_backup_skips_memory_database() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        # Must not raise — there is no on-disk file to snapshot.
        _backup_database_for_migration(connection, from_version=0, to_version=1)
    finally:
        connection.close()
