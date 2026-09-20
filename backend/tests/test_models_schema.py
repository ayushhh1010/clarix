"""
The ORM models and the migrated database must agree.

The bug this exists to prevent
------------------------------
Migration 0002 added seven provenance columns to `repositories`
(`indexed_commit_sha`, `index_version`, `last_indexed_at` and friends).
`models.Repository` was never updated to declare them.

Nothing failed for a long time, because the worker reads those columns
through raw SQL. But `/api/repo/{id}/status` and `/{id}/file-content` read
them off the ORM object, so both raised

    AttributeError: 'Repository' object has no attribute 'indexed_commit_sha'

and returned 500 -- on every call, for every repository. The dashboard
polls status continuously while indexing, so the two endpoints a user hits
most were the two that were broken.

It survived the whole test suite because the suite exercises the queue,
the worker, the pipeline and retrieval directly, and never drives the HTTP
routes with a real ORM object. It was found by running the stack and
calling the API.

This test compares both directions against a really-migrated database, so
a column added to one side and not the other fails immediately rather than
at the first request that happens to touch it.
"""

from __future__ import annotations

import pytest

# Importing the v2 models registers their tables on the same metadata.
from app import models, models_v2  # noqa: F401
from app.database import Base

# Tables created outside the application's metadata.
NOT_MAPPED = {"alembic_version"}


def _db_columns(conn, table: str) -> set[str]:
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s",
        (table,),
    ).fetchall()
    return {r[0] for r in rows}


def _mapped_tables() -> dict:
    return dict(Base.metadata.tables)


def test_the_suite_has_tables_to_check():
    """A vacuous pass here would hide every other assertion in this file."""
    assert len(_mapped_tables()) >= 6, sorted(_mapped_tables())


@pytest.mark.parametrize("table_name", sorted(_mapped_tables()))
def test_every_orm_column_exists_in_the_database(conn, table_name):
    """
    A column the ORM declares but the database lacks fails on first use,
    usually as a confusing UndefinedColumn from deep inside a query.
    """
    table = _mapped_tables()[table_name]
    actual = _db_columns(conn, table_name)
    assert actual, f"table {table_name!r} does not exist in the migrated database"

    declared = {c.name for c in table.columns}
    missing = sorted(declared - actual)
    assert not missing, (
        f"{table_name}: the ORM declares columns the database does not have: "
        f"{missing}. Add them in a migration."
    )


@pytest.mark.parametrize("table_name", sorted(_mapped_tables()))
def test_every_database_column_is_declared_on_the_model(conn, table_name):
    """
    The direction that actually bit.

    A column present in the database but absent from the model is
    invisible to any code holding an ORM instance -- reading it raises
    AttributeError at runtime, in a route, in production.
    """
    table = _mapped_tables()[table_name]
    actual = _db_columns(conn, table_name)
    assert actual, f"table {table_name!r} does not exist in the migrated database"

    declared = {c.name for c in table.columns}
    undeclared = sorted(actual - declared)
    assert not undeclared, (
        f"{table_name}: the database has columns the model does not declare: "
        f"{undeclared}. Anything reading them off an ORM object raises "
        f"AttributeError."
    )


def test_repository_exposes_the_provenance_columns(conn):
    """
    Named explicitly, so the failure says what broke rather than just
    listing a set difference. These seven are the ones that were missing.
    """
    declared = {c.name for c in models.Repository.__table__.columns}
    for name in (
        "default_branch",
        "head_commit_sha",
        "indexed_commit_sha",
        "index_version",
        "indexed_chunk_count",
        "indexed_token_count",
        "last_indexed_at",
    ):
        assert name in declared, f"Repository.{name} is missing from the model"
        assert name in _db_columns(conn, "repositories"), (
            f"repositories.{name} is missing from the database"
        )


def test_no_migrated_table_is_left_unmapped(conn):
    """
    A table the migrations create but no model maps is not necessarily
    wrong, but it should be a deliberate choice rather than an oversight.
    """
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    ).fetchall()
    in_db = {r[0] for r in rows} - NOT_MAPPED
    unmapped = sorted(in_db - set(_mapped_tables()))
    assert not unmapped, f"tables exist with no ORM model: {unmapped}"
