"""
Test fixtures.

Postgres comes from `embedded-postgres` (PostgreSQL 18.6 + pgvector 0.8.6 as
a pip wheel) rather than Docker. Two reasons: the schema depends on pgvector
specifics that only a real server can validate, and a test suite that needs a
running daemon is a test suite that gets skipped in CI and on a fresh
machine. This one runs anywhere `pip install` works.

The server is started once per session -- initdb costs a few seconds -- and
each test that needs isolation gets its own database on that instance.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
import uuid

import pytest

# NOTE: deliberately NOT a module-level importorskip.
#
# `pytest.importorskip` at module scope in a conftest skips the whole
# directory, so a missing Postgres dependency would silently skip every test
# including the ones that need no database at all -- and a green run with
# nothing executed looks identical to a green run. The skip belongs on the
# fixtures that actually need the dependency.

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def pg_server():
    """A PostgreSQL instance for the whole test session."""
    pytest.importorskip("psycopg", reason="dev extra not installed")
    get_server = pytest.importorskip(
        "embedded_postgres", reason="dev extra not installed"
    ).get_server

    datadir = pathlib.Path(tempfile.mkdtemp(prefix="clarix_test_pg_"))
    server = get_server(datadir, cleanup_mode=None)
    try:
        yield server
    finally:
        try:
            server.cleanup()
        except Exception:  # noqa: BLE001 - teardown must not mask failures
            pass
        shutil.rmtree(datadir, ignore_errors=True)


@pytest.fixture
def pg_url(pg_server) -> str:
    """A fresh, empty database. Returns a psycopg (sync) URL."""
    import psycopg

    admin = pg_server.get_uri()
    name = f"t{uuid.uuid4().hex[:16]}"
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    url = pg_server.get_uri(database=name)
    # `embedded_postgres` yields a bare postgresql:// URL; SQLAlchemy needs
    # the driver named explicitly or it reaches for psycopg2.
    yield url.replace("postgresql://", "postgresql+psycopg://", 1)
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def migrated(pg_url: str) -> str:
    """A database with `alembic upgrade head` applied."""
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    previous = os.environ.get("ALEMBIC_DATABASE_URL")
    os.environ["ALEMBIC_DATABASE_URL"] = pg_url
    try:
        command.upgrade(cfg, "head")
        yield pg_url
    finally:
        if previous is None:
            os.environ.pop("ALEMBIC_DATABASE_URL", None)
        else:
            os.environ["ALEMBIC_DATABASE_URL"] = previous


@pytest.fixture
def conn(migrated: str):
    """An autocommit connection to a migrated database."""
    import psycopg

    raw = migrated.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(raw, autocommit=True) as c:
        yield c


@pytest.fixture
async def async_session(migrated: str):
    """An AsyncSession against the migrated database (asyncpg driver)."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    url = migrated.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(url, poolclass=None)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def seeded_repo(conn) -> str:
    """A repository row plus a handful of chunks with deterministic vectors."""
    import hashlib

    rid = "22222222-2222-2222-2222-222222222222"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (id, name, local_path, status) "
            "VALUES (%s, 'seed', '/tmp/seed', 'ready')",
            (rid,),
        )

    fixtures = [
        ("get_current_user", "app/security.py",
         "def get_current_user(token: str):\n    return decode_access_token(token)"),
        ("decode_access_token", "app/security.py",
         "def decode_access_token(token: str):\n    return jwt.decode(token, SECRET)"),
        ("hash_password", "app/security.py",
         "def hash_password(password: str):\n    return bcrypt.hashpw(password)"),
        ("getCurrentUserCamel", "src/lib/auth.ts",
         "export function getCurrentUserCamel() { return session.user; }"),
        ("chunk_repository", "app/indexing/chunker.py",
         "def chunk_repository(files, repo_id):\n    return [chunk_file(f) for f in files]"),
        (None, "README.md",
         "Clarix indexes a git repository and answers questions about the code."),
    ]

    dim = 768
    with conn.cursor() as cur:
        for i, (symbol, path, body) in enumerate(fixtures):
            # Deterministic pseudo-vectors: distinct per chunk, stable
            # across runs, so ranking assertions are reproducible.
            seed = int(hashlib.sha256(f"{symbol}{path}".encode()).hexdigest()[:8], 16)
            vals = [((seed >> (j % 24)) & 0xFF) / 255.0 - 0.5 for j in range(dim)]
            lit = "[" + ",".join(f"{v:.4f}" for v in vals) + "]"
            cur.execute(
                f"""
                INSERT INTO chunks (chunk_id, repo_id, content_sha, file_path, language,
                                    kind, symbol, parent_scope, symbol_path,
                                    start_line, end_line, token_count, content,
                                    embedding, embedding_bits)
                VALUES (%s, %s, %s, %s, %s, 'definition', %s, '', %s,
                        1, 3, 20, %s, %s::halfvec({dim}),
                        binary_quantize(%s::vector({dim})))
                """,
                (
                    f"seed{i:020d}", rid, f"{i:064d}", path,
                    "python" if path.endswith(".py") else "typescript",
                    symbol, f"{path}::{symbol or ''}", body, lit, lit,
                ),
            )
    return rid
