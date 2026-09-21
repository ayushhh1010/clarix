"""
Accept a database URL in whatever form the provider hands out.

Why this exists
---------------
Managed Postgres providers give you a libpq-style URL. Neon's looks like

    postgresql://user:pass@ep-x.aws.neon.tech/db?sslmode=require&channel_binding=require

and pasting that straight into the application used to break in two
different ways at once:

  The driver. `postgresql://` was rewritten to `postgresql+asyncpg://`
  for the application, but Alembic only recognised `postgresql+asyncpg`
  and `postgres://`, so a plain `postgresql://` fell through to
  SQLAlchemy's default dialect -- psycopg2, which is not installed.

  The TLS parameter. libpq spells it `sslmode`, asyncpg spells it `ssl`,
  and the two are not interchangeable:

      TypeError: connect() got an unexpected keyword argument 'sslmode'

  asyncpg also rejects `channel_binding`, which Neon appends by default.

Rather than telling an operator to hand-edit the string their provider
gave them -- and getting a confusing error when they do not -- this
normalises it for whichever driver is asking. The application uses
asyncpg; Alembic cannot, and uses psycopg.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ASYNC_DRIVER = "postgresql+asyncpg"
SYNC_DRIVER = "postgresql+psycopg"

# libpq accepts these words for sslmode; asyncpg's `ssl` understands the
# same vocabulary, so the value carries across unchanged. Booleans do not,
# and are mapped to the nearest libpq mode.
_BOOL_TO_SSLMODE = {
    "true": "require", "1": "require", "yes": "require", "on": "require",
    "false": "disable", "0": "disable", "no": "disable", "off": "disable",
}

# Query parameters asyncpg's connect() does not accept. They are libpq
# connection options, and passing them through raises TypeError. Dropping
# them is safe: `ssl` already carries the security-relevant setting.
_LIBPQ_ONLY = {
    "channel_binding",
    "connect_timeout",
    "application_name",
    "options",
    "target_session_attrs",
    "gssencmode",
}


def _split_scheme(url: str) -> tuple[str, str]:
    """Return (scheme, rest) without assuming which postgres spelling."""
    marker = "://"
    index = url.find(marker)
    if index == -1:
        return "", url
    return url[:index], url[index + len(marker) :]


def normalise_database_url(url: str, *, driver: str) -> str:
    """
    Rewrite `url` for `driver`, which is "asyncpg" or "psycopg".

    Handles every spelling a provider or PaaS is likely to inject:
    `postgres://` (Heroku, Railway), `postgresql://` (Neon, Supabase,
    libpq) and an already-qualified `postgresql+asyncpg://`.
    """
    if driver not in ("asyncpg", "psycopg"):
        raise ValueError(f"unknown driver {driver!r}")
    if not url:
        return url

    scheme, rest = _split_scheme(url)
    if not scheme.startswith("postgres"):
        # Not a postgres URL; leave it alone rather than mangle it.
        return url

    target = ASYNC_DRIVER if driver == "asyncpg" else SYNC_DRIVER
    parts = urlsplit(f"{target}://{rest}")

    kept: list[tuple[str, str]] = []
    ssl_value: str | None = None
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key in ("ssl", "sslmode"):
            # Last one wins, which matches how libpq reads duplicates.
            ssl_value = value
            continue
        if driver == "asyncpg" and key in _LIBPQ_ONLY:
            continue
        kept.append((key, value))

    if ssl_value is not None:
        canonical = _BOOL_TO_SSLMODE.get(ssl_value.lower(), ssl_value)
        kept.append(("ssl" if driver == "asyncpg" else "sslmode", canonical))

    return urlunsplit(parts._replace(query=urlencode(kept)))


def redact(url: str) -> str:
    """
    The URL with its password removed, for logs and error messages.

    Connection strings end up in deploy logs and exception text, and the
    password is the one part that must not.
    """
    scheme, rest = _split_scheme(url)
    if not scheme or "@" not in rest:
        return url
    creds, _, host = rest.partition("@")
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"
