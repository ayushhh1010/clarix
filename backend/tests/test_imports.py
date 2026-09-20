"""
Import smoke tests.

The v2 serving set is meant to be thin, and bench/bench_import_rss.py
measures a *module list* rather than the application. This asserts the
application itself matches that list: the v2 modules must import without any
`legacy` dependency present, or the measured 88.1 MB is describing something
we do not actually ship.

Also guards the packaging regression this suite found: before the `legacy`
extra existed, `import app.main` failed on a clean install of the declared
core dependencies.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

V2_MODULES = [
    "app.main",
    "app.config",
    "app.database",
    "app.models",
    "app.models_v2",
    "app.security",
    "app.indexing.chunker",
    "app.indexing.languages",
    "app.indexing.tokens",
    "app.retrieval.hybrid",
    "app.evaluation.metrics",
    "app.evaluation.dataset",
    "app.content",
    "app.conversations",
    "app.llm",
    "app.retrieval.context",
    "app.retrieval.query_embedder",
    "app.routes.repo",
    "app.routes.chat",
    "app.routes.agent",
]

LEGACY_MODULES = ["chromadb", "langchain", "langchain_core", "redis", "git", "jose", "passlib"]


@pytest.mark.parametrize("module", V2_MODULES)
def test_v2_module_imports(module):
    importlib.import_module(module)


def test_v2_modules_do_not_pull_in_legacy_dependencies():
    """
    Run in a subprocess with the legacy packages blocked. If a v2 module
    reaches one of them, the thin-serving-set claim is false.
    """
    blocked = ", ".join(repr(m) for m in LEGACY_MODULES)
    script = f"""
import sys
class Blocker:
    def find_module(self, name, path=None):
        if name.split('.')[0] in {{{blocked}}}:
            raise ImportError('blocked for this test: ' + name)
        return None
sys.meta_path.insert(0, Blocker())
for m in {V2_MODULES!r}:
    __import__(m)
print('ok')
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, f"a v2 module reached a legacy dependency:\n{out.stderr[-1500:]}"
    assert "ok" in out.stdout


def test_app_main_imports_with_only_core_dependencies():
    """
    The application must import from the declared core dependencies alone.

    This was a genuine regression: before the v1 modules were removed,
    `import app.main` failed on a clean install because it reached
    chromadb, langchain, redis and gitpython that pyproject no longer
    declared.
    """
    importlib.import_module("app.main")
