"""
Language registry for AST-grounded chunking.

Maps file extensions to tree-sitter grammars, and declares -- per language --
which AST node types are worth indexing as standalone units.

Three categories per language:

  definitions  Nodes that become their own chunk (functions, methods, types).

  containers   Nodes we descend into so their members are indexed separately,
               while still emitting a header chunk for the container itself.
               A 600-line class must not be one chunk: it blows the context
               budget and buries the one method the query was about. The v1
               chunker had exactly this failure -- `_chunk_by_structure` emits
               a class as a single chunk and then skips its body entirely, so
               methods were never independently retrievable.

  wrappers     Nodes that decorate or export a definition and must be absorbed
               into it rather than chunked separately. Losing these is not
               cosmetic: stripping `@router.post("/api/chat")` from a FastAPI
               handler removes the HTTP verb and path from the indexed text,
               which is exactly what a user searches for.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanguageSpec:
    """Chunking configuration for one tree-sitter grammar."""

    # Grammar name as understood by tree_sitter_language_pack.
    grammar: str

    # Node types emitted as standalone chunks.
    definitions: frozenset[str]

    # Node types we descend into; members become chunks, container gets a
    # header chunk holding its signature and any leading documentation.
    containers: frozenset[str] = field(default_factory=frozenset)

    # Node types that wrap a definition and must be absorbed into its span.
    wrappers: frozenset[str] = field(default_factory=frozenset)

    # Comment node types, absorbed when directly above a definition.
    comments: frozenset[str] = frozenset({"comment"})


_PY = LanguageSpec(
    grammar="python",
    # `decorated_definition` is targeted directly so decorators land inside
    # the chunk. The inner function/class node is then suppressed by the
    # walker to avoid emitting the body twice.
    definitions=frozenset({"function_definition", "class_definition", "decorated_definition"}),
    # Only `class_definition` is a container. Container-ness is decided on the
    # *unwrapped* node, so `@dec class Foo` still resolves to a container
    # while `@dec def foo` does not -- see ASTChunker._walk.
    containers=frozenset({"class_definition"}),
    wrappers=frozenset({"decorated_definition"}),
)

_JS_DEFS = {
    "function_declaration",
    "generator_function_declaration",
    "class_declaration",
    "method_definition",
    "lexical_declaration",  # `const f = () => {}` / `const f = function () {}`
    "variable_declaration",
}
_JS_CONTAINERS = {"class_declaration"}
_JS_WRAPPERS = {"export_statement"}

_JS = LanguageSpec(
    grammar="javascript",
    definitions=frozenset(_JS_DEFS),
    containers=frozenset(_JS_CONTAINERS),
    wrappers=frozenset(_JS_WRAPPERS),
)

_TS = LanguageSpec(
    grammar="typescript",
    definitions=frozenset(
        _JS_DEFS
        | {
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "abstract_class_declaration",
            "internal_module",  # `namespace Foo {}`
        }
    ),
    containers=frozenset(_JS_CONTAINERS | {"abstract_class_declaration", "internal_module"}),
    wrappers=frozenset(_JS_WRAPPERS),
)

_TSX = LanguageSpec(
    grammar="tsx",
    definitions=_TS.definitions,
    containers=_TS.containers,
    wrappers=_TS.wrappers,
)

_GO = LanguageSpec(
    grammar="go",
    definitions=frozenset({"function_declaration", "method_declaration", "type_declaration"}),
)

_RUST = LanguageSpec(
    grammar="rust",
    definitions=frozenset(
        {
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "mod_item",
            "macro_definition",
        }
    ),
    containers=frozenset({"impl_item", "trait_item", "mod_item"}),
    comments=frozenset({"line_comment", "block_comment"}),
)

_JAVA = LanguageSpec(
    grammar="java",
    definitions=frozenset(
        {
            "method_declaration",
            "constructor_declaration",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        }
    ),
    containers=frozenset(
        {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}
    ),
    comments=frozenset({"line_comment", "block_comment"}),
)

_CSHARP = LanguageSpec(
    grammar="csharp",
    definitions=frozenset(
        {
            "method_declaration",
            "constructor_declaration",
            "property_declaration",
            "class_declaration",
            "interface_declaration",
            "struct_declaration",
            "record_declaration",
        }
    ),
    containers=frozenset(
        {"class_declaration", "interface_declaration", "struct_declaration", "namespace_declaration"}
    ),
)

_C = LanguageSpec(
    grammar="c",
    definitions=frozenset(
        {"function_definition", "struct_specifier", "enum_specifier", "type_definition"}
    ),
)

_CPP = LanguageSpec(
    grammar="cpp",
    definitions=frozenset(
        {
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
            "template_declaration",
        }
    ),
    containers=frozenset({"class_specifier", "namespace_definition"}),
)

_RUBY = LanguageSpec(
    grammar="ruby",
    definitions=frozenset({"method", "singleton_method", "class", "module"}),
    containers=frozenset({"class", "module"}),
)

_PHP = LanguageSpec(
    grammar="php",
    definitions=frozenset(
        {
            "function_definition",
            "method_declaration",
            "class_declaration",
            "interface_declaration",
            "trait_declaration",
        }
    ),
    containers=frozenset({"class_declaration", "interface_declaration", "trait_declaration"}),
    comments=frozenset({"comment"}),
)

_KOTLIN = LanguageSpec(
    grammar="kotlin",
    definitions=frozenset({"function_declaration", "class_declaration", "object_declaration"}),
    containers=frozenset({"class_declaration", "object_declaration"}),
    comments=frozenset({"line_comment", "multiline_comment"}),
)


# Extension -> language key. Extensions are matched lowercase, with the dot.
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".kts": "kotlin",
}

LANGUAGES: dict[str, LanguageSpec] = {
    "python": _PY,
    "javascript": _JS,
    "typescript": _TS,
    "tsx": _TSX,
    "go": _GO,
    "rust": _RUST,
    "java": _JAVA,
    "csharp": _CSHARP,
    "c": _C,
    "cpp": _CPP,
    "ruby": _RUBY,
    "php": _PHP,
    "kotlin": _KOTLIN,
}

# Text formats we index but do not parse: chunked by token-window instead.
# Documentation carries real retrieval value (READMEs answer "how do I run
# this" better than any source file), so it is indexed, just not parsed.
PROSE_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".mdx", ".rst", ".txt", ".adoc", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json"}
)


def spec_for_path(path: str) -> LanguageSpec | None:
    """Return the LanguageSpec for a file path, or None if it is not parsed."""
    lower = path.lower()
    idx = lower.rfind(".")
    if idx == -1:
        return None
    key = EXTENSION_MAP.get(lower[idx:])
    return LANGUAGES.get(key) if key else None


def language_name_for_path(path: str) -> str | None:
    """Return the language key for a path, or None."""
    lower = path.lower()
    idx = lower.rfind(".")
    if idx == -1:
        return None
    return EXTENSION_MAP.get(lower[idx:])


def is_prose(path: str) -> bool:
    lower = path.lower()
    idx = lower.rfind(".")
    return idx != -1 and lower[idx:] in PROSE_EXTENSIONS
