"""
Build a retrieval evaluation set from a code corpus.

Protocol
--------
Weak supervision in the style of CodeSearchNet: a function's docstring is a
natural-language description of that function, so (docstring -> function) is
a usable query/gold pair, and it can be harvested at scale without hand
labelling.

THE LEAKAGE PROBLEM, and what is done about it
-----------------------------------------------
If the docstring is both the query *and* part of the indexed chunk, retrieval
is trivial -- the lexical arm matches the query text verbatim. Every metric
inflates toward 1.0 and the benchmark loses all power to discriminate between
systems, which is the only thing a benchmark is for.

So each pair carries the chunk with its docstring **stripped**, and eval runs
index that stripped text.

Attribution, checked: the docstring-as-query idea is CodeSearchNet's. The
stripping is NOT -- their README documents extracting code and docstring as
separate fields but says nothing about removing the docstring from the
indexed code, so this control is our own decision and is justified by
measurement (BENCHMARKS.md section 7) rather than by precedent.

It has a consequence worth stating plainly: the eval index is not
byte-identical to the production index, which does contain docstrings.
Absolute numbers from this set therefore understate production retrieval.
Relative comparisons between systems -- which is what the set exists for --
remain valid, because every system sees the same stripped index.

`--keep-docstrings` runs without stripping. The gap between the two runs is a
direct measurement of how much leakage was present, and is worth reporting
once rather than arguing about.

Quality filters
---------------
Most docstrings make poor queries. Discarded:
  * shorter than 5 words, or longer than 50 (those are usually API reference
    blocks, not questions anyone asks)
  * boilerplate ("TODO", "See above", ":param x:", copyright headers)
  * text that merely restates the identifier ("get_user: gets the user") --
    these reduce to exact symbol lookup and measure nothing about semantics
  * duplicates, which over-weight whatever the copied function does

What this set is NOT
--------------------
It is not human-labelled, and docstring phrasing is not how developers
actually ask questions ("why does auth fail on refresh"). It is a
regression-detection instrument and a system-comparison instrument. A
hand-written set would be the complement; there is not one yet, and this
module does not pretend otherwise.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_QUERY_WORDS = 5
MAX_QUERY_WORDS = 50

# Docstring text that is structure, not description.
_BOILERPLATE = re.compile(
    r"^\s*(todo|fixme|xxx|note:|see above|see below|copyright|licen[cs]e|"
    r"deprecated|no-?op|placeholder|internal use only)\b",
    re.IGNORECASE,
)
_PARAM_LINE = re.compile(r"^\s*(:param|:return|:rtype|:raises|@param|@return|Args:|Returns:|Raises:)")


@dataclass
class EvalExample:
    """One query and its gold chunk(s)."""

    query_id: str
    query: str
    gold_chunk_ids: list[str]
    gold_symbol_path: str
    repo: str
    language: str
    source: str = "docstring"
    difficulty: str = "unknown"
    meta: dict = field(default_factory=dict)


def _first_sentence(text: str) -> str:
    """Leading prose of a docstring, stopping before parameter blocks."""
    lines: list[str] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            if lines:
                break
            continue
        if _PARAM_LINE.match(line):
            break
        lines.append(line)
    joined = " ".join(lines)
    # Cut at the first sentence boundary, but keep short multi-clause text.
    match = re.search(r"(?<=[.!?])\s+", joined)
    return (joined[: match.start()] if match and match.start() > 30 else joined).strip()


def _normalise_identifier(name: str) -> set[str]:
    parts = re.split(r"[_\-.]+|(?<=[a-z0-9])(?=[A-Z])", name)
    return {p.lower() for p in parts if len(p) > 2}


def is_usable_query(text: str, symbol: str | None) -> tuple[bool, str]:
    """Return (usable, reason-if-not). Reasons are counted and reported."""
    if not text:
        return False, "empty"
    if _BOILERPLATE.match(text):
        return False, "boilerplate"

    words = text.split()
    if len(words) < MIN_QUERY_WORDS:
        return False, "too_short"
    if len(words) > MAX_QUERY_WORDS:
        return False, "too_long"

    if symbol:
        # A docstring that only restates its identifier collapses the task to
        # exact symbol lookup and measures nothing the symbol arm does not
        # already trivially win.
        ident = _normalise_identifier(symbol)
        content = {w.lower().strip(".,:;()") for w in words}
        content -= {
            "the", "a", "an", "of", "for", "to", "and", "or", "is", "are",
            "this", "that", "it", "its", "with", "from", "in", "on", "by",
            "return", "returns", "get", "gets", "set", "sets",
        }
        if content and ident and content <= ident:
            return False, "restates_identifier"

    return True, ""


def strip_docstring(content: str, language: str) -> str:
    """
    Remove the leading docstring or doc-comment from a chunk's text.

    Only the *leading* doc block is removed: inline comments inside the body
    are legitimate signal that production retrieval also sees.
    """
    if language == "python":
        # Leading triple-quoted string after the def/class line.
        return re.sub(
            r'(\A\s*(?:@[^\n]*\n\s*)*(?:async\s+)?(?:def|class)[^\n]*:\n)'
            r'(\s*)(?:"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\')\n?',
            r"\1",
            content,
            count=1,
        )
    # JSDoc / javadoc block immediately above or inside the declaration.
    without_block = re.sub(r"\A\s*/\*\*(?:.|\n)*?\*/\n?", "", content, count=1)
    if without_block != content:
        return without_block

    prefix = _LINE_DOC_PREFIX.get(language)
    if prefix:
        _text, consumed = _leading_line_comment(content, prefix)
        if consumed:
            return "\n".join(content.splitlines()[consumed:])
    return content


# Languages whose doc convention is a run of line comments immediately above
# the declaration rather than a block inside it. The chunker already absorbs
# those leading comments into the chunk, so they are there to extract.
#
# Without this, Go contributed 0 of 1,185 chunks to the evaluation set and it
# came out 100% Python -- a benchmark that cannot see a regression in any
# other language.
_LINE_DOC_PREFIX = {
    "go": "//",
    "rust": "///",
    "javascript": "//",
    "typescript": "//",
    "tsx": "//",
    "c": "//",
    "cpp": "//",
    "csharp": "///",
    "java": "//",
    "kotlin": "//",
}


def _leading_line_comment(content: str, prefix: str) -> tuple[str, int]:
    """Return (joined comment text, number of leading lines it occupies)."""
    out: list[str] = []
    consumed = 0
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            out.append(stripped[len(prefix) :].strip())
            consumed += 1
            continue
        if not stripped and out:
            consumed += 1
            continue
        break
    return " ".join(t for t in out if t).strip(), consumed


def extract_docstring(content: str, language: str) -> str:
    if language == "python":
        m = re.search(r'(?:"""((?:.|\n)*?)"""|\'\'\'((?:.|\n)*?)\'\'\')', content)
        if m:
            return (m.group(1) or m.group(2) or "").strip()
        return ""

    m = re.search(r"/\*\*((?:.|\n)*?)\*/", content)
    if m:
        body = m.group(1)
        cleaned = "\n".join(re.sub(r"^\s*\*+\s?", "", ln).strip() for ln in body.splitlines())
        return "\n".join(ln for ln in cleaned.splitlines() if not ln.startswith("@")).strip()

    prefix = _LINE_DOC_PREFIX.get(language)
    if prefix:
        text, _consumed = _leading_line_comment(content, prefix)
        return text
    return ""


@dataclass
class BuildStats:
    chunks_seen: int = 0
    with_docstring: int = 0
    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def report(self) -> str:
        lines = [
            f"chunks seen        {self.chunks_seen:,}",
            f"with docstring     {self.with_docstring:,}",
            f"accepted           {self.accepted:,}",
        ]
        for reason, n in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            lines.append(f"  rejected {reason:<22} {n:,}")
        return "\n".join(lines)


def build_from_chunks(
    chunks: list, repo: str, stats: BuildStats | None = None
) -> Iterator[EvalExample]:
    """
    Yield eval examples from an iterable of `app.indexing.chunker.Chunk`.

    Deduplicates on normalised query text: copy-pasted docstrings would
    otherwise weight whatever they describe several times over.
    """
    stats = stats or BuildStats()
    seen: set[str] = set()

    for chunk in chunks:
        stats.chunks_seen += 1
        if chunk.kind not in {"definition", "container_header"}:
            continue

        doc = extract_docstring(chunk.content, chunk.language)
        if not doc:
            continue
        stats.with_docstring += 1

        query = _first_sentence(doc)
        usable, reason = is_usable_query(query, chunk.symbol)
        if not usable:
            stats.reject(reason)
            continue

        key = re.sub(r"\W+", " ", query.lower()).strip()
        if key in seen:
            stats.reject("duplicate")
            continue
        seen.add(key)

        stats.accepted += 1
        yield EvalExample(
            query_id=f"{repo}:{chunk.chunk_id}",
            query=query,
            gold_chunk_ids=[chunk.chunk_id],
            gold_symbol_path=chunk.symbol_path,
            repo=repo,
            language=chunk.language,
            source="docstring",
            meta={
                "file_path": chunk.file_path,
                "symbol": chunk.symbol,
                "token_count": chunk.token_count,
                "query_words": len(query.split()),
                # Go's doc convention begins with the identifier ("ServeHTTP
                # conforms to..."), so such queries are trivially winnable by
                # the symbol arm alone. Flagged so results can be reported
                # split by it rather than silently inflated by it.
                "names_symbol": bool(
                    chunk.symbol and chunk.symbol.lower() in query.lower()
                ),
            },
        )


def write_jsonl(examples: list[EvalExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[EvalExample]:
    out: list[EvalExample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(EvalExample(**json.loads(line)))
    return out


def split(
    examples: list[EvalExample], train: float = 0.5, seed: int = 20260919
) -> tuple[list[EvalExample], list[EvalExample]]:
    """
    Deterministic dev/test split.

    Tuning on the same data used to report is how an eval harness starts
    lying: retrieval knobs (over-fetch depth, RRF k, arm weights) get fitted
    to noise. Tune on dev, report on test, and report test only once per
    change.
    """
    import random

    # Deterministic shuffle for a reproducible split; not a security context.
    rng = random.Random(seed)  # noqa: S311
    shuffled = sorted(examples, key=lambda e: e.query_id)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * train)
    return shuffled[:cut], shuffled[cut:]


# --------------------------------------------------------------------------
# Symbol-lookup queries
# --------------------------------------------------------------------------
#
# The docstring-derived set measures semantic matching only, and the first
# dev run showed dense retrieval alone beats every hybrid configuration on
# it. That is a real result but it cannot be the whole story: the symbol and
# lexical arms exist for identifier lookups ("where is get_current_user
# defined"), and the docstring set contains essentially none of those.
#
# Deleting the arms on evidence from a benchmark that cannot see their use
# case would be over-generalising; keeping them without such evidence would
# be ignoring the measurement. This set supplies the missing evidence.

SYMBOL_QUERY_TEMPLATES = [
    "where is {sym} defined",
    "{sym}",
    "show me the {sym} implementation",
    "find the definition of {sym}",
]


def build_symbol_lookup(
    chunks: list, repo: str, templates: list[str] | None = None
) -> Iterator[EvalExample]:
    """
    Build identifier-lookup queries: (phrasing of a symbol name -> its chunk).

    Only symbols that are **unique within the repository** are used. A name
    defined in three places has three defensible gold answers, and scoring
    one of them as correct and the others as misses would measure ambiguity
    rather than retrieval.

    Very short names are skipped: `f`, `id` and `ok` match half the corpus
    by trigram and would measure nothing but the fuzzy fallback.
    """
    templates = templates or SYMBOL_QUERY_TEMPLATES

    counts: dict[str, int] = {}
    for c in chunks:
        if c.symbol and c.kind in {"definition", "container_header"}:
            counts[c.symbol] = counts.get(c.symbol, 0) + 1

    for chunk in chunks:
        sym = chunk.symbol
        if not sym or chunk.kind not in {"definition", "container_header"}:
            continue
        if counts.get(sym, 0) != 1 or len(sym) < 4:
            continue

        for i, template in enumerate(templates):
            yield EvalExample(
                query_id=f"{repo}:{chunk.chunk_id}:sym{i}",
                query=template.format(sym=sym),
                gold_chunk_ids=[chunk.chunk_id],
                gold_symbol_path=chunk.symbol_path,
                repo=repo,
                language=chunk.language,
                source="symbol_lookup",
                difficulty="bare_name" if template == "{sym}" else "phrased",
                meta={
                    "file_path": chunk.file_path,
                    "symbol": sym,
                    "template": template,
                    "names_symbol": True,
                },
            )
