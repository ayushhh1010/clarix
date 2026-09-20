"""
Compare the v1 (indentation-heuristic) chunker against the v2 (AST) chunker.

Ground truth is enumerated with tree-sitter *in this file*, independently of
either chunker, so the measurement code and the code under test are separate.

Honesty note on metric selection
--------------------------------
Two classes of metric are reported, and they are not equally strong:

  Independent   Decorator preservation, method isolation, oversize rate and
                token distribution are objective properties of the emitted
                spans. Either the decorator bytes are inside the chunk or
                they are not. v2 has no structural advantage in being scored
                on these -- it simply does not have the defect.

  Circular      `definition_coverage` asks "what fraction of tree-sitter
                definitions is fully contained in some chunk". v2 finds its
                boundaries with the same grammar, so it should approach 1.0
                by construction. It is reported for v1's sake and is marked
                CIRCULAR in the output. Do not quote it as evidence for v2.

Usage:
    python bench_chunking.py --clone            # fetch the corpus (shallow)
    python bench_chunking.py --run              # measure
    python bench_chunking.py --run --json-out results/chunking.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BENCH_DIR = Path(__file__).parent
CORPUS_DIR = BENCH_DIR / "corpus"
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

# Shallow-cloned corpus. Chosen for language spread and for density of the
# constructs v1 mishandles: decorated route handlers, large classes, and
# TS arrow-function exports.
CORPUS = [
    ("flask", "https://github.com/pallets/flask.git"),
    ("httpx", "https://github.com/encode/httpx.git"),
    ("requests", "https://github.com/psf/requests.git"),
    ("gin", "https://github.com/gin-gonic/gin.git"),
]

# Encoder context window for the embedding model (jina v2 base code).
ENCODER_WINDOW = 8192
# Per-chunk ceiling implied by packing 5-6 chunks into a 4,000-token budget.
PACKING_BUDGET = 1024

SKIP_PARTS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    ".next", "vendor", "testdata", ".mypy_cache", ".pytest_cache",
}
CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java"}


# --------------------------------------------------------------------------
# Ground truth (tree-sitter, independent of both chunkers)
# --------------------------------------------------------------------------

@dataclass
class Definition:
    kind: str           # "function" | "class" | "method"
    name: str
    start_line: int     # 1-indexed, at the `def`/`class` keyword
    end_line: int
    decorator_line: int | None   # first decorator line, if any
    parent: str | None           # enclosing class name, if any


_GT_NODES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
    "tsx": {"function_declaration", "class_declaration", "method_definition"},
    "go": {"function_declaration", "method_declaration"},
    "rust": {"function_item", "struct_item", "impl_item"},
    "java": {"method_declaration", "class_declaration"},
}
_CLASS_NODES = {"class_definition", "class_declaration", "impl_item"}


def ground_truth(source: str, language: str) -> list[Definition]:
    """Enumerate definitions via tree-sitter, with decorator spans."""
    from tree_sitter_language_pack import get_parser

    wanted = _GT_NODES.get(language)
    if not wanted:
        return []

    data = source.encode("utf-8")
    tree = get_parser(language).parse(data)
    found: list[Definition] = []

    def name_of(node) -> str:
        n = node.child_by_field_name("name")
        return data[n.start_byte : n.end_byte].decode("utf-8", "replace") if n else "<anon>"

    def decorator_start(node) -> int | None:
        """First decorator line for this definition, if decorated."""
        parent = node.parent
        # Python wraps decorated defs in `decorated_definition`.
        if parent is not None and parent.type == "decorated_definition":
            decs = [c for c in parent.named_children if c.type == "decorator"]
            if decs:
                return decs[0].start_point[0] + 1
        # Java/TS put modifiers/annotations as leading siblings.
        decs = [c for c in node.named_children if c.type in {"decorator", "modifiers", "annotation"}]
        if decs and decs[0].start_point[0] < node.start_point[0]:
            return decs[0].start_point[0] + 1
        return None

    def walk(node, parent_class: str | None) -> None:
        for child in node.named_children:
            if child.type in wanted:
                is_class = child.type in _CLASS_NODES
                nm = name_of(child)
                found.append(
                    Definition(
                        kind="class" if is_class else ("method" if parent_class else "function"),
                        name=nm,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        decorator_line=decorator_start(child),
                        parent=parent_class,
                    )
                )
                walk(child, nm if is_class else parent_class)
            else:
                walk(child, parent_class)

    walk(tree.root_node, None)
    return found


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

@dataclass
class Span:
    start_line: int
    end_line: int
    token_count: int

    def covers(self, d: Definition) -> bool:
        """Contains the definition in its entirety."""
        return self.start_line <= d.start_line and self.end_line >= d.end_line

    def covers_head(self, d: Definition) -> bool:
        """Contains the `def`/`class` line -- i.e. the signature is indexed."""
        return self.start_line <= d.start_line <= self.end_line


def score(spans: list[Span], defs: list[Definition]) -> dict:
    """
    Score one file's chunk spans against its ground-truth definitions.

    Head-coverage, not whole-definition coverage, is the basis for the
    decorator and method metrics. Deliberately splitting an oversized
    definition into parts means no single chunk contains the whole thing --
    correct behaviour that a whole-coverage test would score as a regression.
    What actually matters is whether the chunk carrying the signature also
    carries the decorator.
    """
    classes = {d.name: d for d in defs if d.kind == "class"}

    covered = 0
    head_indexed = 0
    decorated_total = 0
    decorated_kept = 0
    methods_total = 0
    methods_isolated = 0

    for d in defs:
        covering = [s for s in spans if s.covers(d)]
        head = [s for s in spans if s.covers_head(d)]
        if covering:
            covered += 1
        if head:
            head_indexed += 1

        if d.decorator_line is not None:
            decorated_total += 1
            # Preserved iff the chunk carrying the signature also starts at
            # or above the first decorator line.
            if any(s.start_line <= d.decorator_line for s in head):
                decorated_kept += 1

        if d.kind == "method":
            methods_total += 1
            parent = classes.get(d.parent or "")
            parent_span = (parent.end_line - parent.start_line) if parent else 10**9
            # Isolated iff some chunk carrying this method's signature is
            # strictly tighter than the whole enclosing class -- i.e. the
            # method is independently retrievable rather than buried inside
            # one giant class chunk.
            if any((s.end_line - s.start_line) < parent_span for s in head):
                methods_isolated += 1

    toks = [s.token_count for s in spans]
    return {
        "chunks": len(spans),
        "defs": len(defs),
        "defs_covered": covered,
        "head_indexed": head_indexed,
        "decorated_total": decorated_total,
        "decorated_kept": decorated_kept,
        "methods_total": methods_total,
        "methods_isolated": methods_isolated,
        "tokens": toks,
        "over_encoder": sum(1 for t in toks if t > ENCODER_WINDOW),
        "over_budget": sum(1 for t in toks if t > PACKING_BUDGET),
    }


def merge(acc: dict, one: dict) -> None:
    for k, v in one.items():
        if k == "tokens":
            acc.setdefault("tokens", []).extend(v)
        else:
            acc[k] = acc.get(k, 0) + v


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------

def iter_corpus() -> list[tuple[str, Path, str, str]]:
    """Yield (repo, path, relative_path, language) for every code file."""
    from app.indexing.languages import language_name_for_path

    out = []
    for repo_dir in sorted(CORPUS_DIR.iterdir()) if CORPUS_DIR.exists() else []:
        if not repo_dir.is_dir():
            continue
        for path in repo_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in CODE_EXTS:
                continue
            if any(p in SKIP_PARTS for p in path.parts):
                continue
            try:
                if path.stat().st_size > 1_000_000:
                    continue
            except OSError:
                continue
            lang = language_name_for_path(str(path))
            if lang:
                out.append((repo_dir.name, path, path.relative_to(repo_dir).as_posix(), lang))
    return out


def run_v1(files, counter) -> tuple[dict, float]:
    """Run the v1 indentation chunker over the corpus."""
    # Frozen copy, not the live module: app/ingestion/ has been deleted and
    # a baseline that disappears with the code it measured is not a baseline.
    from v1_chunker import ParsedFile
    from v1_chunker import chunk_file as v1_chunk_file

    acc: dict = {}
    t0 = time.perf_counter()
    for _repo, path, rel, lang in files:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parsed = ParsedFile(path, rel, source, lang, source.count("\n") + 1)
        chunks = v1_chunk_file(parsed, "bench")
        spans = [
            Span(c.start_line, c.end_line, counter.count(c.content)) for c in chunks
        ]
        merge(acc, score(spans, ground_truth(source, lang)))
    return acc, time.perf_counter() - t0


def run_v2(files, counter) -> tuple[dict, float, object]:
    """Run the v2 AST chunker over the corpus."""
    from app.indexing.chunker import ASTChunker

    chunker = ASTChunker(count_tokens=counter.count)
    acc: dict = {}
    t0 = time.perf_counter()
    for _repo, path, rel, lang in files:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks = chunker.chunk_file("bench", rel, source)
        spans = [Span(c.start_line, c.end_line, c.token_count) for c in chunks]
        merge(acc, score(spans, ground_truth(source, lang)))
    return acc, time.perf_counter() - t0, chunker.stats


def pct(num: int, den: int) -> str:
    return f"{num / den * 100:5.1f}%" if den else "    --"


def summarise(name: str, acc: dict, seconds: float, n_files: int) -> dict:
    toks = sorted(acc.get("tokens", []))
    return {
        "chunker": name,
        "files": n_files,
        "chunks": acc.get("chunks", 0),
        "seconds": round(seconds, 2),
        "files_per_sec": round(n_files / seconds, 1) if seconds else None,
        "definition_coverage_CIRCULAR": (
            round(acc.get("defs_covered", 0) / acc["defs"], 4) if acc.get("defs") else None
        ),
        "signature_indexed_CIRCULAR": (
            round(acc.get("head_indexed", 0) / acc["defs"], 4) if acc.get("defs") else None
        ),
        "decorator_preservation": (
            round(acc.get("decorated_kept", 0) / acc["decorated_total"], 4)
            if acc.get("decorated_total") else None
        ),
        "decorated_total": acc.get("decorated_total", 0),
        "method_isolation": (
            round(acc.get("methods_isolated", 0) / acc["methods_total"], 4)
            if acc.get("methods_total") else None
        ),
        "methods_total": acc.get("methods_total", 0),
        "tokens_p50": toks[len(toks) // 2] if toks else None,
        "tokens_p95": toks[int(len(toks) * 0.95)] if toks else None,
        "tokens_max": toks[-1] if toks else None,
        "tokens_mean": round(statistics.mean(toks), 1) if toks else None,
        "over_encoder_window": acc.get("over_encoder", 0),
        "over_packing_budget": acc.get("over_budget", 0),
        "over_budget_pct": (
            round(acc.get("over_budget", 0) / acc["chunks"] * 100, 2) if acc.get("chunks") else None
        ),
    }


def clone() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in CORPUS:
        dest = CORPUS_DIR / name
        if dest.exists():
            print(f"  = {name} already present")
            continue
        print(f"  + cloning {name} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
            check=True, timeout=600,
        )
    for name, _ in CORPUS:
        dest = CORPUS_DIR / name
        if dest.exists():
            sha = subprocess.run(
                ["git", "-C", str(dest), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True,
            ).stdout.strip()
            print(f"  {name:<12} @ {sha}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    if args.clone:
        clone()
        if not args.run:
            return 0

    if not args.run:
        ap.error("pass --run (and/or --clone)")

    from app.indexing.tokens import get_token_counter

    counter = get_token_counter()
    print(f"tokenizer: {counter.name}")

    files = iter_corpus()
    if not files:
        print("no corpus found -- run with --clone first")
        return 1

    by_repo: dict[str, int] = {}
    for repo, *_ in files:
        by_repo[repo] = by_repo.get(repo, 0) + 1
    print(f"corpus: {len(files)} files  {dict(sorted(by_repo.items()))}\n")

    v1_acc, v1_s = run_v1(files, counter)
    v1 = summarise("v1 (indentation heuristic)", v1_acc, v1_s, len(files))

    v2_acc, v2_s, v2_stats = run_v2(files, counter)
    v2 = summarise("v2 (AST / tree-sitter)", v2_acc, v2_s, len(files))

    rows = [
        ("chunks emitted", "chunks", "{:,}"),
        ("throughput (files/sec)", "files_per_sec", "{:.1f}"),
        ("decorator preservation", "decorator_preservation", "{:.1%}"),
        ("method isolation", "method_isolation", "{:.1%}"),
        ("tokens p50", "tokens_p50", "{:.0f}"),
        ("tokens p95", "tokens_p95", "{:.0f}"),
        ("tokens max", "tokens_max", "{:,.0f}"),
        ("chunks > encoder window", "over_encoder_window", "{:,}"),
        ("chunks > packing budget", "over_packing_budget", "{:,}"),
        ("  as % of chunks", "over_budget_pct", "{:.2f}%"),
        ("signature indexed [CIRCULAR]", "signature_indexed_CIRCULAR", "{:.1%}"),
        ("whole-def in 1 chunk [CIRCULAR]", "definition_coverage_CIRCULAR", "{:.1%}"),
    ]

    print(f"{'metric':<32} {'v1':>14} {'v2':>14}")
    print("-" * 62)
    for label, key, fmt in rows:
        a, b = v1.get(key), v2.get(key)
        fa = fmt.format(a) if a is not None else "--"
        fb = fmt.format(b) if b is not None else "--"
        print(f"{label:<32} {fa:>14} {fb:>14}")

    print(
        f"\nsample sizes: {v1['decorated_total']:,} decorated defs, "
        f"{v1['methods_total']:,} methods, {v1_acc.get('defs', 0):,} definitions"
    )
    print(
        f"v2 internals: parsed={v2_stats.files_parsed} windowed={v2_stats.files_windowed} "
        f"failed={v2_stats.files_failed} parse_errors={v2_stats.parse_errors} "
        f"split={v2_stats.oversized_split} merged={v2_stats.tiny_merged}"
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "tokenizer": counter.name,
                    "corpus_files": len(files),
                    "corpus_by_repo": by_repo,
                    "v1": v1,
                    "v2": v2,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
