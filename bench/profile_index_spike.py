"""
Find what causes the transient memory spike during indexing.

The trajectory from bench_index_memory.py showed the shape clearly: resident
memory sits at 405-417 MB for the whole run, spikes once to ~530 MB, and
returns. It is not a leak, and nothing that scales with repository size --
it is one bounded event.

Guessing at it has already cost several wrong answers (the arena, the
allocator, the flush batch, the memory-pattern cache), so this measures per
file instead: RSS before and after chunking each file, sorted by the jump.
Whatever tops that list is the thing to bound.

Usage (Linux, or WSL on a Windows host):
    python bench/profile_index_spike.py --repo https://github.com/ayushhh1010/nanoserve
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"

CHILD = r'''
import json, os, subprocess, sys, tempfile
sys.path.insert(0, BACKEND_PATH)
os.chdir(BACKEND_PATH)

def rss():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return 0.0

from app.config import get_settings
from app.indexing.chunker import ASTChunker
from app.indexing.embedder import OnnxEmbedder
from app.indexing.pipeline import iter_source_files
from app.indexing.service import SharedEmbedder
from app.indexing.tokens import get_token_counter

cfg = get_settings()
embedder = SharedEmbedder(
    OnnxEmbedder(
        model_id=cfg.embedding_model_id,
        onnx_file=cfg.embedding_onnx_file,
        max_tokens=cfg.embedding_max_tokens,
        threads=cfg.embedding_threads or None,
    ),
    batch_size=cfg.indexer_embed_batch,
)
baseline = rss()

work = tempfile.mkdtemp(prefix="spike_")
root = Path_ = os.path.join(work, "repo")
subprocess.run(
    ["git", "-c", "core.symlinks=false", "clone", "--depth", "1",
     "--single-branch", "--no-recurse-submodules", "--no-tags", "--quiet",
     "--", REPO_URL, root],
    check=True,
)
after_clone = rss()

from pathlib import Path
chunker = ASTChunker(count_tokens=get_token_counter().count)

rows = []
peak_seen = baseline
for path in iter_source_files(Path(root)):
    before = rss()
    size = path.stat().st_size
    src = path.read_text(encoding="utf-8", errors="replace")
    after_read = rss()
    chunks = chunker.chunk_file("r", str(path.relative_to(root)), src)
    after_chunk = rss()
    # Embed them, which is what the worker does next.
    if chunks:
        embedder.embed([c.content for c in chunks])
    after_embed = rss()
    peak_seen = max(peak_seen, after_read, after_chunk, after_embed)
    rows.append({
        "file": str(path.relative_to(root)),
        "bytes": size,
        "chunks": len(chunks),
        "max_chunk_chars": max((len(c.content) for c in chunks), default=0),
        "before": round(before, 1),
        "after_read": round(after_read, 1),
        "after_chunk": round(after_chunk, 1),
        "after_embed": round(after_embed, 1),
        "jump_read": round(after_read - before, 1),
        "jump_chunk": round(after_chunk - after_read, 1),
        "jump_embed": round(after_embed - after_chunk, 1),
        "net": round(after_embed - before, 1),
    })
    del src, chunks

import shutil
shutil.rmtree(work, ignore_errors=True)
print(json.dumps({
    "baseline_mb": round(baseline, 1),
    "after_clone_mb": round(after_clone, 1),
    "peak_mb": round(peak_seen, 1),
    "files": rows,
}))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--repo", default="https://github.com/ayushhh1010/nanoserve")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    if not Path("/proc/self/status").exists():
        print("Reads /proc; run on Linux (or `wsl` on a Windows host).")
        return 2

    source = f"BACKEND_PATH = {str(BACKEND)!r}\nREPO_URL = {args.repo!r}\n" + CHILD
    proc = subprocess.run(  # noqa: S603
        [args.python, "-c", source], capture_output=True, text=True,
        cwd=str(BACKEND),
    )
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        return 1

    data = json.loads(proc.stdout.strip().splitlines()[-1])
    files = data["files"]
    print(f"model + tokenizer   {data['baseline_mb']:.0f} MB")
    print(f"after clone         {data['after_clone_mb']:.0f} MB")
    print(f"peak during walk    {data['peak_mb']:.0f} MB")
    print(f"files processed     {len(files)}")

    print("\nlargest RSS jumps, by file:")
    print(f"  {'file':<46}{'KB':>8}{'chunks':>7}{'read':>7}{'chunk':>7}"
          f"{'embed':>7}{'peak':>8}")
    worst = sorted(files, key=lambda r: max(r["after_embed"], r["after_chunk"]),
                   reverse=True)[:12]
    for r in worst:
        print(f"  {r['file'][:44]:<46}{r['bytes'] // 1024:>8}{r['chunks']:>7}"
              f"{r['jump_read']:>7.0f}{r['jump_chunk']:>7.0f}"
              f"{r['jump_embed']:>7.0f}{max(r['after_chunk'], r['after_embed']):>8.0f}")

    big = max(files, key=lambda r: r["max_chunk_chars"])
    print(f"\nlongest single chunk: {big['max_chunk_chars']:,} chars "
          f"in {big['file']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
