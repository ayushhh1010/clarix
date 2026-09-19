# Benchmarks

Every number here was produced by a script in `bench/` on this machine. Nothing
is estimated. Where a measurement contradicted a prior estimate, the estimate is
recorded alongside it.

**Environment.** Windows 11, Python 3.11.9, 2026-09-19.
**Corpus.** Shallow clones, 278 code files:
`flask@d73fa1c` (83) · `gin@3b08cd7` (98) · `httpx@b5addb6` (60) · `requests@dae7ef6` (37).
**Tokenizer.** `jinaai/jina-embeddings-v2-base-code` (real tokenizer, not an estimate).

Reproduce:

```bash
python bench/bench_import_rss.py --compare --reps 3
python bench/bench_ingest_memory.py --chunks 10000 25000 50000
python bench/bench_chunking.py --clone --run
```

---

## 1. Import-time RSS — a prior estimate was wrong by 3.8×

`bench/bench_import_rss.py`, 3 repetitions, median. Baseline (bare interpreter
+ psutil) is 18.0 MB and is excluded from every figure below.

| Dependency set | Import RSS |
|---|---|
| v1 serving set (chromadb + langchain family) | **124.6 MB** |
| v2 serving set (pgvector + httpx + langgraph core) | **88.1 MB** |
| Reclaimed | **36.5 MB (29%)** |
| Indexer set, out-of-process (tree-sitter + onnxruntime) | 28.7 MB |

Largest single contributors, v1: `chromadb` 49.3 MB, `fastapi` 29.5 MB,
`sqlalchemy.ext.asyncio` 17.3 MB, `langchain_community.llms` 11.4 MB.

> **Correction.** The architecture proposal estimated ~470 MB for v1 and
> attributed ~250 MB to ChromaDB. Measured: **124.6 MB total, 49.3 MB for
> ChromaDB** — the estimate was 3.8× too high.
>
> This materially weakens the *memory* argument for replacing ChromaDB. The
> replacement still stands, on the ground that was always stronger: vector
> writes and the `status="ready"` flag cannot be made transactional across two
> stores, which is the cause of the silent data-loss bug in §4. 36 MB is a
> real but secondary benefit.

## 2. What actually caused the OOM

If imports cost 125 MB, the 512 MB ceiling was not breached by imports — so
commit `46925d0` ("replace local embeddings with HF Inference API to fix OOM on
Render free tier") was treating a symptom. `bench/bench_ingest_memory.py`
measures the pipeline's live data structures instead.

`run_ingestion_pipeline` holds `parsed_files`, `chunks` and `embeddings` alive
simultaneously when it calls `store_chunks`. Measured peak RSS for that shape,
384-d vectors:

| chunks | chunk text | `embeddings` as `list[list[float]]` | **peak** |
|---:|---:|---:|---:|
| 10,000 | 15 MB | 149 MB | **164 MB** |
| 25,000 | 36 MB | 373 MB | **409 MB** |
| 50,000 | 72 MB | 747 MB | **819 MB** |

The embedder returns `list[list[float]]`. A Python list of 384 floats is not
1,536 bytes — every element is a boxed 24-byte object behind an 8-byte pointer:

| chunks | actual | same data as f32 | overhead |
|---:|---:|---:|---:|
| 10,000 | 149 MB | 14.6 MB | **10.2×** |
| 25,000 | 373 MB | 36.6 MB | **10.2×** |
| 50,000 | 747 MB | 73.2 MB | **10.2×** |

Same vectors, other representations:

| representation | 50,000 chunks |
|---|---:|
| `list[list[float]]` (v1) | 747 MB |
| numpy float32 | 73 MB |
| numpy float16 | 37 MB |
| int8 | 18.3 MB |
| binary, 1 bit/dim | **2.3 MB** |

**Conclusion.** With ~143 MB resident after imports, the v1 shape exhausts a
512 MB container at roughly **22,000 chunks** — one medium repository. The fix
is streaming batches and a numpy representation, not dependency removal. Moving
embeddings off-box changed *when* the pipeline died, not *why*.

Binary quantisation being 325× smaller than the v1 representation is also what
makes the index fit Supabase's 500 MB free tier.

## 3. Chunking: v1 (indentation heuristic) vs v2 (tree-sitter)

`bench/bench_chunking.py`. Ground truth is enumerated by tree-sitter *in the
benchmark*, separately from both chunkers.

| metric | v1 | v2 |
|---|---:|---:|
| chunks emitted | 2,258 | 3,601 |
| **decorator preservation** (n=1,034) | **68.4%** | **100.0%** |
| **method isolation** (n=1,414) | **0.0%** | **91.0%** |
| tokens p50 | 137 | 117 |
| tokens p95 | 637 | 466 |
| **tokens max** | **21,238** | **1,065** |
| chunks over encoder window (8,192) | 3 | **0** |
| chunks over packing budget (1,024) | 45 (1.99%) | 4 (0.11%) |
| throughput (files/sec) | 46.1 | 46.3 |
| signature indexed *(circular)* | 100.0% | 100.0% |
| whole definition in one chunk *(circular)* | 94.9% | 96.0% |

### Reading these honestly

**Method isolation 0.0%** is the headline. v1's `_chunk_by_structure` consumes a
class to its next dedent and then advances past the body, so of 1,414 methods in
the corpus, **zero** were independently retrievable. Every method query had to
match a whole-class chunk.

**Decorator preservation 68.4%** confirms the defect but is less severe than
predicted: a third of decorated definitions lost their decorators, not all of
them, because some are incidentally covered by a larger enclosing chunk.

**tokens max 21,238** is a silent correctness failure, not a size complaint.
That chunk is 2.6× the encoder's 8,192-token window, so the embedder truncated
it — the tail was indexed as if it did not exist, with no error raised.

**Throughput is a wash (46.1 vs 46.3), not a win.** An earlier run showed
28.4 vs 73.0; that was warm-up ordering — v1 runs first and absorbs tree-sitter
grammar loading for the ground-truth pass. No speedup is claimed.

**v2 emits 59% more chunks**, which is a real cost: more embedding tokens and a
larger index, in exchange for method-level granularity. On a free embedding
budget this tradeoff should be revisited if it binds.

**The two metrics marked *circular* should not be quoted as evidence for v2.**
v2 finds boundaries with the same grammar the ground truth uses, so it is
graded on its own definitions. Note v1 scores *higher* on whole-definition
coverage in the first place — its oversized class chunks trivially contain
everything — which is exactly why that metric is weak.

### Not yet measured

Retrieval quality. Better chunk boundaries are expected to improve recall, but
that is a hypothesis until the eval harness exists. No recall or nDCG number is
claimed here.

## 4. Open defect: index/metadata state split

Not a benchmark — a bug found by reading `render.yaml` against `config.py`.

`render.yaml` declares no `disk:`, so the filesystem is ephemeral, but
`chroma_persist_dir` and `repos_dir` both point at `./data`. On restart the
index is gone while Postgres still reports `status="ready"`.
`vectorstore.search` catches the missing collection, logs a warning and returns
`[]`; `build_context_string` then returns `""`, and the model answers with no
repository context at all. No error surfaces to the user.

Fixed by moving vectors into Postgres, where the index write and the status
flag commit together.
