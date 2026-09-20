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
# 1. dependency weight, 2. ingestion memory, 3. chunking
python bench/bench_import_rss.py --compare --reps 3
python bench/bench_ingest_memory.py --chunks 10000 25000 50000
python bench/bench_chunking.py --clone --run

# 4. pgvector capability probe, 5. quantisation recall, 6. encoder throughput
python bench/verify_pgvector.py
python bench/bench_quantization.py --chunks 8000 --queries 200
python bench/bench_embed_model.py --chunks 400

# 7. retrieval evaluation (build the sets, then ablate)
python bench/build_eval_set.py
python bench/build_symbol_eval_set.py
python bench/run_retrieval_eval.py --split dev --dataset retrieval_v1
python bench/run_retrieval_eval.py --split dev --dataset symbol_v1
```

Everything runs against an ephemeral PostgreSQL 18.6 + pgvector 0.8.6 from
the `embedded-postgres` wheel -- no Docker daemon, so the suite runs on a
clean checkout and in CI unchanged.

---

## 1. Import-time RSS — the application, measured

`bench/bench_import_rss.py` for dependency sets; `import app.main` measured
directly, 3 repetitions each, v1 from a git worktree at the last commit
where it still existed.

### The application

| | `import app.main` | import time |
|---|---:|---:|
| v1 (chromadb + langchain + redis + gitpython) | **140.0 MB** | 2.3 s |
| v2 | **59.3 MB** | 0.9 s |
| **reclaimed** | **80.7 MB (58%)** | **2.5x faster** |

On a 512 MB container that is 80 MB returned to the request path before a
single request is served. The import time matters separately: on a platform
that scales to zero, it is paid on every cold start.

### Dependency sets, for attribution

Baseline (bare interpreter + psutil) is 18.0 MB and excluded.

| Dependency set | Import RSS |
|---|---|
| v1 serving set | 124.6 MB |
| v2 serving set | 88.1 MB |
| Indexer set, out-of-process (tree-sitter + onnxruntime) | 28.7 MB |

Largest v1 contributors: `chromadb` 49.3 MB, `fastapi` 29.5 MB,
`sqlalchemy.ext.asyncio` 17.3 MB, `langchain_community.llms` 11.4 MB.

Note the application (59.3 MB) is *lighter* than its own dependency set
(88.1 MB): `app.main` does not import langgraph eagerly. The set is an
upper bound on what the process can reach, not what it does reach.

> **Two corrections, kept because they were load-bearing.**
>
> The architecture proposal estimated ~470 MB for v1 and attributed ~250 MB
> to ChromaDB. Measured: **140.0 MB for the whole application, 49.3 MB for
> ChromaDB** — the estimate was over 3x too high, and the memory argument
> for replacing ChromaDB was much weaker than claimed. The replacement
> still stands on the ground that was always stronger: vector writes and
> the `status="ready"` flag cannot be made transactional across two stores,
> which is the cause of the silent data-loss bug in §8.
>
> An earlier version of this section reported 88.1 MB as the v2 serving
> figure while `app.main` still imported chromadb and langchain — a
> projection presented as a measurement. It is now the measured 59.3 MB,
> and `tests/test_imports.py::test_v2_modules_do_not_pull_in_legacy_dependencies`
> keeps it honest by importing every v2 module in a subprocess with
> chromadb, langchain, redis, gitpython, jose and passlib blocked at the
> import hook.

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

## 4. pgvector capabilities, verified rather than assumed

`bench/verify_pgvector.py`, PostgreSQL 18.6 + pgvector 0.8.6, 5,000 rows at
768 dimensions. 16/16 checks pass. Written because the schema rests on
specifics that secondary sources get wrong.

Corrections this produced:

| Commonly stated | Actually |
|---|---|
| `vector` is limited to 2,000 dimensions | That is the **index** ceiling; the type goes to 16,000. `halfvec` indexes to 4,000. |
| (not mentioned) | `binary_quantize()` accepts `halfvec` input, not only `vector` |
| (not mentioned) | `hnsw.iterative_scan` exists and matters here -- every query filters by `repo_id`, and without it HNSW under-returns |

Measured storage, which is what drove the index layout:

| | per row | HNSW index (per 5,000 rows) |
|---|---:|---:|
| `halfvec(768)` | 1,544 B | 10,008 kB |
| `bit(768)` | 101 B | 1,952 kB |

Indexing both would cost ~3,594 B/row (719 MB at 200k chunks, over Supabase's
500 MB free tier). Indexing only the bit column costs ~2,045 B/row (409 MB).
The halfvec column stays unindexed and is read only to rescore a small
candidate set.

## 5. Binary quantisation recall -- and a scare that meant nothing

`bench/bench_quantization.py`. 3,601 chunks from the corpus, embedded with the
real model (`jina-embeddings-v2-base-code`, fp32 ONNX), 200 held-out queries,
recall@10 against exact cosine over the same vectors.

First, a sanity check that the embeddings are structured at all: mean
similarity of a query to its nearest *other* chunk is **0.796** (random
vectors would give ~0.0).

Recall is measured against exact fp32 top-10, over 200 queries (2,000
slots), so one displaced result is worth 0.0005.

| method | over-fetch | recall@10 | ms/query |
|---|---:|---:|---:|
| halfvec exact, no ANN | - | 99.95% | - |
| binary + rescore | 10 | 83.35% | 4.1 |
| binary + rescore | 50 | 99.80% | 4.0 |
| **binary + rescore** | **100** | **99.95%** | **3.7** |
| binary + rescore | 200 | 100.00% | 4.4 |
| binary + rescore | 500 | 99.95% | 10.1 |
| binary + rescore | 1000 | 99.95% | 5.2 |
| binary + rescore | 2000 | 100.00% | 5.3 |

**From an over-fetch of 100, binary quantisation with rescoring costs at
most one result slot in 2,000.** It does not improve monotonically to a
clean 1.0 -- it oscillates between 0.9995 and 1.0000, which is what a
single borderline tie looks like, not a trend. Calling it "lossless" would
be rounding 99.95% up; the honest claim is that the loss is at the
resolution limit of a 200-query sample.

Note also that exact halfvec scores the same 99.95% against fp32. The
binary stage is therefore not the lossy part -- at fetch>=100 the ANN
returns whatever halfvec itself would have returned, so the residual comes
from fp16 storage, not from quantisation. The storage layout in section 4
is free in quality terms, and `DEFAULT_ANN_CANDIDATES = 400` sits four
times above where recall saturates.

The ms/query column is single-threaded NumPy over 3,601 vectors and is
*not* a projection of Postgres HNSW latency; it is here only to show that
over-fetching further is not free. The 500 row at 10.1 ms against 5.2 ms
at 1000 is measurement noise on an unpinned CPU, which is itself a reason
not to read these as production numbers.

### The 18.5% that meant nothing

An earlier run of the same pipeline reported **recall@10 = 18.5%** and briefly
looked like it had killed the design. That run used random Gaussian vectors.
Random vectors in 768 dimensions are near-orthogonal and carry no structure
for quantisation to preserve, so the number measured nothing about real
embeddings. It is recorded here because the failure mode -- a synthetic
benchmark producing a confident, wrong, decision-shaped number -- is more
dangerous than having no benchmark.

### A negative result: centroid thresholding does not help

`binary_quantize()` in pgvector thresholds at zero. The embedder was written
with a centroid-thresholding alternative on the reasoning that encoder outputs
carry per-dimension bias, and thresholding a biased dimension at zero discards
its signal.

Measured: per-dimension |mean| averages **0.0114** (max 0.0711) -- these
embeddings are already near-centred -- and centroid thresholding is
**indistinguishable** from zero thresholding:

| over-fetch | zero | centroid |
|---:|---:|---:|
| 10 | 0.8335 | 0.8275 |
| 50 | 0.9980 | 0.9975 |
| 100 | 0.9995 | 1.0000 |
| 200 | 1.0000 | 1.0000 |
| 500 | 0.9995 | 1.0000 |
| 1000 | 0.9995 | 1.0000 |
| 2000 | 1.0000 | 0.9995 |

The largest gap at any factor is **0.0005** -- one slot out of 200 queries
x 10 -- and its sign flips (centroid leads at 500, trails at 2000). That is
sampling noise, not an effect. Exact halfvec itself scores 0.9995 against
fp32, so from fetch=100 up, binary-plus-rescore is already at the ceiling
the storage format sets, and no thresholding change can move it.

So the ingestion path uses pgvector's SQL `binary_quantize()` directly. The
centroid argument is kept as a hedge for a future model swap, with the claim
in its docstring corrected to say it was tested and found unnecessary.

## 6. Embedding throughput -- 3.86x, measured, free

The first measurement of the fp32 embedder was **0.90 chunks/sec**; indexing
the 3,601-chunk corpus took 53-67 minutes. That is not a viable ingestion
time, and on Modal -- billed per second -- it is the dominant cost of the
whole pipeline.

### Model variant (`bench/bench_embed_model.py`, 400 chunks)

fp32 output is the reference; each variant is scored as a retrieval index
against fp32's own top-10, because a faster encoder that reorders results is
not a win.

| variant | chunks/sec | speedup | recall@10 vs fp32 | mean cos to fp32 |
|---|---:|---:|---:|---:|
| fp32 | 0.90 | 1.00x | 100.0% | 1.0000 |
| **fp16** | **1.22** | **1.36x** | **100.0%** | **1.0000** |
| int8 | 1.89 | 2.10x | 91.0% | 0.9809 |

fp16 is free and is now the default. **int8 is not free** -- 9% of top-10
results change -- and is a lever to pull only if throughput becomes critical
*and* that 9% is shown not to matter end to end.

### Length-sorted batching

Every sequence in a batch is padded to the longest one in it, and chunk
lengths are heavily skewed: p50 117 tokens, p90 325, p99 935, max 1,065. One
long chunk pads fifteen short ones up to its length.

Counted directly on the corpus at batch 16:

| batching | padded tokens processed | real tokens | wasted work |
|---|---:|---:|---:|
| arrival order | 1,731,831 | 593,384 | **2.92x** |
| length-sorted | 601,401 | 593,384 | 1.01x |

Measured end to end on 400 chunks with fp16:

| | chunks/sec |
|---|---:|
| arrival order | 1.26 |
| **length-sorted** | **3.47** |
| speedup | **2.75x** |

Output is unchanged: minimum cosine between the two orderings is
**1.000000** across all 400 chunks. That is safe only because padding
invariance is a *verified* property of the pooling implementation rather
than an assumption -- `tests/test_embedder.py::test_padding_does_not_change_an_embedding`
embeds a short text alone and again in a padded batch and requires
cos > 0.9999. Without that guarantee, reordering would silently change
every vector.

### Combined

| | chunks/sec | 3,601-chunk repo |
|---|---:|---:|
| fp32, arrival order (starting point) | 0.90 | 66.7 min |
| fp16, length-sorted | **3.47** | **17.3 min** |
| **improvement** | **3.86x** | |

No quality cost at any step. The remaining cost is CPU-bound on 2 effective
cores; more cores are the next lever and are what the indexer host provides.

### Pooling correctness, verified rather than assumed

Everything above is meaningless if the embeddings are wrong, and a pooling
bug does not crash -- it degrades every metric by a few percent, silently.
The model's `1_Pooling/config.json` declares `pooling_mode_mean_tokens: true`
(cls and max false), which is what the implementation does. The decisive
check is padding invariance, described above: it fails loudly if the mask is
applied incorrectly, and it needs no reference implementation to run.

## 7. Retrieval evaluation -- the hybrid design was wrong, and routing fixes it

`bench/run_retrieval_eval.py`. Weights were tuned on the dev splits; the
held-out test splits were run **once**, after tuning was complete, and are
reported below as the headline. 3,601 indexed chunks with docstrings
stripped, real PostgreSQL + pgvector via the production Alembic migrations,
every configuration over identical queries, paired bootstrap with
Holm-corrected p-values.

Two evaluation sets, because one of them could not see half the problem:

  **retrieval_v1** 519 queries, docstring -> function (semantic matching)
  **symbol_v1**    300 queries, identifier lookup over 150 unique symbols

### Docstring / semantic queries (n=519)

| configuration | recall@1 | recall@10 | MRR | p50 ms |
|---|---:|---:|---:|---:|
| **dense only** | 0.636 | **0.908** | **0.737** | 13.0 |
| lexical only | 0.052 | 0.339 | 0.143 | 27.4 |
| symbol only | 0.210 | 0.285 | 0.234 | 1.1 |
| hybrid w=1,1,1 | 0.410 | 0.813 | 0.535 | 24.2 |
| hybrid w=1,.1,.1 | 0.570 | 0.906 | 0.688 | 23.1 |
| **ROUTED** | 0.611 | 0.902 | 0.716 | **7.1** |

### Identifier-lookup queries (n=300)

| configuration | recall@1 | recall@10 | MRR | p50 ms |
|---|---:|---:|---:|---:|
| dense only | 0.690 | 0.953 | 0.794 | 7.8 |
| symbol only | 0.943 | 0.953 | 0.948 | 15.3 |
| hybrid w=.5,.1,2 | 0.943 | 0.990 | 0.960 | 50.9 |
| **ROUTED** | **0.953** | **0.997** | **0.974** | 52.8 |

### ROUTED vs dense-only, paired and Holm-corrected

| set | metric | delta | 95% CI | p_adj | |
|---|---|---:|---|---:|---|
| semantic | recall@10 | −0.006 | [−0.015, +0.002] | 0.741 | no difference |
| semantic | MRR | **−0.021** | [−0.034, −0.009] | 0.003 | significantly worse |
| identifier | recall@10 | **+0.043** | [+0.023, +0.067] | 0.001 | significantly better |
| identifier | MRR | **+0.180** | [+0.146, +0.215] | <0.0001 | significantly better |

### Held-out test splits

| set | configuration | recall@1 | recall@10 | MRR |
|---|---|---:|---:|---:|
| semantic (n=519) | dense only | 0.617 | 0.913 | 0.729 |
| semantic | **ROUTED** | 0.592 | 0.911 | 0.707 |
| identifier (n=300) | dense only | 0.680 | 0.910 | 0.774 |
| identifier | symbol only | 0.943 | 0.947 | 0.945 |
| identifier | **ROUTED** | **0.937** | **0.997** | **0.958** |

| set | metric | delta | 95% CI | p_adj | |
|---|---|---:|---|---:|---|
| semantic | recall@10 | −0.002 | [−0.010, +0.006] | 1.000 | no difference |
| semantic | MRR | −0.022 | [−0.036, −0.010] | 0.005 | worse |
| identifier | recall@10 | **+0.087** | [+0.057, +0.120] | <0.0001 | better |
| identifier | **MRR** | **+0.184** | [+0.150, +0.219] | <0.0001 | **better** |

### Replicated on an independent resample

The evaluation sets were regenerated after a portability fix: file paths
were being stored with Windows separators, and since `chunk_id` derives
from the path, the ids -- and therefore the seeded dev/test split and the
symbol sampling -- were platform-specific.

Regenerating drew **different** test queries and a different 150 symbols.
Running the same configuration on them is a replication, not a rerun:

| | first sample | independent resample |
|---|---:|---:|
| semantic MRR delta | −0.026 (p<0.0001) | −0.022 (p=0.005) |
| identifier MRR delta | +0.171 (p<0.0001) | +0.184 (p<0.0001) |
| identifier recall@10 delta | +0.010 (ns) | +0.087 (p<0.0001) |

Same direction, similar magnitude, on disjoint samples. That is better
evidence than either run alone, and it is worth more than the tidiness of
reporting a single number -- so both are kept.

Dev predicted −0.021 / +0.180; the two test samples measured −0.026 / +0.171
and −0.022 / +0.184. The tuning generalised.

One thing test shows more clearly than dev: **the benefit is in ranking.**
On identifier queries recall@1 goes 0.680 -> 0.937. With a context budget
that fits five or six chunks, rank is what matters.

### What this changed

The three-arm hybrid I designed was **wrong**, and the first evaluation said
so unambiguously: on semantic queries *every* fixed hybrid configuration was
significantly worse than dense alone on MRR, equal-weight RRF worst of all
(0.535 vs 0.737). The mechanism is that unweighted RRF assumes comparable
arms -- a rank-1 hit from an arm with 0.285 recall scores exactly as much as
one from an arm with 0.908.

Deleting the arms on that evidence would have been over-generalising, because
the docstring set contains almost no identifier lookups -- the thing the
symbol arm exists for. Building the second set inverted the conclusion: there
the symbol-heavy hybrid beats dense by +0.166 MRR.

So the configuration is chosen per query. Routing is not a compromise between
the two: it **beats every fixed configuration on the identifier set**
(MRR 0.974 vs the best fixed 0.960), because the few genuinely-semantic
queries in that set get sent to dense, which handles them better.

### The cost, stated plainly

Routing is **significantly worse than dense on semantic MRR** -- −0.022 on
test (p=0.005), −0.026 on the first sample. That is the price of a 7.9% prose misroute rate.

The trade is about −0.022 MRR on semantic against +0.184 on identifier, so
it breaks even at roughly **11% identifier queries** and wins above that. We do
not have production traffic to know that share. This is the one number in
this document resting on an assumption about usage rather than a
measurement, and it is the first thing real traffic should settle.

### The leakage control, measured rather than asserted

`--keep-docstrings` runs the same evaluation without stripping, so the size
of the leak is a number rather than an argument:

| arm | stripped (controlled) | kept (leaky) | inflation |
|---|---:|---:|---:|
| dense recall@10 | 0.908 | 0.973 | +0.065 |
| dense MRR | 0.737 | 0.882 | +0.145 |
| **lexical recall@10** | **0.339** | **0.603** | **+0.264** |
| lexical MRR | 0.143 | 0.264 | +0.121 |

The lexical arm's recall **nearly doubles** when the query text is left in
the indexed chunk, exactly as predicted -- it is matching the query verbatim.

This is not only an inflation of absolute numbers. Uncontrolled, the gap
between dense and lexical narrows from 2.7x to 1.6x, which would have made
the lexical arm look far more competitive than it is and could plausibly
have changed the routing weights. The control is load-bearing, not hygiene.

### Caveats that matter

**Router accuracy on symbol_v1 is circular.** That set is built from four
templates, and the router's intent regex contains phrases from those same
templates, so its 99% routing rate there measures nothing. A held-out check
on 15 phrasings that did *not* inform the regex ("which file has X", "def of
X", "what calls X", plus prose controls) caught 10/10 lookups with 0/5 false
positives, via the short-query + identifier-shape path. n=15 is a smoke test,
not a measurement.

**Run-to-run variance is not captured by the confidence intervals.** The CIs
are over queries within one run. HNSW is approximate and the index is rebuilt
per run, so the same configuration moves by ~0.01 between runs (symbol-only
recall@10 measured 0.274 and 0.285 on two runs). Differences smaller than
that should not be read as real even when the within-run CI excludes zero.

**Neither set is human-labelled**, and docstring phrasing is not how people
actually ask questions. These are instruments for comparing systems and
catching regressions, not estimates of production quality.

## 8. Open defect: index/metadata state split

Not a benchmark — a bug found by reading `render.yaml` against `config.py`.

`render.yaml` declares no `disk:`, so the filesystem is ephemeral, but
`chroma_persist_dir` and `repos_dir` both point at `./data`. On restart the
index is gone while Postgres still reports `status="ready"`.
`vectorstore.search` catches the missing collection, logs a warning and returns
`[]`; `build_context_string` then returns `""`, and the model answers with no
repository context at all. No error surfaces to the user.

Fixed by moving vectors into Postgres, where the index write and the status
flag commit together.
