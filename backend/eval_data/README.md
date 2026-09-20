# Evaluation data

Two retrieval evaluation sets, generated from the benchmark corpus in
`bench/corpus/` and versioned so that a result can be reproduced rather
than taken on trust.

| file | queries | task |
|---|---:|---|
| `retrieval_v1*.jsonl` | 1,038 | docstring -> function (semantic matching) |
| `symbol_v1*.jsonl` | 600 | identifier lookup over 150 unique symbols |

Each set is split into `_dev` and `_test`. Weights and thresholds are tuned
on dev; test is run once and reported. See BENCHMARKS.md section 7.

## Regenerating

```bash
python bench/bench_chunking.py --clone     # fetch the corpus
python bench/build_eval_set.py
python bench/build_symbol_eval_set.py
```

Deterministic given the same corpus commits. `gold_chunk_ids` derive from
the file path, so paths are stored POSIX-style: an earlier version emitted
Windows separators and the sets were not portable across platforms.

## Provenance and licensing

Queries are the first sentence of a docstring or doc-comment taken from
four permissively licensed open-source repositories, at the commits
recorded in BENCHMARKS.md:

| repo | licence |
|---|---|
| pallets/flask | BSD-3-Clause |
| encode/httpx | BSD-3-Clause |
| psf/requests | Apache-2.0 |
| gin-gonic/gin | MIT |

No source code is stored here -- only one sentence of documentation per
example, plus a symbol path and metadata. Reproducing the corpus requires
cloning those repositories.

## What these sets are not

Not human-labelled, and docstring phrasing is not how developers actually
ask questions. They are instruments for comparing systems and catching
regressions, not estimates of production quality. The limitations that
matter are listed under "Caveats that matter" in BENCHMARKS.md section 7.
