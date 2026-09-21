# Clarix

**Ask questions about a codebase and get answers grounded in its actual
source, with citations.**

Clarix indexes a Git repository by parsing it into syntax-aware chunks,
retrieves against those chunks with a routed hybrid search over pgvector,
and answers with a language model that is given only the retrieved code.

The thing that distinguishes this from a weekend RAG demo is that the
design decisions are **measured rather than asserted**. Retrieval quality is
reported on a held-out split with confidence intervals and corrected
p-values. Memory and throughput figures come from benchmarks in `bench/`,
and `bench/check_documented_numbers.py` re-derives every number quoted in
this repository from the artifact it came from, so prose cannot drift away
from evidence.

Some of those measurements contradicted decisions that had already shipped.
Those are documented as corrections in [BENCHMARKS.md](BENCHMARKS.md) rather
than quietly fixed — a three-arm hybrid that lost to plain dense retrieval,
an "fp16 is 36% faster" claim that turned out to be 24% *slower*, and a
"lossless" quantisation result that was 99.95% rounded up.

---

## Architecture

```
  Next.js frontend
         │
         ▼
  ┌─────────────────┐        ┌──────────────────────────┐
  │   API (59 MB)   │        │    Indexer (432 MB)      │
  │                 │        │                          │
  │  auth           │ POST   │  /embed  ── query vectors│
  │  retrieval  ────┼───────▶│                          │
  │  generation     │ /embed │  worker  ── clone, chunk,│
  │                 │        │             embed, store │
  └────────┬────────┘        └─────────────┬────────────┘
           │                               │
           └───────────┬───────────────────┘
                       ▼
            PostgreSQL 16+ with pgvector ≥ 0.7
            chunks · embedding cache · job queue
```

**Two processes, because they have different shapes.** The API is always-on
and holds no model: 59 MB of import RSS. The indexer holds the ONNX weights
and peaks at 432 MB, measured on Linux with the worker running and a cold
model download (`bench/bench_service_memory.py`).

The indexer runs int8 weights at a 384-token cap, and that is not a detail —
fp16 needs about a gigabyte just to load, because CPUs have no fp16 kernels
and ONNX Runtime upcasts every weight to fp32. An earlier version shipped
fp16 on the strength of a measurement taken on Windows against a cached
model with the worker off; the container was killed for exceeding its
memory limit. The quality cost of int8 was then measured end to end rather
than inferred: identifier queries are unchanged, semantic MRR moves 0.707 →
0.696. Both the mistake and the correction are in BENCHMARKS.md section 8.

The indexer serves query embeddings *and* runs the indexing worker, because
the worker already holds the model, so answering queries from it is free.

There is no Redis and no agent framework. Both were in v1; neither did
anything. See "What was removed" below.

---

## How it works

### Indexing

Repositories are parsed with tree-sitter across **13 languages** and chunked
along syntax boundaries, not line windows. A class becomes a header chunk
plus one chunk per method; decorators and leading comments stay attached to
what they decorate (68.4% → **100%** decorator preservation against the v1
line-window chunker).

Each chunk is embedded locally with `jina-embeddings-v2-base-code` (int8,
384-token cap — see Deployment) and
stored twice: as `halfvec(768)` for exact scoring and as `bit(768)` for
approximate search. Ingestion streams — chunks are embedded and flushed in
batches rather than accumulated — because v1 materialised every vector as
Python lists at 10.2× the float32 payload and exhausted 512 MB at roughly
22,000 chunks.

Embeddings are cached by content hash, so re-indexing a repository only pays
for chunks whose text actually changed.

### Retrieval

Three arms run against Postgres:

| arm | mechanism |
|---|---|
| dense | binary Hamming ANN over `bit(768)`, then exact rescore on `halfvec` |
| lexical | weighted `tsvector`, with identifiers split on camelCase |
| symbol | exact symbol match plus trigram similarity |

Results are fused with **weighted Reciprocal Rank Fusion**, and the weights
are **routed by query type** — which is the central empirical finding here.
No single configuration wins both kinds of query:

| query type | best configuration |
|---|---|
| natural-language ("how does retry work") | dense alone |
| identifier lookup (`handle_full_index`) | symbol-weighted hybrid |

Held-out test results, against the dense-only baseline v1 used:

| set | metric | Δ | 95% CI | p_adj |
|---|---|---:|---|---:|
| identifier (n=300) | MRR | **+0.184** | [+0.150, +0.219] | <0.0001 |
| identifier | recall@10 | **+0.087** | [+0.057, +0.120] | <0.0001 |
| semantic (n=519) | MRR | −0.022 | [−0.036, −0.010] | 0.005 |

On identifier lookups recall@1 goes **0.680 → 0.937**. With a context budget
that fits five or six chunks, rank is what matters. The semantic loss is
real and is reported rather than buried; the trade breaks even near 11%
identifier traffic.

These numbers were **replicated on an independent resample** after a
path-portability fix changed the split. Both runs are in BENCHMARKS.md.

### Generation

Context is packed to a hard **4,000-token** budget counted with the real
tokenizer. v1 estimated `len(text)//4` against a 12,000-token budget, which
exceeds the free-tier context limits it was running under — so it failed
outright rather than merely costing more.

Generation routes across free-tier providers with per-provider quota
tracking and circuit breakers. If every provider is exhausted, the endpoint
returns the retrieved code with citations and `degraded: true` rather than a
500, because the citations are the part the user can still act on.

---

## Quick start

### Prerequisites

- Python 3.11 or 3.12
- Node.js 18+
- PostgreSQL with **pgvector ≥ 0.7.0** (Docker Compose provides it)

pgvector 0.7.0 is a hard floor: it introduced `halfvec` and
`binary_quantize`, which the schema depends on. The migration checks the
installed version and fails with a clear message rather than dying later on
a missing type.

### 1. Database

```bash
docker compose up -d
```

This starts `pgvector/pgvector:pg17`. The stock `postgres` image will not
work — it ships no pgvector.

### 2. Backend

```bash
cd backend
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install '.[indexer,dev]'
cp .env.example .env
alembic upgrade head
```

Run the two processes in separate terminals:

```bash
uvicorn app.main:app --reload --port 8000
```

```bash
python -m app.indexing --port 8081
```

Then point the API at the indexer by setting `EMBEDDING_ENDPOINT` in `.env`:

```
EMBEDDING_ENDPOINT=http://localhost:8081/embed
EMBEDDING_API_KEY=any-shared-secret
```

**Without `EMBEDDING_ENDPOINT` the API still works, but the dense arm is
disabled** and only lexical and symbol retrieval run. That degradation is
deliberate and visible in `/health` and in each response's `route` field.

### 3. Frontend

```bash
cd frontend && npm install && npm run dev
```

---

## Deployment

`render.yaml` defines both services. The indexer needs an instance with at
least 512 MB and peaks at 432 MB of it — 105 MB of headroom, which matters
because the same measurement varies by ~30 MB between runs. Set
`EMBEDDING_API_KEY` to the same value on both services; the indexer refuses
to start in production without one, since an open model endpoint is free
compute for whoever finds it.

Dependencies come from `pyproject.toml`. There is no `requirements.txt`: it
had drifted to the v1 set (chromadb, langgraph, redis) that nothing imports
any more, so deploys installed ~300 MB of dead weight and missed what v2
needs.

---

## Testing

```bash
cd backend && pytest -q
```

**421 tests, no Docker required.** PostgreSQL 18.6 with pgvector 0.8.6 comes
from the `embedded-postgres` wheel, so schema and retrieval tests run real
SQL against a real server anywhere `pip install` works — a suite that needs
a daemon is a suite that gets skipped.

```bash
python bench/check_documented_numbers.py
```

Re-derives every number quoted in docstrings and BENCHMARKS.md from its JSON
artifact. This exists because six documented figures went stale at once when
the evaluation sets were regenerated.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/repo/upload` | Clone and index a repository |
| `GET` | `/api/repo/{id}/status` | Indexing status |
| `GET` | `/api/repo/{id}/files` | List indexed files |
| `DELETE` | `/api/repo/{id}` | Delete a repository |
| `POST` | `/api/chat` | Ask a question |
| `POST` | `/api/chat/stream` | Ask a question (SSE) |
| `GET` | `/api/chat/{id}/history` | Conversation history |
| `POST` | `/api/agent/run` | Retrieval + generation with step trace |
| `GET` | `/health` | Liveness, provider and breaker state |

Indexer process: `POST /embed`, `GET /health`.

---

## What was removed, and why

**The LangGraph agent.** Its tool path was unreachable: `needs_tools` was a
substring match on the model's own prose, gated behind an empty-context
check that retrieval almost never produced, and the tool agent's prompt
said "Do NOT attempt to call any tools". The graph had no cycles. It cost
three LLM calls to produce what one call produces — on free-tier quotas, the
difference between ~200 questions a day and ~65.

**ChromaDB.** Vectors lived on ephemeral disk while status lived in
Postgres, so a restart produced repositories marked `ready` with no index
behind them, and retrieval silently returned nothing. Vector writes are now
transactional with the status flip.

**Redis.** Declared as a memory layer. Nothing ever wrote to it.

A real agent loop — with cycles, a relevance grader and genuine tool calls —
is worth building. It should be justified against this single-pass baseline
by the evaluation harness before it ships, which is exactly what v1 never
did.

---

## Honest limitations

- The worker and API have not been run against a real managed Postgres or
  real provider keys end to end; everything is verified against
  `embedded-postgres` and fakes.
- Evaluation uses docstring-derived weak supervision, not human relevance
  judgements. Docstrings are stripped from the indexed text to prevent
  leakage — keeping them inflates lexical recall@10 from 0.339 to 0.603 —
  but the queries still come from the same distribution as the code.
- Indexing throughput is **2.3 chunks/sec** in the deployed configuration.
  A 3,600-chunk repository takes around 26 minutes. The lever is more cores,
  not more memory.
- The free-tier instance sleeps when idle on most hosts, so the first
  request after a quiet period pays a cold start.

---

## Further reading

[BENCHMARKS.md](BENCHMARKS.md) — every measurement, including the ones that
went against the design, with methodology and the corrections that followed.
