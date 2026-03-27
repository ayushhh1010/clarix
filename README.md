# AI Engineering Copilot

**A production-grade Agentic AI backend for understanding, debugging, and modifying codebases — powered by LangGraph, RAG, and Llama 3.**

> Think of it as a mini Cursor + Devin backend.

---

## Architecture

```
    Frontend (Next.js)
          │
          ▼
    Backend API (FastAPI)
          │
          ▼
    Agent Orchestration (LangGraph)
          │
    ┌─────┼─────────────────┐
    ▼     ▼                 ▼
  Planner  Retrieval     Tool Agent
  Agent    Agent         (6 tools)
    │        │               │
    └────────┴───────┬───────┘
                     ▼
              Execution Agent
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
  ChromaDB       PostgreSQL        Redis
  (vectors)      (data)           (memory)
```

### Multi-Agent Workflow

```
START → Planner → Retrieval → (conditional) Tool Agent → Executor → END
```

- **Planner Agent** — generates a step-by-step plan from the user query
- **Retrieval Agent** — performs RAG semantic search over the codebase
- **Tool Agent** — uses GPT-4o tool-calling to inspect/modify code
- **Execution Agent** — synthesizes everything into a final answer

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python 3.11+) |
| Agent Framework | LangGraph + LangChain |
| LLM (Production) | Groq API — Llama 3.3 70B |
| LLM (Local Dev) | Ollama — Llama 3.1 8B |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector Database | ChromaDB (persistent) |
| Database | PostgreSQL 16 |
| Cache / Memory | Redis 7 |
| Frontend | Next.js 15 + React 19 |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git
- Ollama (for local dev) or Groq API key (for production)

### 2. Clone & Configure

```bash
cd copilotcodebase/backend
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Start Services

```bash
# From project root
docker-compose up -d   # PostgreSQL + Redis
```

### 4. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 5. Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

### 6. Use It

1. Open `http://localhost:3000`
2. Click **"+ Add Repository"** and paste a Git URL
3. Wait for ingestion to complete
4. Start asking questions about the codebase!

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/repo/upload` | Clone and ingest a Git repository |
| `GET` | `/api/repo/{id}` | Get repository details |
| `GET` | `/api/repo/{id}/status` | Check ingestion status |
| `GET` | `/api/repo/{id}/files` | List repository files |
| `DELETE` | `/api/repo/{id}` | Delete a repository |
| `POST` | `/api/chat` | Ask a question (sync) |
| `POST` | `/api/chat/stream` | Ask a question (SSE streaming) |
| `GET` | `/api/chat/{conv_id}/history` | Get conversation history |
| `POST` | `/api/agent/run` | Run multi-agent workflow (sync) |
| `POST` | `/api/agent/run/stream` | Run multi-agent workflow (SSE) |
| `GET` | `/health` | Health check |

---

## Tools Available to Agents

| Tool | Description |
|------|-------------|
| `search_code` | Semantic search over the codebase |
| `get_file` | Read file contents with line ranges |
| `list_files` | Directory tree listing |
| `find_function_definition` | Find function/class definitions |
| `run_tests` | Execute test suites |
| `modify_file` | Apply code modifications |

---

## Project Structure

```
copilotcodebase/
├── docker-compose.yml
├── backend/
│   ├── requirements.txt
│   ├── .env.example
│   └── app/
│       ├── main.py              # FastAPI entry point
│       ├── config.py            # Pydantic settings
│       ├── database.py          # Async SQLAlchemy
│       ├── models.py            # ORM models
│       ├── schemas.py           # Request/response schemas
│       ├── ingestion/           # Codebase ingestion pipeline
│       │   ├── cloner.py        #   Git clone
│       │   ├── parser.py        #   File parsing
│       │   ├── chunker.py       #   Smart code chunking
│       │   ├── embedder.py      #   OpenAI embeddings
│       │   ├── vectorstore.py   #   ChromaDB storage
│       │   └── pipeline.py      #   Orchestrator
│       ├── rag/                 # Retrieval Augmented Generation
│       │   ├── retriever.py     #   Vector search
│       │   ├── prompt_builder.py#   Context + prompt
│       │   └── llm.py           #   GPT-4o wrapper
│       ├── tools/               # Agent tools
│       │   ├── code_search.py
│       │   ├── file_reader.py
│       │   ├── file_tree.py
│       │   ├── code_analysis.py
│       │   ├── test_runner.py
│       │   └── code_modifier.py
│       ├── agents/              # LangGraph agents
│       │   ├── state.py         #   Shared state
│       │   ├── planner.py       #   Plan generation
│       │   ├── retrieval.py     #   RAG retrieval
│       │   ├── tool_agent.py    #   Tool invocation
│       │   ├── executor.py      #   Final synthesis
│       │   └── graph.py         #   State machine
│       ├── memory/              # Memory system
│       │   ├── short_term.py    #   Conversation history
│       │   ├── long_term.py     #   Redis memory
│       │   └── manager.py       #   Unified interface
│       └── routes/              # API routes
│           ├── repo.py
│           ├── chat.py
│           └── agent.py
└── frontend/
    ├── package.json
    ├── next.config.js
    └── src/
        ├── lib/api.ts           # API client
        └── app/
            ├── layout.tsx
            ├── globals.css
            └── page.tsx         # Main application
```
