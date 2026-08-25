# AI Financial Analysis Agent

AI Financial Analysis Agent is a full-stack agentic RAG application for uploading financial documents, indexing them into Chroma Cloud, and asking analysis questions that may require document retrieval, web search, and reliable calculations.

This is Project 2. It extends the previous fixed Advanced RAG project by exposing retrieval and supporting capabilities as tools. The LLM can now decide which tools to call, inspect tool results, call more tools when needed, and then produce a final management-level answer.

## What Changed From Project 1

Project 1 used a fixed application-controlled pipeline:

```text
User question
  ↓
Intent detection + query rewriting
  ↓
Vector search + BM25
  ↓
RRF fusion
  ↓
Top chunks
  ↓
LLM final answer
```

Project 2 uses an agentic tool-calling workflow:

```text
User question
  ↓
LLM agent
  ↓
Decide which tool is needed
  ├── document_search(query)
  ├── web_search(query)
  └── calculator(expression)
  ↓
Tool result returned to LLM
  ↓
More tool calls or final answer
```

The old Project 1 files are still present for reference, but the active chat route is:

```text
POST /api/agent/chat
```

The legacy fixed RAG route is currently commented out:

```text
POST /api/chat
```

## Agent Tools

### `document_search(query)`

Uses the existing RAG retrieval system:

```text
query
  ↓
MMR vector search
  +
BM25 keyword search
  ↓
RRF fusion
  ↓
relevant chunks from uploaded documents
```

This lets the LLM decide when uploaded document evidence is needed instead of always forcing retrieval.

### `web_search(query)`

Uses the OpenAI Responses API built-in web search tool through the existing `OPENAI_API_KEY`.

No separate Tavily, SerpAPI, or `requests` dependency is required for the current implementation.

The agent also has a guardrail for current or external questions. If the question looks time-sensitive or clearly outside uploaded documents, the backend prefetches web evidence before the model writes the final answer. This avoids stale-memory answers with `0 steps`.

### `calculator(expression)`

Uses a safe Python `ast`-based arithmetic parser for financial math.

Supported examples:

```text
(200.606 - 191.063) / 191.063 * 100
round((127.4 - 121.2) / 121.2 * 100, 2)
sqrt(81)
```

Unsupported code execution, imports, attribute access, arbitrary names, and unsafe function calls are rejected.

## Agent Behavior

The agent is designed to:

- Prefer uploaded financial documents when they contain the required facts
- Use web search for missing, external, current, or time-sensitive information
- Use calculator for percentage changes, margins, ratios, variances, and growth rates
- Avoid inventing missing financial numbers
- Avoid unsupported assumptions, such as evenly distributing six-month revenue into quarters
- Return a concise management-level answer
- Return an optional trace showing each tool call

## Conversation Handling

Short histories are passed directly to the LLM.

Long histories are compacted:

```text
history length < 12
  ↓
pass all history directly

history length >= 12
  ↓
summarize older history
  ↓
keep last 8 exact messages
  ↓
send summary + recent messages + current question
```

The constants are currently hardcoded in `backend/app/agent/agent.py`:

```python
RECENT_HISTORY_LIMIT = 8
SUMMARY_TRIGGER_LIMIT = 12
```

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| Agent LLM | OpenAI via LangChain `ChatOpenAI` tool calling |
| Web Search | OpenAI Responses API built-in `web_search` |
| Vector Database | Chroma Cloud |
| Embeddings | OpenAI `text-embedding-3-small` |
| Document Retrieval | MMR vector search + BM25 + RRF |
| Calculation | Safe Python `ast` arithmetic parser |
| PDF Parsing | `pypdf` with in-memory `BytesIO` |

## Project Structure

```text
AI_Agent/
  backend/
    app/
      main.py
      config.py
      schemas.py
      agent/
        agent.py
        tools.py
      rag/
        chains.py              # legacy Project 1 fixed RAG chain, currently commented
        ingest.py
        keyword_store.py
        query_processing.py    # legacy Project 1 query rewrite / intent logic
        rerank.py              # optional/future reranking
        retrieval.py
        vector_store.py
    api/
      index.py
    requirements.txt
  frontend/
    src/
      App.tsx
      api.ts
      types.ts
      styles.css
  docs/
    backend-learning-log.md
```

## Environment Variables

Create `backend/.env` and configure:

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

CHUNK_SIZE=900
CHUNK_OVERLAP=150
RETRIEVAL_K=5
RETRIEVAL_FETCH_K=20

CHROMA_TENANT=your_chroma_tenant
CHROMA_DATABASE=your_chroma_database
CHROMA_API_KEY=your_chroma_api_key
```

Do not commit `backend/.env` to GitHub.

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend API docs:

```text
http://127.0.0.1:8000/docs
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:5173
```

If npm has cache permission issues, use:

```bash
npm install --cache ./.npm-cache
```

For local development, the frontend defaults to:

```text
http://127.0.0.1:8000
```

For production builds, the frontend defaults to the deployed backend:

```text
https://aiagentbe.vercel.app
```

You can override either behavior by setting:

```bash
VITE_API_BASE_URL=https://your-backend-url
```

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Backend service status and docs pointer |
| GET | `/api/health` | Health check |
| POST | `/api/documents/upload` | Upload and index PDF, TXT, or Markdown documents |
| GET | `/api/documents` | List indexed documents |
| DELETE | `/api/documents/{filename}` | Delete indexed chunks by exact filename |
| POST | `/api/agent/chat` | Ask the agent a question |

Example agent request:

```bash
curl -X POST http://127.0.0.1:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Use the uploaded document to find Amazon Q2 net sales. If Q1 is missing, search the web and calculate QoQ growth.",
    "history": [],
    "max_steps": 8,
    "use_web_search": true,
    "return_trace": true
  }'
```

Example response shape:

```json
{
  "answer": "Management-level answer...",
  "trace": [
    {
      "step": 1,
      "tool": "document_search_tool",
      "input": {"query": "Amazon Q2 net sales"},
      "output": []
    }
  ]
}
```

## Notes

- Uploaded originals are processed from RAM and are not persisted locally.
- Chunks, embeddings, and metadata are persisted in Chroma Cloud.
- BM25 uses stored chunk text from Chroma Cloud.
- RRF combines vector and BM25 rankings without comparing raw score scales.
- The active retrieval path currently returns hybrid RRF results directly; semantic reranking is retained as an optional future enhancement.
- The frontend now calls `/api/agent/chat` and renders the agent trace.
- Session-based document isolation is planned but not implemented yet.

## Detailed Documentation

For detailed backend learning notes, architecture decisions, and implementation history, see:

[docs/backend-learning-log.md](docs/backend-learning-log.md)
