# Backend Learning Log

This document tracks what we have built so far in the AI Agent project. Project 2 starts from the previous advanced RAG backend and adds an agentic tool-calling layer.

## Project Goal

Build an AI Financial Analysis Agent with:

- React frontend
- Python FastAPI backend
- ChromaDB vector database
- LangChain framework
- Existing advanced RAG retrieval: vector search, BM25 keyword search, and RRF fusion
- OpenAI LLM tool/function calling
- Agent tools for document search, web search, and calculation
- Tool execution loop with configurable max steps and optional trace output

Project 1 target architecture:

```text
User Question
  ↓
Query Processing / Intent Detection
  ↓
Query Rewriting
  ↓
Parallel Retrieval
  ├── Vector Search using embeddings + ChromaDB
  └── Keyword Search using BM25
  ↓
Combine Results
  ↓
Reranking
  ↓
Final Top Chunks
  ↓
LLM
  ↓
Answer with sources
```

Project 2 active architecture:

```text
User Question
  ↓
POST /api/agent/chat
  ↓
run_agent()
  ↓
OpenAI LLM chooses tools
  ├── document_search(query)
  │     ↓
  │   retrieve_final_chunks()
  │     ↓
  │   Vector Search + BM25 + RRF
  ├── web_search(query)
  │     ↓
  │   OpenAI Responses API with built-in web_search
  └── calculator(expression)
        ↓
      Python-side calculation
  ↓
Tool result returned to LLM
  ↓
More tool calls or final answer
```

## Current Project Folder

```text
/Users/saitejachatarajupalli/Downloads/Coding/AI_Agent
```

Current backend folder:

```text
/Users/saitejachatarajupalli/Downloads/Coding/AI_Agent/backend
```

## Files Created So Far

```text
backend/
  app/
    main.py
    config.py
    schemas.py
    agent/
      agent.py
      tools.py
    rag/
      chains.py
      ingest.py
      keyword_store.py
      query_processing.py
      rerank.py
      retrieval.py
      vector_store.py
  .env
```

If `backend/app/rag/__init__.py` does not exist yet, create it as an empty file so Python treats `rag` as a package.

## Step 1: Basic FastAPI App

File:

```text
backend/app/main.py
```

Purpose:

- Create the FastAPI app object.
- Add a simple health check endpoint.
- Confirm the backend server can run.

Core code:

```python
from fastapi import FastAPI

app = FastAPI(title="AI Financial Analysis Agent")


@app.get("/api/health")
def health():
    return {"status": "ok"}
```

Command to run backend:

```bash
uvicorn app.main:app --reload
```

Health check URL:

```text
http://127.0.0.1:8000/api/health
```

Expected response:

```json
{"status":"ok"}
```

## Step 2: Configuration With `.env`

File:

```text
backend/app/config.py
```

Purpose:

- Load configuration from `.env`.
- Avoid hardcoding API keys and model names in code.
- Load Chroma Cloud credentials.
- Keep runtime settings such as chunk size and retrieval count in one place.

Important concept:

```python
BACKEND_DIR = Path(__file__).resolve().parents[1]
```

`__file__` means the current Python file path. We use it to find the backend directory reliably.

Settings included:

- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVAL_K`
- `RETRIEVAL_FETCH_K`
- `collection_name`
- `CHROMA_TENANT`
- `CHROMA_DATABASE`
- `CHROMA_API_KEY`

Removed after the latest architecture change:

- `upload_dir`: no longer needed because uploads are parsed from RAM.
- `chroma_dir`: no longer needed because vectors are stored in Chroma Cloud.

Environment file:

```text
backend/.env
```

Current `.env` values:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=900
CHUNK_OVERLAP=150
RETRIEVAL_K=5
RETRIEVAL_FETCH_K=20
CHROMA_TENANT=your_chroma_tenant
CHROMA_DATABASE=My_Chroma_Db
CHROMA_API_KEY=your_chroma_cloud_api_key
```

Do not expose the real API keys.

Config test command:

```bash
python -c "from app.config import get_settings; s=get_settings(); print(s.openai_chat_model); print(s.chroma_database)"
```

Expected output includes:

```text
gpt-4.1-mini
My_Chroma_Db
```

`get_settings()` should no longer create `settings.upload_dir` or `settings.chroma_dir`. If it still does, the backend will fail with an `AttributeError`.

## Step 3: API Schemas

File:

```text
backend/app/schemas.py
```

Purpose:

Schemas define the data contract between frontend and backend.

They answer:

- What should the frontend send?
- What should the backend return?
- Which fields are required?
- What type should each field be?

Schemas created:

```text
ChatTurn        one user/assistant chat message
ChatRequest     legacy Project 1 question + chat history, currently commented out
IntentResult    output from intent detection
SourceChunk     retrieved document chunk shown as source
ChatResponse    legacy Project 1 final answer + rewritten query + sources, currently commented out
UploadResult    upload/indexing result for one document
UploadResponse  upload/indexing result for multiple documents
DocumentInfo    indexed document info
AgentChatRequest   agent question + history + agent controls
AgentChatResponse  agent answer + optional trace
```

Current agent schemas:

```python
class AgentChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)
    max_steps: int = 8
    use_web_search: bool = True
    return_trace: bool = True


class AgentChatResponse(BaseModel):
    answer: str
    trace: list[dict] | None = None
```

Why the agent has separate schemas:

```text
ChatRequest / ChatResponse belong to the old fixed RAG route and are currently commented out.
AgentChatRequest / AgentChatResponse belong to the new agent route.
```

The agent request includes controls:

```text
max_steps        max tool-call rounds before stopping
use_web_search   enables or disables the web_search tool for this request
return_trace     returns the tool-call history for debugging and learning
```

Schema test command:

```bash
python -c "from app.schemas import AgentChatRequest; print(AgentChatRequest(question='What is RAG?'))"
```

Expected output:

```text
question='What is RAG?' history=[] max_steps=8 use_web_search=True return_trace=True
```

## Step 4: Query Processing From Project 1

File:

```text
backend/app/rag/query_processing.py
```

Purpose:

This file handles the first part of the advanced RAG pipeline:

```text
User Question
  ↓
Intent Detection
  ↓
Conversation Summary
  ↓
Relevant Earlier Conversation
  ↓
Recent Messages
  ↓
Query Rewriting
```

Project 2 note:

```text
The active /api/agent/chat route does not call process_query().
It does not explicitly run detect_intent() or rewrite_query().
The LLM agent decides whether a tool is needed and what query to pass to that tool.
```

The file is kept because:

```text
1. It documents the old fixed RAG architecture.
2. It can be reused later if document_search() needs explicit query rewriting.
3. It allows comparison between fixed RAG and agentic RAG.
```

Functions created:

```text
process_query()              main entry point for query processing
detect_intent()              classifies the question and decides if retrieval is needed
summarize_conversation()     creates long-term conversation context
retrieve_relevant_history()  finds relevant older chat turns with keyword overlap
rewrite_query()              rewrites latest question into a standalone retrieval query
_normalize_terms()           helper for keyword matching
_get_llm()                   creates LangChain ChatOpenAI instance
```

Current query rewriting uses three context sources:

```text
1. Conversation summary
2. Recent messages
3. Relevant older messages
```

This is better than only using `history[-8:]`, because it can understand long conversations and references to older topics.

Current relevant-history retrieval is simple keyword overlap. Later we will upgrade it to semantic conversation memory using embeddings and ChromaDB.

Import test command:

```bash
python -c "from app.rag.query_processing import process_query; print('query processing ok')"
```

Expected output:

```text
query processing ok
```

Do not call `process_query()` yet unless a real OpenAI API key is configured.

## Step 5: Vector Store And Chroma Cloud

File:

```text
backend/app/rag/vector_store.py
```

Purpose:

- Create the OpenAI embedding model.
- Connect LangChain to Chroma Cloud.
- Use the existing `rag_documents` collection in `My_Chroma_Db`.

Important function:

```python
get_vector_store(settings)
```

Current implementation uses:

```python
client = chromadb.CloudClient(
    tenant=settings.chroma_tenant,
    database=settings.chroma_database,
    api_key=settings.chroma_api_key,
)

return Chroma(
    client=client,
    collection_name=settings.collection_name,
    embedding_function=get_embeddings(settings),
)
```

This creates the flow:

```text
Our Python code
  ↓
LangChain Chroma wrapper
  ↓
Chroma Cloud
  ↓
rag_documents collection
```

Important point:

```text
LangChain is not ChromaDB.
LangChain helps us talk to Chroma Cloud.
Chroma Cloud is the actual vector database.
```

The old local Chroma settings are no longer used:

```text
persist_directory=settings.chroma_dir
collection_metadata={"hnsw:space": "cosine"}
```

Cosine similarity is configured directly on the Chroma Cloud collection. The Cloud configuration has been verified with `space = cosine`.

The app still supplies its own embedding function:

```python
embedding_function=get_embeddings(settings)
```

So even if Chroma Cloud displays `DefaultEmbeddingFunction` during inspection, this application uses:

```text
OpenAI text-embedding-3-small
1536-dimensional vectors
```

## Step 6: Document Ingestion From RAM

File:

```text
backend/app/rag/ingest.py
```

Purpose:

This file handles the upload-to-Chroma-Cloud pipeline without persisting uploaded originals locally.

Old flow:

```text
UploadFile
  ↓
write to data/uploads/
  ↓
PyPDFLoader(path)
  ↓
chunk → embed → Chroma Cloud
```

Current flow:

```text
UploadFile
  ↓
await file.read()
  ↓
file_bytes in RAM
  ↓
_load_documents(file_bytes)
  ↓
PDF → pypdf PdfReader(BytesIO)
TXT/MD → decode UTF-8
  ↓
LangChain Documents
  ↓
_split_documents()
  ↓
Chunks
  ↓
Add metadata
  ↓
get_vector_store()
  ↓
vector_store.add_documents(chunks)
  ↓
OpenAI text-embedding-3-small
  ↓
1536-dimensional vectors
  ↓
Chroma Cloud / rag_documents
```

Function duties:

```text
ingest_upload()
  Main function. Reads the uploaded file into RAM, loads text, chunks it, and stores chunks in Chroma Cloud.

list_indexed_documents()
  Reads Chroma Cloud metadata and returns the list of indexed documents.

_load_documents()
  Loads text from PDF, TXT, or MD file bytes. PDF uses PdfReader(BytesIO(file_bytes)); TXT/MD uses UTF-8 decode. Also adds metadata like source_id, filename, and page.

_split_documents()
  Splits loaded document text into smaller chunks using chunk_size and chunk_overlap from Settings.
```

Important line:

```python
vector_store.add_documents(chunks)
```

At this point, LangChain uses the embedding model from `vector_store.py` to create vectors for every chunk and store them in Chroma Cloud.

RAM is temporary processing memory only. It is not session storage. After text extraction and indexing, the uploaded file bytes can be discarded. Persistent RAG data lives in Chroma Cloud as chunk text, embeddings, and metadata.

## LangChain Document To ChromaDB Storage

LangChain owns the `Document` format. We did not design this shape:

```python
Document(
    page_content="RAG retrieves relevant chunks...",
    metadata={
        "source_id": "abc123",
        "filename": "rag_notes.pdf",
        "page": 0,
        "chunk_index": 3,
    },
)
```

We control the metadata we attach:

```python
for doc in docs:
    doc.metadata.update(
        {
            "source_id": source_id,
            "filename": filename,
            "page": doc.metadata.get("page"),
        }
    )
```

After chunking, we add:

```python
for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_index"] = index
```

When we call:

```python
vector_store.add_documents(chunks)
```

LangChain + Chroma Cloud internally stores:

```text
documents   -> chunk.page_content
metadatas   -> chunk.metadata
embeddings  -> vector generated from chunk.page_content
```

Conceptually:

```json
{
  "documents": [
    "RAG retrieves relevant chunks before answering..."
  ],
  "embeddings": [
    [0.12, -0.03, 0.44]
  ],
    "metadatas": [
    {
      "source_id": "abc123",
      "filename": "rag_notes.pdf",
      "page": 0,
      "chunk_index": 3
    }
  ]
}
```

## Step 7: SourceChunk Adapter

File:

```text
backend/app/rag/retrieval.py
```

Function:

```python
_source_from_document()
```

Purpose:

Convert LangChain's `Document` format into our API response format.

Mapping:

```text
Document.page_content         -> SourceChunk.content
Document.metadata.source_id   -> SourceChunk.source_id
Document.metadata.filename    -> SourceChunk.filename
Document.metadata.page        -> SourceChunk.page
Document.metadata.chunk_index -> SourceChunk.chunk_index
```

Why we need this:

```text
LangChain Document = internal RAG/library format
SourceChunk        = our frontend/API response format
```

The frontend should not depend on LangChain's internal object shape.

## Step 8: BM25 Keyword Search

File:

```text
backend/app/rag/keyword_store.py
```

Important concept:

BM25 does not use vectors.

It compares the tokenized query against the text chunks stored in ChromaDB.

When we store chunks in ChromaDB, Chroma stores more than vectors:

```text
chunk text
embedding vector
metadata
```

Example stored chunk:

```text
content: "RAG retrieves relevant chunks before answering..."
vector: [0.12, -0.03, ...]
metadata: { filename, page, chunk_index }
```

Vector search uses:

```text
query embedding vs chunk embeddings
```

BM25 uses:

```text
query words vs chunk text words
```

In `bm25_search()`, this line loads stored chunk text and metadata from ChromaDB:

```python
raw = vector_store.get(include=["documents", "metadatas"])
```

That returns something like:

```python
{
    "documents": [
        "RAG retrieves relevant chunks before answering...",
        "Embeddings convert text into vectors...",
        "BM25 is a keyword ranking algorithm..."
    ],
    "metadatas": [
        {"filename": "rag_notes.pdf", "page": 0, "chunk_index": 0},
        {"filename": "rag_notes.pdf", "page": 1, "chunk_index": 1},
        {"filename": "rag_notes.pdf", "page": 2, "chunk_index": 2}
    ]
}
```

BM25 flow:

```text
ChromaDB stored chunk text
  ↓
_load_all_chunks()
  ↓
Tokenize all chunks
  ↓
Build BM25 index in memory
  ↓
Tokenize query
  ↓
Compare query tokens against chunk tokens
  ↓
Return top keyword matches
```

## Step 9: Hybrid Retrieval With RRF

File:

```text
backend/app/rag/retrieval.py
```

Current retrieval flow:

```text
retrieve_final_chunks()
  ↓
hybrid_search()
  ↓
vector_search_mmr()
  +
bm25_search()
  ↓
rrf_fusion()
  ↓
final chunks
```

The `rerank.py` file still exists, but the current `retrieve_final_chunks()` implementation returns `hybrid_search()` directly. Semantic reranking is currently optional/commented out, not active in the main retrieval path.

RRF means Reciprocal Rank Fusion.

We use RRF because vector search and BM25 produce scores that are not directly comparable:

```text
Vector score/distance: 0.18
BM25 score:            12.7
```

RRF does not compare raw scores. It uses rank positions.

Formula:

```text
RRF score = 1 / (RRF_K + rank)
```

With `RRF_K = 60`:

```text
Rank 1 = 1 / 61
Rank 2 = 1 / 62
Rank 3 = 1 / 63
```

If the same chunk appears in both vector search and BM25, its RRF scores are added. This means duplicate chunks are deduplicated but rewarded.

Deduplication happens through:

```python
key = _chunk_key(chunk)
chunks_by_key[key] = chunk
```

Because `chunks_by_key` is a dictionary, only one chunk object is kept per key:

```python
return f"{chunk.source_id}:{chunk.chunk_index}"
```

But RRF score still accumulates:

```python
fused_scores[key] = fused_scores.get(key, 0.0) + (
    1.0 / (RRF_K + rank)
)
```

So a chunk found by both retrievers is:

```text
deduplicated
but rewarded
```

## Step 10: Semantic Reranking

File:

```text
backend/app/rag/rerank.py
```

Purpose:

Semantic reranking takes the RRF-fused candidate chunks and decides which chunks best answer the exact query.

Input:

```text
rewritten standalone query
RRF-fused candidate chunks from hybrid_search()
settings
```

Current Project 2 status:

```text
rerank.py exists as a future enhancement.
retrieve_final_chunks() currently returns hybrid RRF results directly.
```

Flow:

```text
RRF candidate chunks
  ↓
semantic_rerank(query, candidates, settings)
  ↓
pair query + each chunk
  ↓
cross-encoder reranker
  ↓
semantic relevance score
  ↓
final top K chunks
```

Important distinction:

```text
RRF = combines retriever rankings
Semantic reranker = judges query + chunk relevance
```

RRF helps the semantic reranker by sending a cleaner, deduplicated, better-ranked candidate pool.

## Step 11: Final LLM Answer Chain From Project 1

File:

```text
backend/app/rag/chains.py
```

Purpose:

Pass the final retrieved chunks to the LLM and return one answer with sources.

Flow:

```text
answer_question()
  ↓
process_query()
  ↓
retrieve_final_chunks()
  ↓
_format_context()
  ↓
ChatOpenAI
  ↓
ChatResponse
```

Project 2 note:

```text
The active agent endpoint does not use answer_question().
The old /api/chat route is commented out in main.py.
```

The agent only reuses the retrieval part:

```text
agent.py
  ↓
tools.py document_search()
  ↓
retrieve_final_chunks()
```

That means Project 2 reuses:

```text
Vector search
BM25
RRF fusion
Chroma Cloud
OpenAI embeddings
```

But it does not use the old fixed chain:

```text
answer_question()
process_query()
detect_intent()
rewrite_query()
```

## Step 12: Agent Tools

Files:

```text
backend/app/agent/tools.py
backend/app/agent/agent.py
```

The agent has three tools.

### document_search(query)

Purpose:

```text
Expose the existing RAG retrieval system as an LLM-callable tool.
```

Implementation:

```text
document_search(query, settings)
  ↓
retrieve_final_chunks(query, settings)
  ↓
Vector search + BM25 + RRF
  ↓
list[dict] containing source_id, filename, page, chunk_index, content, score, retrieval_method
```

This is the main Project 1 to Project 2 transition:

```text
Project 1: application code always runs retrieval.
Project 2: LLM decides when to call document_search().
```

### web_search(query)

Purpose:

```text
Fetch current, external, or missing information that is not available in uploaded documents.
```

Current implementation uses the OpenAI Responses API built-in web search tool:

```python
client = OpenAI(api_key=settings.openai_api_key)

response = client.responses.create(
    model=settings.openai_chat_model,
    tools=[{"type": "web_search"}],
    input=(
        "Search the web for reliable information related to this query. "
        "Return concise findings and include source references when available.\n\n"
        f"Query: {query}"
    ),
)
```

This means:

```text
No Tavily key is required.
No SerpAPI key is required.
The same OPENAI_API_KEY is used.
The openai package in requirements.txt is enough for this implementation.
```

### calculator(expression)

Purpose:

```text
Run numerical calculations in Python instead of letting the LLM do arithmetic mentally.
```

Examples:

```text
QoQ percentage change
YoY growth
Operating margin
Revenue growth
Variance
Ratios
```

Current implementation uses a safe arithmetic parser based on Python's `ast` module.

Allowed expression elements:

```text
numeric literals
+ - * / ** %
unary + and -
parentheses
abs()
round()
min()
max()
sqrt()
```

Unsupported names, imports, attribute access, keyword arguments, and arbitrary function calls are rejected. This lets the LLM request financial math while Python performs the calculation in a bounded way.

## Step 13: Agent Tool-Calling Loop

File:

```text
backend/app/agent/agent.py
```

Main function:

```python
run_agent(
    question,
    history,
    settings,
    max_steps=8,
    use_web_search=True,
    return_trace=True,
)
```

Flow:

```text
Build LangChain tools
  ↓
Optionally include web_search_tool
  ↓
Bind tools to ChatOpenAI
  ↓
Prepare conversation context
  ├── if history has fewer than 12 turns: keep history as-is
  └── if history has 12 or more turns: summarize older history and keep last 8 exact turns
  ↓
Send system prompt + optional summary + recent history + question
  ↓
If the question is clearly current/external, prefetch web_search evidence
  ↓
LLM response
  ├── no tool calls → final answer
  └── tool calls → execute Python tool
          ↓
        append ToolMessage
          ↓
        continue loop
```

The loop is bounded:

```python
for step in range(max_steps):
```

This prevents the agent from calling tools forever.

### Agent Conversation Summary

The active agent no longer passes the whole conversation history to the LLM once the chat becomes long.

Current hardcoded limits in `backend/app/agent/agent.py`:

```python
RECENT_HISTORY_LIMIT = 8
SUMMARY_TRIGGER_LIMIT = 12
```

Behavior:

```text
history length < 12
  ↓
pass all available history directly

history length >= 12
  ↓
summarize older history
  ↓
keep last 8 messages exactly
  ↓
send summary + recent messages + current question to the agent
```

The summary is created by:

```python
summarize_agent_history(llm, history)
```

The summary prompt is agent-specific. It asks the LLM to preserve:

```text
companies
financial periods
metrics
document names
numbers already discussed
assumptions
unresolved questions
user preferences
```

This replaces the earlier behavior where `run_agent()` appended every history item:

```python
for turn in history:
    messages.append(...)
```

The current message order is:

```text
System prompt
Optional summary of older history
Last 8 exact history messages
Current user question
Optional web_search verification result for current/external questions
Tool call responses as ToolMessage objects
```

### Current / External Fact Guardrail

Issue found during testing:

```text
Question: "who won the fifa 2026 world cup"
Agent response: said the tournament had not happened
Trace: 0 steps
```

Root cause:

```text
The model answered directly from stale/internal knowledge and did not call web_search_tool.
```

Fix:

```text
If use_web_search=True and the question looks time-sensitive or external, run web_search before the main agent loop and inject the result into the messages as verified evidence.
```

Implemented in:

```python
if use_web_search and _requires_external_verification(question):
    result = web_search(query=question, settings=settings)
```

The helper checks terms like:

```text
latest
current
today
recent
winner
won
world cup
stock price
market cap
current or future-looking years
```

This means a question like:

```text
Who won the FIFA 2026 World Cup?
```

will call web search before the model writes the final answer. The trace should show at least one `web_search_tool` step instead of `0 steps`.

Trace format:

```python
{
    "step": 1,
    "tool": "document_search_tool",
    "input": {"query": "..."},
    "output": [...]
}
```

The trace helps show how the answer was produced:

```text
Which tools were called
What inputs the LLM selected
What each tool returned
Whether the agent needed multiple steps
```

## Step 14: Agent Chat API

File:

```text
backend/app/main.py
```

The old fixed RAG route is commented out:

```python
# @app.post("/api/chat", response_model=ChatResponse)
```

The active agent route is:

```python
@app.post("/api/agent/chat", response_model=AgentChatResponse)
def agent_chat(
    request: AgentChatRequest,
    settings: Settings = Depends(get_settings),
) -> AgentChatResponse:
    result = run_agent(
        question=request.question,
        history=request.history,
        settings=settings,
        max_steps=request.max_steps,
        use_web_search=request.use_web_search,
        return_trace=request.return_trace,
    )

    return AgentChatResponse(
        answer=result["answer"],
        trace=result.get("trace"),
    )
```

Request body:

```json
{
  "question": "Analyze Amazon revenue and calculate the percentage change.",
  "history": [],
  "max_steps": 8,
  "use_web_search": true,
  "return_trace": true
}
```

Response body:

```json
{
  "answer": "Management-level analysis...",
  "trace": [
    {
      "step": 1,
      "tool": "document_search_tool",
      "input": {"query": "Amazon revenue latest quarter"},
      "output": []
    }
  ]
}
```

The LLM does not receive Python `SourceChunk` objects directly. We convert them to plain text context:

```python
context = _format_context(sources)
```

The context looks like:

```text
[1] rag_notes.pdf, page 2
RAG retrieves relevant chunks before answering...

[2] rag_notes.pdf, page 4
Vector search uses embeddings...
```

The chunks are passed into the LLM here:

```python
f"Retrieved context:\n{context}\n\n"
```

The LLM reads the final chunks and generates one grounded answer. It does not choose only one chunk; it synthesizes a single answer from all useful chunks.

## Future: Frontend Session Scoping

The backend does not currently know whether a browser page was refreshed or whether a new frontend session started.

Right now, tools like Postman are simply making HTTP requests:

```text
Postman
  ↓
POST /api/documents/upload
```

The backend has no browser session concept unless the frontend explicitly sends one.

When we build the frontend, we can define the session lifecycle there:

```text
User opens frontend
  ↓
Frontend generates session_id
  ↓
Upload request sends session_id
Chat request sends session_id
```

One design choice remains:

```text
A. Refresh = new session
   Open -> Session A
   Refresh -> Session B

B. New Chat = new session
   Open -> Session A
   Refresh -> Still Session A
   New Chat -> Session B
```

For a real chat application, option B is usually better because a browser refresh should not normally destroy the user's conversation and uploaded-document context.

For now, we are not modifying the backend for sessions. We will finish and test the RAG pipeline first, then add session isolation when we build the frontend.

## Current Backend Status

Done:

```text
FastAPI app
.env configuration
Pydantic schemas
OpenAI embeddings setup
ChromaDB setup with cosine distance
document ingestion
document chunking
BM25 keyword search
MMR vector search
RRF fusion
agent tools.py
agent.py tool-calling loop
document_search tool using existing RAG retrieval
web_search tool using OpenAI Responses API built-in web search
current/external question guardrail that prefetches web_search evidence
calculator tool
/api/agent/chat endpoint
agent trace output
agent conversation summary for long chat history
frontend API updated to call /api/agent/chat
frontend response types updated for answer + trace
frontend trace rendering
safe AST-based calculator
```

Legacy Project 1 pieces retained but not used by the active `/api/agent/chat` path:

```text
query_processing.py
detect_intent()
rewrite_query()
summarize_conversation()
relevant history keyword retrieval
chains.py final fixed RAG answer chain
rerank.py optional semantic reranking
```

Remaining:

```text
Add real OPENAI_API_KEY in backend/.env
Test upload endpoint
Test /api/agent/chat endpoint end to end
Add reset/delete indexed documents later
Add endpoint to clear one session's uploads and Chroma chunks
```

Current frontend integration note:

```text
frontend/src/api.ts now posts questions to /api/agent/chat.
frontend/src/types.ts now expects AgentChatResponse shape: answer + trace.
frontend/src/App.tsx now renders the trace in the assistant message details.
```

So the frontend chat UI is now aligned with the active backend agent endpoint.

## Dependencies Installed So Far

### Initial FastAPI Backend Dependencies

Installed when setting up the backend and health endpoint:

```bash
pip install fastapi uvicorn python-dotenv pydantic-settings
```

What they are for:

```text
fastapi             Python API framework
uvicorn             Runs the FastAPI development server
python-dotenv       Loads .env files
pydantic-settings   Environment-based settings management
```

Recommended command with Uvicorn standard extras:

```bash
pip install fastapi "uvicorn[standard]" python-dotenv pydantic-settings
```

### LangChain Query Processing Dependencies

Installed when creating `query_processing.py`:

```bash
pip install langchain langchain-openai
```

What they are for:

```text
langchain         Core LangChain framework
langchain-openai  LangChain integration for OpenAI chat models and embeddings
```

## Requirements File

We should maintain this file:

```text
backend/requirements.txt
```

Current pinned contents:

```text
fastapi==0.141.1
uvicorn==0.52.0
python-multipart==0.0.32
pydantic==2.13.4
pydantic-settings==2.15.0
python-dotenv==1.2.2
langchain-core==1.5.4
langchain-community==0.4.2
langchain-text-splitters==1.1.2
langchain-openai==1.4.3
langchain-chroma==1.1.0
chromadb==1.5.9
openai==2.54.0
rank-bm25==0.2.2
pypdf==6.15.0
```

Why these packages are included:

```text
python-multipart         needed soon for file uploads
openai                   OpenAI SDK dependency, including Responses API web search
langchain-community      document loaders and BM25 utilities
langchain-chroma         LangChain integration with ChromaDB
langchain-text-splitters document chunking
chromadb                 Chroma Cloud client/vector database
pypdf                    PDF text extraction
rank-bm25                keyword/BM25 search
```

`requests` is not needed for the current web search implementation because `web_search()` uses the OpenAI SDK instead of Tavily or SerpAPI.

Install from requirements:

```bash
pip install -r requirements.txt
```

Dependency sanity check:

```bash
python -c "from fastapi import FastAPI; from langchain_openai import ChatOpenAI; print('deps ok')"
```

Expected output:

```text
deps ok
```

## Where LangChain Comes Into The Architecture

LangChain is not the whole backend. It is used inside the RAG modules.

```text
FastAPI              API layer
Our modules          Application architecture
LangChain            LLM/RAG helper framework
ChromaDB             Vector database
OpenAI               LLM + embeddings provider
```

LangChain usage by stage:

```text
Intent detection      ChatOpenAI
Query rewriting       ChatOpenAI
Document loading      LangChain document loaders
Chunking              LangChain text splitters
Embeddings            LangChain OpenAIEmbeddings
Vector search         LangChain Chroma integration against Chroma Cloud
BM25 keyword search   rank-bm25 over ChromaDB stored documents
RRF fusion            our own code in retrieval.py
Semantic reranking    sentence-transformers CrossEncoder
Final answer          ChatOpenAI
```

## Step 15: Document Delete API

We added document deletion to the RAG backend.

Important architecture:

```text
ChromaDB is hosted on Chroma Cloud.
Uploaded files are processed from RAM and are not persisted locally.
Chroma stores the chunks and metadata.
Every chunk from the same uploaded file has the same filename and source_id.
```

### Delete Behavior

Deletion is based on the exact `filename` stored in Chroma metadata.

Example metadata:

```json
{
  "chunk_index": 0,
  "filename": "Amazon_Pharmacy Finance_Saiteja_Senior_Financial_Analyst_Resume copy.pdf",
  "page": 0,
  "source_id": "e2ba7b6ef7054fa5a6d8146b00a0b551"
}
```

The filename must remain exactly as stored:

```text
Preserve spaces
Preserve underscores
Preserve capitalization
Preserve the file extension
Do not normalize or replace spaces
```

### Delete Endpoint

File:

```text
backend/app/main.py
```

Endpoint:

```python
@app.delete("/api/documents/{filename}")
def delete_document(
    filename: str,
    settings: Settings = Depends(get_settings),
):
    delete_indexed_document(
        filename=filename,
        settings=settings,
    )

    return {
        "message": "Document deleted successfully",
        "filename": filename,
    }
```

Conceptual flow:

```text
DELETE /api/documents/{filename}
  ↓
delete_indexed_document(filename, settings)
  ↓
find chunks in Chroma Cloud where metadata.filename == filename
  ↓
delete those chunk IDs from Chroma Cloud
  ↓
return deleted filename
```

Important frontend/API note:

If the filename contains spaces or special characters, the frontend/client must URL-encode it when calling the endpoint.

Example:

```text
Amazon_Pharmacy Finance_Saiteja.pdf
```

should be sent as a URL path-safe value, for example:

```text
Amazon_Pharmacy%20Finance_Saiteja.pdf
```
