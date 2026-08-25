# AI Financial Analysis Agent

AI RAG Workbench is a full-stack Retrieval-Augmented Generation application for uploading documents, indexing them into Chroma Cloud, and asking grounded questions over the uploaded knowledge base.

The project is built as a learning-focused but production-oriented RAG system with hybrid retrieval, query rewriting, RRF fusion, semantic reranking, and source-grounded answer generation.

## Features

- Upload PDF, TXT, and Markdown files
- Parse uploaded files directly from RAM without persisting originals locally
- Chunk documents for retrieval
- Generate embeddings with OpenAI `text-embedding-3-small`
- Store chunk text, metadata, and vectors in Chroma Cloud
- Use cosine similarity for vector search
- Rewrite follow-up questions into standalone retrieval queries
- Retrieve with MMR vector search and BM25 keyword search
- Fuse retrieval results with Reciprocal Rank Fusion
- Rerank candidate chunks with a semantic reranker
- Generate answers with source citations
- Delete indexed documents by exact filename
- React frontend for upload, chat, rewritten query, and source inspection

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| RAG Framework | LangChain |
| Vector Database | Chroma Cloud |
| Embeddings | OpenAI `text-embedding-3-small` |
| Keyword Search | BM25 |
| Fusion | Reciprocal Rank Fusion |
| Reranking | Semantic reranker |
| PDF Parsing | `pypdf` with in-memory `BytesIO` |

## Architecture

Upload and indexing flow:

```text
UploadFile
  ↓
Read bytes into RAM
  ↓
Parse PDF/TXT/MD
  ↓
Chunk documents
  ↓
OpenAI embeddings
  ↓
Chroma Cloud
```

Question answering flow:

```text
User question
  ↓
Intent detection + query rewriting
  ↓
MMR vector search + BM25 keyword search
  ↓
RRF fusion
  ↓
Semantic reranking
  ↓
Final context chunks
  ↓
LLM answer with citations
```

## Project Structure

```text
AI_RAG/
  backend/
    app/
      main.py
      config.py
      schemas.py
      rag/
        chains.py
        ingest.py
        keyword_store.py
        query_processing.py
        rerank.py
        retrieval.py
        vector_store.py
    chroma_cloud.py
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
CHROMA_DATABASE=My_Chroma_Db
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

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Health check |
| POST | `/api/documents/upload` | Upload and index documents |
| GET | `/api/documents` | List indexed documents |
| DELETE | `/api/documents/{filename}` | Delete indexed chunks by exact filename |
| POST | `/api/chat` | Ask a question over indexed documents |

## Notes

- Uploaded originals are processed from RAM and are not persisted locally.
- Chunks, embeddings, and metadata are persisted in Chroma Cloud.
- Vector search uses cosine similarity.
- BM25 uses stored chunk text, not vectors.
- RRF is used because BM25 scores and vector similarity scores are not directly comparable.
- Deleting a document is based on the exact filename stored in Chroma metadata.
- Session-based document isolation is planned but not yet implemented.

## Detailed Documentation

For detailed backend learning notes, architecture decisions, and implementation history, see:

[docs/backend-learning-log.md](docs/backend-learning-log.md)
