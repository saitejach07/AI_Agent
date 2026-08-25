from fastapi import Depends, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agent.agent import run_agent
from app.config import Settings, get_settings
from app.rag.ingest import ingest_upload, list_indexed_documents
from app.rag.ingest import delete_indexed_document
from app.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    DocumentInfo,
    UploadResponse,
)


# Create the FastAPI application.
app = FastAPI(title="AI Financial Analysis Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173","https://aiagentfe.vercel.app",],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Basic health check to verify the backend is running.
@app.get("/api/health")
def health():
    return {"status": "ok"}


# Upload one or more documents and send them through the ingestion pipeline.
@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile],
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    results = []

    # Process each uploaded file independently.
    for file in files:
        result = await ingest_upload(file, settings)
        results.append(result)

    # Return the result for all uploaded documents.
    return UploadResponse(documents=results)


# Return the list of documents that have already been indexed.
@app.get("/api/documents", response_model=list[DocumentInfo])
def documents(
    settings: Settings = Depends(get_settings),
) -> list[DocumentInfo]:
    return list_indexed_documents(settings)


# @app.post("/api/chat", response_model=ChatResponse)
# def chat(
#     request: ChatRequest,
#     settings: Settings = Depends(get_settings),
# ) -> ChatResponse:
#     return answer_question(
#         question=request.question,
#         history=request.history,
#         settings=settings,
#     )

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
