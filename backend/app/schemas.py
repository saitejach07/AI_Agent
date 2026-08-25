from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


# class ChatRequest(BaseModel):
#     question: str = Field(min_length=1)
#     history: list[ChatTurn] = Field(default_factory=list)


class IntentResult(BaseModel):
    intent: str
    needs_retrieval: bool


class SourceChunk(BaseModel):
    source_id: str
    filename: str
    page: int | None = None
    chunk_index: int | None = None
    content: str
    score: float | None = None
    retrieval_method: str | None = None


# class ChatResponse(BaseModel):
#     answer: str
#     intent: IntentResult
#     rewritten_question: str
#     sources: list[SourceChunk]


class UploadResult(BaseModel):
    source_id: str
    filename: str
    chunks: int


class UploadResponse(BaseModel):
    documents: list[UploadResult]


class DocumentInfo(BaseModel):
    source_id: str
    filename: str
    chunks: int

class AgentChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)
    max_steps: int = 8
    use_web_search: bool = True
    return_trace: bool = True


class AgentChatResponse(BaseModel):
    answer: str
    trace: list[dict] | None = None
