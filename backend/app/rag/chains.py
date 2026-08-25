# from fastapi import HTTPException, status
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI

# from app.config import Settings
# from app.rag.query_processing import process_query
# from app.rag.retrieval import retrieve_final_chunks
# from app.schemas import ChatResponse, ChatTurn, SourceChunk


# def answer_question(
#     question: str,
#     history: list[ChatTurn],
#     settings: Settings,
# ) -> ChatResponse:
#     """
#     Full RAG answer pipeline.

#     Flow:
#     1. Detect intent.
#     2. Rewrite question into standalone retrieval query.
#     3. Retrieve final chunks using hybrid retrieval + semantic reranking.
#     4. Format final chunks as LLM context.
#     5. Ask LLM to answer using only retrieved context.
#     6. Return answer, intent, rewritten question, and sources.
#     """

#     llm = _get_llm(settings)

#     intent, rewritten_question = process_query(
#         question=question,
#         history=history,
#         settings=settings,
#     )

#     if intent.needs_retrieval:
#         sources = retrieve_final_chunks(
#             query=rewritten_question,
#             settings=settings,
#         )
#     else:
#         sources = []

#     context = _format_context(sources)

#     response = llm.invoke(
#         [
#             SystemMessage(
#                 content=(
#                     "You are a careful RAG assistant. "
#                     "Answer using only the provided context. "
#                     "If the context does not contain the answer, say: "
#                     "'I do not know from the uploaded documents.' "
#                     "Cite relevant sources inline using bracket numbers like [1] or [2]. "
#                     "Do not invent citations."
#                 )
#             ),
#             HumanMessage(
#                 content=(
#                     f"Original user question:\n{question}\n\n"
#                     f"Rewritten retrieval question:\n{rewritten_question}\n\n"
#                     f"Retrieved context:\n{context}\n\n"
#                     "Write a concise, useful answer."
#                 )
#             ),
#         ]
#     )

#     return ChatResponse(
#         answer=str(response.content),
#         intent=intent,
#         rewritten_question=rewritten_question,
#         sources=sources,
#     )


# def _format_context(sources: list[SourceChunk]) -> str:
#     """
#     Converts retrieved chunks into numbered context blocks for the LLM.
#     """

#     if not sources:
#         return "No relevant context was retrieved."

#     formatted_chunks = []

#     for index, source in enumerate(sources, start=1):
#         page_label = f", page {source.page}" if source.page else ""

#         formatted_chunks.append(
#             f"[{index}] {source.filename}{page_label}\n"
#             f"{source.content}"
#         )

#     return "\n\n".join(formatted_chunks)


# def _get_llm(settings: Settings) -> ChatOpenAI:
#     if (
#         not settings.openai_api_key
#         or settings.openai_api_key == "your_openai_api_key_here"
#     ):
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="OPENAI_API_KEY is missing. Set it in backend/.env.",
#         )

#     return ChatOpenAI(
#         model=settings.openai_chat_model,
#         api_key=settings.openai_api_key,
#         temperature=0.1,
#     )
