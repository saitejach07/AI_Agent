# import json
# import re

# from fastapi import HTTPException, status
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI

# from app.config import Settings
# from app.schemas import ChatTurn, IntentResult


# RECENT_HISTORY_LIMIT = 8
# SUMMARY_TRIGGER_LIMIT = 12
# RELEVANT_HISTORY_LIMIT = 6


# def process_query(
#     question: str,
#     history: list[ChatTurn],
#     settings: Settings,
# ) -> tuple[IntentResult, str]:
#     """
#     Runs query processing before retrieval.

#     Flow:
#     1. Detect user intent.
#     2. Build conversation summary for long-term context.
#     3. Retrieve relevant older messages from conversation history.
#     4. Rewrite latest question into a standalone retrieval query.
#     """

#     llm = _get_llm(settings)

#     intent = detect_intent(llm, question)

#     conversation_summary = summarize_conversation(llm, history)
#     relevant_history = retrieve_relevant_history(question, history)

#     rewritten_question = rewrite_query(
#         llm=llm,
#         question=question,
#         history=history,
#         conversation_summary=conversation_summary,
#         relevant_history=relevant_history,
#     )

#     return intent, rewritten_question


# def detect_intent(llm: ChatOpenAI, question: str) -> IntentResult:
#     response = llm.invoke(
#         [
#             SystemMessage(
#                 content=(
#                     "You are an intent detection system for a RAG application. "
#                     "Return only valid JSON. Do not include markdown. "
#                     "The JSON must have these fields: intent, needs_retrieval."
#                 )
#             ),
#             HumanMessage(
#                 content=(
#                     f"User question: {question}\n\n"
#                     "Classify the intent. "
#                     "If the question should be answered from uploaded documents, "
#                     "set needs_retrieval to true."
#                 )
#             ),
#         ]
#     )

#     try:
#         data = json.loads(str(response.content))
#         return IntentResult(
#             intent=data.get("intent", "document_question"),
#             needs_retrieval=bool(data.get("needs_retrieval", True)),
#         )
#     except json.JSONDecodeError:
#         return IntentResult(intent="document_question", needs_retrieval=True)


# def summarize_conversation(
#     llm: ChatOpenAI,
#     history: list[ChatTurn],
# ) -> str | None:
#     """
#     Creates a compact summary when the conversation becomes long.

#     For short conversations, we do not summarize because recent messages are enough.
#     """

#     if len(history) < SUMMARY_TRIGGER_LIMIT:
#         return None

#     history_text = "\n".join(
#         f"{turn.role}: {turn.content}"
#         for turn in history
#     )

#     response = llm.invoke(
#         [
#             SystemMessage(
#                 content=(
#                     "Summarize this conversation for a RAG query rewriting system. "
#                     "Keep only stable context that helps understand future follow-up questions. "
#                     "Preserve project choices, technical decisions, user goals, file names, "
#                     "frameworks, architecture decisions, and unresolved tasks. "
#                     "Do not include unnecessary small talk."
#                 )
#             ),
#             HumanMessage(content=f"Conversation:\n{history_text}"),
#         ]
#     )

#     summary = str(response.content).strip()
#     return summary or None


# def retrieve_relevant_history(
#     question: str,
#     history: list[ChatTurn],
# ) -> list[ChatTurn]:
#     """
#     Retrieves relevant older messages from conversation history.

#     Current version uses simple keyword overlap.
#     Later, we can replace this with conversation embeddings stored in ChromaDB.
#     """

#     if not history:
#         return []

#     recent_history = history[-RECENT_HISTORY_LIMIT:]
#     older_history = history[:-RECENT_HISTORY_LIMIT]

#     if not older_history:
#         return []

#     question_terms = _normalize_terms(question)
#     scored_turns: list[tuple[int, ChatTurn]] = []

#     for turn in older_history:
#         turn_terms = _normalize_terms(turn.content)
#         score = len(question_terms.intersection(turn_terms))

#         if score > 0:
#             scored_turns.append((score, turn))

#     scored_turns.sort(key=lambda item: item[0], reverse=True)

#     return [
#         turn
#         for _, turn in scored_turns[:RELEVANT_HISTORY_LIMIT]
#     ]


# def rewrite_query(
#     llm: ChatOpenAI,
#     question: str,
#     history: list[ChatTurn],
#     conversation_summary: str | None = None,
#     relevant_history: list[ChatTurn] | None = None,
# ) -> str:
#     """
#     Rewrites the latest user question into a standalone search query.

#     Uses:
#     1. Conversation summary for long-term context.
#     2. Recent messages for immediate context.
#     3. Relevant historical messages retrieved from conversation memory.
#     """

#     conversation_summary_text = conversation_summary or "None"

#     recent_history = history[-RECENT_HISTORY_LIMIT:]

#     recent_history_text = "\n".join(
#         f"{turn.role}: {turn.content}"
#         for turn in recent_history
#     )

#     relevant_history_text = "\n".join(
#         f"{turn.role}: {turn.content}"
#         for turn in (relevant_history or [])
#     )

#     response = llm.invoke(
#         [
#             SystemMessage(
#                 content=(
#                     "Rewrite the latest user question into a standalone "
#                     "search query for document retrieval.\n\n"
#                     "Use the conversation summary for long-term context. "
#                     "Use recent conversation messages for immediate context. "
#                     "Use relevant historical messages when they provide "
#                     "important context from earlier in the conversation.\n\n"
#                     "Resolve references such as 'it', 'that', "
#                     "'this document', 'the above topic', or similar "
#                     "references to earlier conversation.\n\n"
#                     "Do not change the user's intended meaning. "
#                     "Do not add information that is not supported by "
#                     "the conversation.\n\n"
#                     "Return only the rewritten standalone search query. "
#                     "Do not answer the question."
#                 )
#             ),
#             HumanMessage(
#                 content=(
#                     f"Conversation summary:\n"
#                     f"{conversation_summary_text}\n\n"
#                     f"Recent conversation:\n"
#                     f"{recent_history_text or 'None'}\n\n"
#                     f"Relevant earlier conversation:\n"
#                     f"{relevant_history_text or 'None'}\n\n"
#                     f"Latest user question:\n"
#                     f"{question}"
#                 )
#             ),
#         ]
#     )

#     rewritten_question = str(response.content).strip()

#     return rewritten_question or question


# def _normalize_terms(text: str) -> set[str]:
#     words = re.findall(r"[a-zA-Z0-9]+", text.lower())

#     stop_words = {
#         "a", "an", "the", "is", "are", "was", "were", "to", "of", "for",
#         "and", "or", "in", "on", "with", "this", "that", "it", "its",
#         "how", "what", "why", "when", "where", "do", "does", "did",
#         "i", "you", "we", "they", "he", "she", "my", "your", "our",
#     }

#     return {
#         word
#         for word in words
#         if word not in stop_words and len(word) > 2
#     }


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
#         temperature=0,
#     )