import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.agent.tools import calculator, document_search, web_search
from app.config import Settings
from app.schemas import ChatTurn

RECENT_HISTORY_LIMIT = 8
SUMMARY_TRIGGER_LIMIT = 12


SYSTEM_PROMPT = """
You are an AI Financial Analysis Agent.

Available tools:
- document_search: use for uploaded financial documents
- web_search: use for current, external, or missing information
- calculator: use for financial math

Rules:
- Prefer uploaded documents first.
- Use web_search only when it is enabled and documents are missing required information or the question is current/time-sensitive.
- For current events, recent results, latest data, winners, market facts, and time-sensitive questions, do not answer from memory. Use web_search when it is enabled.
- Do not invent financial numbers.
- Use calculator for percentage changes, margins, ratios, and variances.
- If information cannot be found, say so clearly.
- Give concise management-level analysis.
"""


def run_agent(
    question: str,
    history: list[ChatTurn],
    settings: Settings,
    max_steps: int = 8,
    use_web_search: bool = True,
    return_trace: bool = True,
) -> dict:
    @tool
    def document_search_tool(query: str) -> list[dict]:
        """Search uploaded financial documents."""
        return document_search(query=query, settings=settings)

    @tool
    def web_search_tool(query: str) -> dict:
        """Search the web for current or missing external information."""
        return web_search(query=query, settings=settings)

    @tool
    def calculator_tool(expression: str) -> dict:
        """Calculate a financial math expression."""
        return calculator(expression=expression)

    tools = [document_search_tool, calculator_tool]

    if use_web_search:
        tools.append(web_search_tool)

    tools_by_name = {t.name: t for t in tools}

    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
    ).bind_tools(tools)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    trace = []

    conversation_summary = summarize_agent_history(llm, history)

    if conversation_summary:
        messages.append(
            SystemMessage(
                content=f"Conversation summary so far:\n{conversation_summary}"
            )
        )

    recent_history = history[-RECENT_HISTORY_LIMIT:]

    for turn in recent_history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        elif turn.role == "assistant":
            messages.append(AIMessage(content=turn.content))

    messages.append(HumanMessage(content=question))

    if use_web_search and _requires_external_verification(question):
        result = web_search(query=question, settings=settings)
        trace.append(
            {
                "step": len(trace) + 1,
                "tool": "web_search_tool",
                "input": {"query": question},
                "output": result,
            }
        )
        messages.append(
            SystemMessage(
                content=(
                    "External verification was required for this time-sensitive "
                    "or external question. Use this web_search result as evidence "
                    "instead of answering from memory:\n"
                    f"{json.dumps(result, ensure_ascii=True)}"
                )
            )
        )

    for step in range(max_steps):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return {
                "answer": str(response.content),
                "trace": trace if return_trace else None,
            }

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            result = tools_by_name[tool_name].invoke(tool_args)

            trace.append(
                {
                    "step": len(trace) + 1,
                    "tool": tool_name,
                    "input": tool_args,
                    "output": result,
                }
            )

            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=True),
                    tool_call_id=tool_call["id"],
                )
            )

    return {
        "answer": "I could not complete the analysis within the tool-call limit.",
        "trace": trace if return_trace else None,
    }


def summarize_agent_history(
    llm: ChatOpenAI,
    history: list[ChatTurn],
) -> str | None:
    if len(history) < SUMMARY_TRIGGER_LIMIT:
        return None

    older_history = history[:-RECENT_HISTORY_LIMIT]

    if not older_history:
        return None

    history_text = "\n".join(
        f"{turn.role}: {turn.content}"
        for turn in older_history
    )

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Summarize this conversation for an AI Financial Analysis Agent. "
                    "Preserve companies, financial periods, metrics, document names, "
                    "numbers already discussed, assumptions, unresolved questions, "
                    "and user preferences. Keep it concise. "
                    "Do not add facts that are not in the conversation."
                )
            ),
            HumanMessage(content=f"Conversation to summarize:\n{history_text}"),
        ]
    )

    summary = str(response.content).strip()
    return summary or None


def _requires_external_verification(question: str) -> bool:
    normalized_question = question.lower()

    trigger_terms = {
        "latest",
        "current",
        "today",
        "yesterday",
        "recent",
        "now",
        "news",
        "stock price",
        "share price",
        "market cap",
        "winner",
        "won",
        "world cup",
    }

    if any(term in normalized_question for term in trigger_terms):
        return True

    current_or_future_years = re.findall(r"\b20\d{2}\b", normalized_question)
    return any(int(year) >= 2024 for year in current_or_future_years)
