from __future__ import annotations

from typing import List, Optional, Tuple

from app.retrieve import retrieve


def mock_llm(prompt: str) -> str:
    if "calculator" in prompt.lower() or "calc" in prompt.lower():
        return "Calculated result: 42"
    if "search" in prompt.lower():
        return "Search result: The sky is blue."
    return "Based on the context, here is the answer: The information requested is present in the context."


def calculator(expr: str) -> str:
    return "42"


def search_tool_mock(q: str) -> str:
    return "Mock web search result"


def generate_answer(query: str, context: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    if context is None:
        context = retrieve(query)
    
    tool_result = ""
    if "calc" in query.lower() or "+" in query:
        tool_result = calculator(query)
    elif "latest" in query.lower():
        tool_result = search_tool_mock(query)
        
    context_str = " ".join(context) + " " + tool_result
    prompt = f"Context: {context_str}\nQuery: {query}\nAnswer:"
    return mock_llm(prompt), context
