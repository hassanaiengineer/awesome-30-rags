from __future__ import annotations

from typing import List, Optional, Tuple

from app.retrieve import retrieve


def mock_llm(prompt: str) -> str:
    if "calculator" in prompt.lower() or "calc" in prompt.lower():
        return "Calculated result: 42"
    if "search" in prompt.lower():
        return "Search result: The sky is blue."
    return "Based on the context, here is the answer: The information requested is present in the context."


def generate_answer(query: str, context: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    trace = "THINK: Analyzing query...\n"
    if context is None:
        trace += "RETRIEVE: Fetching documents...\n"
        context = retrieve(query)
    context_str = " ".join(context)
    trace += "REFINE: Processing context...\n"
    prompt = f"Context: {context_str}\nQuery: {query}\nAnswer:"
    ans = mock_llm(prompt)
    trace += "ANSWER: Generating final response."
    return ans + "\n\nTrace:\n" + trace, context
