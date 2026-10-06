"""LangGraph workflow. Kept deliberately simple for milestone 1:

START -> understand (Gemma) -> search (tools) -> rank (Gemma) -> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    AgentState,
    do_search_node,
    rank_results_node,
    understand_query_node,
)


def _needs_search(state: AgentState) -> str:
    plan = state.get("plan") or {}
    return "search" if plan.get("needs_search", True) else "rank"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("understand", understand_query_node)
    builder.add_node("search", do_search_node)
    builder.add_node("rank", rank_results_node)

    builder.add_edge(START, "understand")
    builder.add_conditional_edges("understand", _needs_search,
                                  {"search": "search", "rank": "rank"})
    builder.add_edge("search", "rank")
    builder.add_edge("rank", END)
    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_search(query: str, timeout_seconds: float = 300.0) -> dict:
    """Run the agent synchronously and return a plain dict for FastAPI."""
    import concurrent.futures

    graph = get_graph()

    def _invoke():
        return graph.invoke({"query": query})

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_invoke)
        final = future.result(timeout=timeout_seconds)
    candidates = final.get("candidates") or []
    results = final.get("results") or []
    return {
        "query": query,
        "candidates": candidates,
        "results": results,
        "best_match": final.get("best_match"),
        "error": final.get("error"),
        "llm_used": final.get("llm_used", False),
    }
