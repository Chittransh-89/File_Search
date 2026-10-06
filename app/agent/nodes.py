"""LangGraph nodes. Simple pipeline: understand -> search -> rank.

Gemma = brain (language understanding + ranking).
Python filesystem tools = hands (real disk access, read-only).
"""

from __future__ import annotations

import re
from typing import TypedDict

from app.agent.prompts import RANK_PROMPT, UNDERSTAND_PROMPT
from app.config import settings
from app.schemas.search import RankedResults, SearchPlan
from app.services.ollama import get_chat_model
from app.tools.filesystem import (
    detect_extension_hint,
    list_directory_core,
    resolve_and_validate,
    search_files_core,
    tokenize,
)


class AgentState(TypedDict, total=False):
    query: str
    plan: dict
    candidates: list
    results: list
    best_match: str | None
    error: str | None
    llm_used: bool


# ---------------- fallback (LLM unavailable) ----------------

_DRIVE_RE = re.compile(r"\b([a-zA-Z])\s*[:\\]?\s*drive\b", re.IGNORECASE)
_DRIVE_LETTER_RE = re.compile(r"\b([a-zA-Z])\s*:[\\/]")


def _fallback_plan(query: str) -> SearchPlan:
    """Generic fallback when Ollama is down: regex drive hints + generic tokens.

    Deliberately NOT a domain synonym list — just location/extension detection
    plus the raw content words. The filesystem scorer stays generic too.
    """
    lowered = query.lower()
    greetings = {"hi", "hello", "hey", "thanks", "thank you", "namaste", "shukriya"}
    if lowered.strip() in greetings:
        return SearchPlan(needs_search=False, location=None, search_terms=query,
                          reasoning="Greeting, no search needed.")

    location: str | None = None
    m = _DRIVE_LETTER_RE.search(query) or _DRIVE_RE.search(query)
    if m:
        letter = m.group(1).upper()
        candidate = f"{letter}:\\"
        try:
            resolve_and_validate(candidate)
            location = candidate
        except Exception:
            location = None
    # "college ki DBMS files" style queries have no drive -> default root.
    tokens = tokenize(query)
    search_terms = " ".join(tokens) if tokens else query.strip()
    ext = detect_extension_hint(query)
    return SearchPlan(needs_search=True, location=location, search_terms=search_terms,
                      extension_filter=ext,
                      reasoning="Fallback parser (LLM unavailable): generic tokens used.")


def understand_query_node(state: AgentState) -> AgentState:
    query = state.get("query", "").strip()
    if not query:
        return {**state, "plan": SearchPlan(
            needs_search=False, search_terms="",
            reasoning="Empty query.").model_dump(),
            "llm_used": False, "error": "Empty query."}
    prompt = UNDERSTAND_PROMPT.format(
        allowed_roots=", ".join(settings.allowed_roots),
        default_root=settings.allowed_roots[0] if settings.allowed_roots else "E:\\",
        query=query,
    )
    try:
        llm = get_chat_model()
        structured = llm.with_structured_output(SearchPlan)
        plan: SearchPlan = structured.invoke(prompt)  # LangChain structured output
        # Validate the LLM's location against allowed roots (never trust blindly).
        if plan.location:
            try:
                plan.location = str(resolve_and_validate(plan.location))
            except Exception:
                plan.location = None
        return {**state, "plan": plan.model_dump(), "llm_used": True}
    except Exception as exc:
        plan = _fallback_plan(query)
        err = f"LLM understand step failed, used fallback ({type(exc).__name__})."
        return {**state, "plan": plan.model_dump(), "llm_used": False,
                "error": ((state.get("error") or "") + " " + err).strip()}


def do_search_node(state: AgentState) -> AgentState:
    plan_d = state.get("plan") or {}
    try:
        plan = SearchPlan(**plan_d)
    except Exception:
        plan = _fallback_plan(state.get("query", ""))
    if not plan.needs_search:
        return {**state, "candidates": []}
    location = plan.location or (settings.allowed_roots[0] if settings.allowed_roots else "E:\\")
    try:
        location = str(resolve_and_validate(location))
    except Exception as exc:
        return {**state, "candidates": [],
                "error": ((state.get("error") or "") + f" Blocked location: {exc}").strip()}
    try:
        candidates = search_files_core(
            location=location,
            query=plan.search_terms or state.get("query", ""),
            limit=settings.max_candidates,
            extension_filter=plan.extension_filter,
        )
        return {**state, "candidates": candidates}
    except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError) as exc:
        return {**state, "candidates": [],
                "error": ((state.get("error") or "") + f" Search failed: {exc}").strip()}
    except Exception as exc:
        return {**state, "candidates": [],
                "error": ((state.get("error") or "") + f" Unexpected search error: {exc}").strip()}


def _fallback_rank(candidates: list, limit: int) -> list[dict]:
    out: list[dict] = []
    for c in (candidates or [])[:limit]:
        out.append({
            "path": c.get("path", ""),
            "score": float(c.get("score", 0.5)),
            "reason": c.get("reason", "Filesystem match"),
        })
    return out


def rank_results_node(state: AgentState) -> AgentState:
    candidates: list = state.get("candidates") or []
    query = state.get("query", "")
    if not candidates:
        return {**state, "results": [], "best_match": None}
    # Grounding set: LLM may ONLY pick from these exact paths.
    # Send only the top few to the LLM to keep local generation fast;
    # remaining candidates rejoin below with their filesystem scores.
    LLM_SHORTLIST = 6
    shortlist = candidates[:LLM_SHORTLIST]
    allowed = {c.get("path") for c in shortlist}
    cand_text = "\n".join(
        f"- {c.get('path')} (ext=.{c.get('extension','')}, why={c.get('reason')})"
        for c in shortlist
    )
    prompt = RANK_PROMPT.format(query=query, candidates=cand_text, limit=LLM_SHORTLIST)
    llm_used = bool(state.get("llm_used"))
    try:
        # No num_predict cap: Windows paths tokenize long, and a cap truncates
        # the ranking (observed empty/partial outputs with small caps on CPU).
        llm = get_chat_model()
        raw = llm.invoke(prompt).content or ""
        results: list[dict] = []
        for line in str(raw).splitlines():
            # Expected: <path> || <score> || <reason>
            if "||" not in line:
                continue
            parts = [p.strip() for p in line.split("||")]
            if len(parts) < 2:
                continue
            path_part = parts[0].lstrip("-•*0123456789. ").strip().strip('"').strip("'")
            if path_part not in allowed:
                continue  # drop hallucinated paths — grounding rule
            try:
                score = max(0.0, min(1.0, float(parts[1])))
            except ValueError:
                continue
            reason = parts[2] if len(parts) > 2 else "LLM ranked match"
            results.append({"path": path_part, "score": score, "reason": reason})
        # The model sometimes wraps lines in numbering/bullets — the strip above
        # handles that; anything unparseable is simply ignored (safe fallback below).
        parsed_any = bool(results)
        if not parsed_any:
            # Unparseable output (not a judgement): fall back to filesystem order.
            results = _fallback_rank(candidates, settings.max_candidates)
            llm_used = False
        # Noise gate (same 0.30 bar as the filesystem tool): when the LLM does
        # answer, it is honest about weak fits via low scores — honour that
        # instead of showing junk.
        results = [r for r in results if r["score"] >= 0.30]
        if not results:
            # Either the LLM judged nothing worthy, or nothing survived the gate.
            return {**state, "results": [], "best_match": None, "llm_used": llm_used}
        else:
            # Append non-shortlisted candidates with filesystem scores.
            seen = {r["path"] for r in results}
            for c in candidates:
                if c.get("path") not in seen and len(results) < settings.max_candidates:
                    results.append({
                        "path": c.get("path", ""),
                        "score": float(c.get("score", 0.5)),
                        "reason": c.get("reason", "Filesystem match"),
                    })
        results.sort(key=lambda r: r["score"], reverse=True)
        best = results[0]["path"] if results else None
        return {**state, "results": results, "best_match": best, "llm_used": llm_used}
    except Exception as exc:
        results = _fallback_rank(candidates, settings.max_candidates)
        best = results[0]["path"] if results else None
        err = f"LLM rank step failed, used filesystem scores ({type(exc).__name__})."
        return {**state, "results": results, "best_match": best, "llm_used": False,
                "error": ((state.get("error") or "") + " " + err).strip()}


# Re-export tool-backed helpers so the API layer can enrich results if needed.
__all__ = ["AgentState", "understand_query_node", "do_search_node", "rank_results_node",
           "list_directory_core"]
