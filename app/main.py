"""FastAPI entrypoint: POST /search returns real, grounded file paths."""

from __future__ import annotations

import concurrent.futures

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.agent.graph import run_search
from app.config import settings
from app.schemas.search import HealthResponse, SearchRequest, SearchResponse, SearchResultItem
from app.services.ollama import check_ollama
from app.tools.filesystem import build_index

app = FastAPI(title="Local File Finder AI", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = check_ollama()
    return HealthResponse(
        ollama_reachable=status.get("reachable", False),
        model=settings.ollama_model,
        allowed_roots=settings.allowed_roots,
    )


@app.post("/index")
def rebuild_index() -> dict:
    """(Re)build the local SQLite index for all allowed roots. Read-only on user files."""
    summary: list[dict] = []
    for root in settings.allowed_roots:
        try:
            summary.append(build_index(root))
        except Exception as exc:
            summary.append({"root": root, "error": str(exc)})
    return {"ok": True, "roots": summary}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_search, query)
            outcome = future.result(timeout=300.0)
    except concurrent.futures.TimeoutError:
        return SearchResponse(query=query, results=[], best_match=None,
                              message="Search timed out. Try a narrower location.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    results = [
        SearchResultItem(path=r["path"], score=float(r["score"]), reason=r.get("reason", ""))
        for r in (outcome.get("results") or [])
        if r.get("path")
    ]
    best = outcome.get("best_match")
    err = (outcome.get("error") or "").strip()
    if not results:
        msg = "I couldn't find a suitable matching file in the requested location."
        if err:
            msg += f" ({err})"
        return SearchResponse(query=query, results=[], best_match=None, message=msg)
    return SearchResponse(query=query, results=results, best_match=best,
                          message=err or None)


@app.exception_handler(Exception)
async def _unhandled(request, exc):  # fail safely, never leak internals
    return JSONResponse(status_code=500, content={"detail": "Internal error during search."})
