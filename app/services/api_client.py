"""Thin HTTP client: Streamlit -> FastAPI. No agent logic lives here.

The backend (FastAPI -> LangGraph -> LangChain -> Gemma -> filesystem tools)
does all understanding, searching, and ranking. This module only transports
the query and returns the backend's JSON, converting transport problems into
a single ServiceUnavailable error the UI can display cleanly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import urllib.request
import urllib.error
import json


def get_api_base_url() -> str:
    return os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


class ServiceUnavailable(Exception):
    """FastAPI/Ollama is unreachable. Never carries a stack trace to the UI."""


@dataclass
class SearchResult:
    path: str
    score: float
    reason: str

    @property
    def filename(self) -> str:
        return self.path.rstrip("\\/").split("\\")[-1].split("/")[-1]


@dataclass
class SearchOutcome:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    best_match: str | None = None
    message: str | None = None

    @property
    def best(self) -> SearchResult | None:
        if not self.results:
            return None
        if self.best_match:
            for r in self.results:
                if r.path == self.best_match:
                    return r
        return self.results[0]


@dataclass
class BackendHealth:
    reachable: bool = False
    ollama_reachable: bool = False
    model: str = ""
    allowed_roots: list[str] = field(default_factory=list)


def _request(method: str, url: str, payload: dict | None = None,
             timeout: float = 300.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8", errors="ignore"))
            msg = detail.get("detail", "") if isinstance(detail, dict) else ""
        except Exception:
            msg = ""
        raise ServiceUnavailable(
            "The local search service returned an error"
            + (f": {msg}" if msg else ".")
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ServiceUnavailable(
            "Local AI service is unavailable. "
            "Make sure Ollama and the FastAPI backend are running."
        ) from exc


def check_health(timeout: float = 8.0) -> BackendHealth:
    """Fast, used by the sidebar on every render. Never raises."""
    try:
        body = _request("GET", get_api_base_url() + "/health", timeout=timeout)
        return BackendHealth(
            reachable=True,
            ollama_reachable=bool(body.get("ollama_reachable", False)),
            model=str(body.get("model", "")),
            allowed_roots=list(body.get("allowed_roots", []) or []),
        )
    except ServiceUnavailable:
        return BackendHealth(reachable=False)


def search_files(query: str, timeout: float = 300.0) -> SearchOutcome:
    """POST /search. Raises ServiceUnavailable on transport problems only."""
    body = _request("POST", get_api_base_url() + "/search",
                    {"query": query}, timeout=timeout)
    results = [
        SearchResult(path=r.get("path", ""),
                     score=float(r.get("score", 0.0)),
                     reason=r.get("reason", ""))
        for r in (body.get("results") or [])
        if r.get("path")
    ]
    return SearchOutcome(
        query=body.get("query", query),
        results=results,
        best_match=body.get("best_match"),
        message=body.get("message"),
    )
