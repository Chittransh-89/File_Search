"""Ollama / Gemma connection helpers. Local-only: http://localhost:11434 by default."""

from __future__ import annotations

import urllib.request
import json

from langchain_ollama import ChatOllama

from app.config import settings


def get_chat_model(temperature: float = 0.0, timeout: float | None = None,
                   num_predict: int | None = None) -> ChatOllama:
    kwargs: dict = {
        "base_url": settings.ollama_base_url,
        "model": settings.ollama_model,
        "temperature": temperature,
        "num_ctx": 4096,
        "timeout": (timeout or settings.llm_timeout_seconds),
    }
    if num_predict is not None:
        kwargs["num_predict"] = num_predict
    return ChatOllama(**kwargs)


def check_ollama() -> dict:
    """Return {'reachable': bool, 'models': [...]} without sending any file data."""
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        models = [m.get("name", "") for m in payload.get("models", [])]
        return {"reachable": True, "models": models}
    except Exception as exc:
        return {"reachable": False, "models": [], "error": str(exc)}
