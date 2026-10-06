"""Central configuration. Everything local; values come from environment."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_allowed_roots(raw: str | None) -> list[str]:
    if not raw:
        return ["E:\\"]
    parts: list[str] = []
    # Support both comma and semicolon separators (Windows paths contain no commas normally,
    # but drive lists are often semicolon separated).
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip().strip('"').strip("'")
        if chunk:
            parts.append(chunk)
    return parts or ["E:\\"]


class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    allowed_roots: list[str] = _parse_allowed_roots(os.getenv("ALLOWED_ROOTS"))
    index_db_path: str = os.getenv(
        "FILE_INDEX_DB", str(PROJECT_ROOT / "data" / "file_index.db")
    )
    search_timeout_seconds: float = float(os.getenv("SEARCH_TIMEOUT_SECONDS", "25"))
    max_candidates: int = int(os.getenv("MAX_CANDIDATES", "10"))
    # Safety caps for live filesystem walks (milestone 1 has no vector search).
    max_dirs_per_search: int = int(os.getenv("MAX_DIRS_PER_SEARCH", "4000"))
    max_files_visited: int = int(os.getenv("MAX_FILES_VISITED", "60000"))
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "180"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
