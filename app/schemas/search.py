"""Pydantic schemas for API requests/responses, tool args, and structured LLM outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------- API ----------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class SearchResultItem(BaseModel):
    path: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem] = Field(default_factory=list)
    best_match: str | None = None
    message: str | None = None


# ---------------- LLM structured outputs ----------------


class SearchPlan(BaseModel):
    """What Gemma understood about the user's request.

    NOTE: no file paths are invented here — only search intent.
    """

    needs_search: bool = Field(
        ..., description="True if a filesystem search is required."
    )
    location: str | None = Field(
        default=None,
        description="Windows path hint, e.g. 'E:\\'. None means default allowed root.",
    )
    search_terms: str = Field(
        ..., description="Key content words for the filesystem tool, e.g. 'aadhar card'."
    )
    extension_filter: str | None = Field(
        default=None, description="Lowercase extension without dot, e.g. 'pdf'. None if unknown."
    )
    reasoning: str = Field(default="", description="Short explanation of understanding.")


class RankedItem(BaseModel):
    path: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class RankedResults(BaseModel):
    items: list[RankedItem] = Field(default_factory=list)


# ---------------- Tool args (validated) ----------------


class SearchFilesArgs(BaseModel):
    location: str = Field(..., description="Directory to search under.")
    query: str = Field(..., min_length=1, description="Natural-language search terms.")
    limit: int = Field(default=10, ge=1, le=50)
    extension_filter: str | None = Field(default=None)


class ListDirectoryArgs(BaseModel):
    location: str
    limit: int = Field(default=100, ge=1, le=500)


class GetFileInfoArgs(BaseModel):
    path: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    ollama_reachable: bool = False
    model: str = ""
    allowed_roots: list[str] = Field(default_factory=list)
