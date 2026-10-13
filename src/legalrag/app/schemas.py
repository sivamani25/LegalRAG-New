"""Pydantic request/response schemas for the LegalRAG API (M7)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_history: list[ChatTurn] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped


class EvidenceItem(BaseModel):
    chunk_id: str
    category: str | None = None
    source_doc: str | None = None
    section: str | None = None
    heading: str | None = None
    parent: str | None = None
    hierarchy_path: list[str] = Field(default_factory=list)
    court: str | None = None
    date: str | None = None
    retrieval_score: float | None = None
    text: str


class RelatedCase(BaseModel):
    chunk_id: str
    source_doc: str | None = None
    section: str | None = None
    heading: str | None = None
    court: str | None = None
    date: str | None = None
    retrieval_score: float | None = None
    text: str


class CitationValidation(BaseModel):
    valid: bool
    has_citations: bool
    used_sources: list[str]
    unknown_sources: list[str]
    support_verified: bool
    warnings: list[str]


class ChatResponse(BaseModel):
    question: str
    answer: str
    evidence: list[EvidenceItem]
    related_cases: list[RelatedCase]
    limitations: list[str]
    citation_validation: CitationValidation
    retrieved_count: int
    used_count: int


class HealthResponse(BaseModel):
    status: str
    categories: list[str]
    retrievers_loaded: bool


class CategoryEval(BaseModel):
    num_cases: int
    metrics_by_k: dict[str, dict[str, float]]


class EvalResponse(BaseModel):
    k_values: list[int]
    cases_file: str
    categories: dict[str, CategoryEval]
    overall_macro: dict[str, dict[str, float]]
