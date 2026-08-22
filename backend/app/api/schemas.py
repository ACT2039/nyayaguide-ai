"""
NyayaGuide AI — API Request/Response Schemas
Pydantic models for FastAPI request validation and response serialization.
"""
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator

from ..config import MAX_QUESTION_LENGTH


class AskRequest(BaseModel):
    """Validated request body for the /api/ask endpoint."""
    question: str = Field(
        ...,
        description="The legal or civic rights question to ask NyayaGuide AI.",
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        examples=["How can I file an RTI application?"]
    )

    @field_validator("question", mode="before")
    @classmethod
    def strip_and_validate_question(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Question must be a string.")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be empty or whitespace-only.")
        if len(stripped) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters."
            )
        return stripped


class SourceCitationResponse(BaseModel):
    """Structured source citation returned by the API."""
    document: str = Field(..., description="Source PDF filename")
    category: str = Field(..., description="RTI or CONSUMER")
    page: int = Field(..., description="Page number in the source PDF")
    legal_reference: Optional[str] = Field(default=None, description="Section, Rule, or Chapter reference")
    title: str = Field(..., description="Official document title")
    source: str = Field(..., description="Issuing authority")
    source_url: Optional[str] = Field(default=None, description="Official source URL if available")
    chunk_id: str = Field(..., description="ID of the chunk cited")


class AskResponse(BaseModel):
    """Structured response from the /api/ask endpoint."""
    question: str = Field(..., description="User's original query")
    answer: str = Field(..., description="Grounded answer with source citations")
    sources: List[SourceCitationResponse] = Field(
        default_factory=list,
        description="Programmatic citations from retrieved chunks"
    )
    is_abstention: bool = Field(
        default=False,
        description="True if the query was outside the knowledge base scope"
    )
    model_used: Optional[str] = Field(
        default=None,
        description="LLM model used for generation (null if abstained)"
    )
    top_score: float = Field(
        default=0.0,
        description="Highest cosine similarity score among retrieved chunks"
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="3-4 suggested follow-up questions grounded in the context"
    )


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str = Field(default="ok", description="Service health status")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    detail: str = Field(..., description="Human-readable error description")


# Re-export Phase 7 Registry Schemas for API documentation
from ..models.registry import (
    DocumentRecord,
    KnowledgeBaseStats,
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentDeleteResponse,
    DocumentStatus
)


