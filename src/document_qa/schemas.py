"""Public request and response models for the HTTP API."""

from datetime import datetime

from pydantic import BaseModel, Field


class CorpusCreated(BaseModel):
    corpus_id: str
    documents: list[str]
    chunk_count: int
    expires_at: datetime
    reused: bool = False


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)


class Citation(BaseModel):
    document_name: str
    page_number: int = Field(ge=1)
    excerpt: str


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]


class ServiceStatus(BaseModel):
    status: str
