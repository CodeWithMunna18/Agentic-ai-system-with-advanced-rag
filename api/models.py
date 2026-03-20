# api/models.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: Pydantic models serve two purposes in FastAPI:
#
# 1. VALIDATION — FastAPI automatically validates incoming JSON
#    against these models. If a required field is missing or the
#    wrong type, FastAPI returns a 422 error before your code runs.
#
# 2. DOCUMENTATION — FastAPI auto-generates OpenAPI (Swagger) docs
#    from these models. Visit /docs to see the interactive API docs.
#
# Request models  = what the client sends TO the API
# Response models = what the API sends BACK to the client
# ─────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional


# ── Request models (client → API) ─────────────────────────────

class AskRequest(BaseModel):
    """Request body for /ask endpoint."""
    question: str = Field(
        ...,                              # ... means required
        min_length=3,
        max_length=1000,
        description="The question to answer from your documents",
        examples=["What is RAG and how does it work?"]
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Scope search to a specific file (e.g. 'report.pdf')"
    )
    mode: str = Field(
        default="hybrid",
        description="Search mode: 'hybrid', 'dense', or 'multi_doc'"
    )


class IngestRequest(BaseModel):
    """Request body for triggering re-ingestion."""
    documents_dir: str = Field(
        default="documents",
        description="Path to the documents folder to ingest"
    )


# ── Response models (API → client) ────────────────────────────

class SourceModel(BaseModel):
    """Metadata about one retrieved source chunk."""
    source_file: str
    score: float
    chunk_index: int
    preview: str


class AskResponse(BaseModel):
    """Response from /ask endpoint."""
    question: str
    answer: str
    sources: list[SourceModel]
    mode: str
    chunks_retrieved: int


class DocumentInfo(BaseModel):
    """Info about one document in the store."""
    file_name: str
    chunk_count: int


class StatusResponse(BaseModel):
    """Response from /status endpoint."""
    status: str
    total_chunks: int
    documents: list[str]
    num_documents: int


class UploadResponse(BaseModel):
    """Response from /upload endpoint."""
    success: bool
    file_name: str
    chunks_created: int
    message: str

class IngestResponse(BaseModel):
    """Response from /ingest endpoint."""
    success: bool
    documents_loaded: int
    chunks_created: int
    message: str