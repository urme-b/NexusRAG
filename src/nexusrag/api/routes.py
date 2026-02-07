"""FastAPI routes for NexusRAG API."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, field_validator

from nexusrag.pipeline import get_nexusrag

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Security limits
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_QUERY_LENGTH = 2000
MAX_FILENAME_LENGTH = 255


class QueryRequest(BaseModel):
    """Query request body."""

    question: str  # Frontend sends 'question' not 'query'

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question length and content."""
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Question too long (max {MAX_QUERY_LENGTH} characters)")
        return v


class QueryResponse(BaseModel):
    """Query response body."""

    answer: str
    confidence: float
    sources: list[dict[str, Any]]
    processing_time_ms: float


class UploadResponse(BaseModel):
    """Upload response body."""

    success: bool
    document_id: str
    filename: str
    chunk_count: int
    word_count: int
    error: str | None = None


class StatusResponse(BaseModel):
    """System status response."""

    total_documents: int
    total_chunks: int
    total_words: int
    llm_available: bool
    llm_model: str
    embedding_model: str


class DocumentInfo(BaseModel):
    """Document metadata."""

    id: str
    filename: str
    original_filename: str = ""
    word_count: int
    chunk_count: int = 0
    file_type: str = ""
    uploaded_at: str = ""


class DocumentListResponse(BaseModel):
    """Document list response."""

    documents: list[DocumentInfo]
    total_documents: int = 0
    total_chunks: int = 0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    llm_available: bool
    llm_model: str = ""
    embedding_model: str = ""
    total_documents: int = 0
    total_chunks: int = 0


class DeleteResponse(BaseModel):
    """Delete response body."""

    success: bool
    message: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for frontend connection status."""
    try:
        rag = get_nexusrag()
        stats = rag.get_stats()
        return HealthResponse(
            status="ok",
            llm_available=stats.llm_available,
            llm_model=stats.llm_model,
            embedding_model=stats.embedding_model,
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks,
        )
    except Exception:
        return HealthResponse(status="error", llm_available=False)


@router.post("/ingest", response_model=UploadResponse)
async def ingest_document(
    file: Annotated[UploadFile, File(description="Document to upload")],
) -> UploadResponse:
    """
    Upload a document for ingestion.

    Accepts PDF, DOCX, TXT, or MD files. Max size: 50MB.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate filename length
    if len(file.filename) > MAX_FILENAME_LENGTH:
        raise HTTPException(status_code=400, detail="Filename too long")

    # Sanitize filename - extract just the name, no paths
    filename = file.filename.split("/")[-1].split("\\")[-1]
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = "." + filename.split(".")[-1].lower()
    if ext not in [".pdf", ".docx", ".txt", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: PDF, DOCX, TXT, MD",
        )

    try:
        # Read file with size limit
        content = await file.read()

        # Check file size
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        rag = get_nexusrag()
        result = rag.ingest_bytes(content, filename, ext)

        return UploadResponse(
            success=result.success,
            document_id=result.document_id,
            filename=result.filename,
            chunk_count=result.chunk_count,
            word_count=result.word_count,
            error=result.error,
        )
    except HTTPException:
        raise
    except Exception:
        # Log the actual error but return generic message
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=500, detail="Failed to process document") from None


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query the knowledge base.

    Returns an answer with sources and confidence score.
    """
    # Validation is now handled by pydantic field_validator
    try:
        rag = get_nexusrag()
        response = rag.query(request.question)

        # Build document name lookup (prefer original_filename)
        doc_names: dict[str, str] = {}
        try:
            docs = rag.list_documents()
            for doc in docs:
                name = doc.get("original_filename") or doc.get("filename") or "Unknown"
                # Skip temp file names
                if name.startswith("tmp"):
                    name = doc.get("original_filename") or "Unknown"
                doc_names[doc.get("id", "")] = name
        except Exception:
            pass

        # Build clean sources response
        sources = []
        for idx, source in enumerate(response.sources):
            # Get original filename (not temp name)
            filename = doc_names.get(source.document_id) or source.document_name or "Unknown"
            # Strip path if present
            if "/" in filename:
                filename = filename.split("/")[-1]
            # Skip temp names
            if filename.startswith("tmp"):
                filename = doc_names.get(source.document_id) or "Unknown"

            # Truncate content for response
            content = source.content
            if len(content) > 500:
                content = content[:500] + "..."

            sources.append(
                {
                    "index": idx + 1,
                    "content": content,
                    "text": content,
                    "filename": filename,
                    "document_id": source.document_id,
                    "section_title": source.section_title or "",
                    "page": source.page_number,
                    "page_number": source.page_number,
                    "score": round(source.score, 3),
                }
            )

        return QueryResponse(
            answer=response.answer,
            confidence=response.confidence,
            sources=sources,
            processing_time_ms=round(response.processing_time_ms, 1),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail="Failed to process query") from None


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all ingested documents."""
    try:
        rag = get_nexusrag()
        docs = rag.list_documents()
        stats = rag.get_stats()

        doc_list = []
        for doc in docs:
            # Get the original filename, not temp name
            filename = doc.get("original_filename") or doc.get("filename") or "Unknown"
            if filename.startswith("tmp"):
                filename = doc.get("original_filename") or "Unknown"

            doc_list.append(
                DocumentInfo(
                    id=doc.get("id", ""),
                    filename=filename,
                    original_filename=doc.get("original_filename", filename),
                    word_count=doc.get("word_count", 0),
                    chunk_count=doc.get("chunk_count", 0),
                    file_type=doc.get("file_type", ""),
                    uploaded_at=doc.get("uploaded_at", ""),
                )
            )

        return DocumentListResponse(
            documents=doc_list,
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks,
        )
    except Exception:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents") from None


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get system status and statistics."""
    try:
        rag = get_nexusrag()
        stats = rag.get_stats()

        return StatusResponse(
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks,
            total_words=stats.total_words,
            llm_available=stats.llm_available,
            llm_model=stats.llm_model,
            embedding_model=stats.embedding_model,
        )
    except Exception:
        logger.exception("Failed to get status")
        raise HTTPException(status_code=500, detail="Failed to retrieve status") from None


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str) -> DeleteResponse:
    """Delete a specific document."""
    # Basic validation of document_id format
    if not document_id or len(document_id) > 64:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    try:
        rag = get_nexusrag()
        success = rag.delete_document(document_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return DeleteResponse(success=True, message="Document deleted")
    except HTTPException:
        raise
    except ValueError:
        # Validation errors from document store
        raise HTTPException(status_code=400, detail="Invalid document ID format") from None
    except Exception:
        logger.exception("Failed to delete document")
        raise HTTPException(status_code=500, detail="Failed to delete document") from None


@router.delete("/documents", response_model=DeleteResponse)
async def clear_all_documents() -> DeleteResponse:
    """Delete all documents and reset the system."""
    try:
        rag = get_nexusrag()
        rag.clear_all()

        return DeleteResponse(success=True, message="All documents cleared")
    except Exception:
        logger.exception("Failed to clear documents")
        raise HTTPException(status_code=500, detail="Failed to clear documents") from None
