"""
NyayaGuide AI — API Routes
GET /health and POST /api/ask endpoints.
All RAG logic stays in the existing NyayaRAGPipeline — the API is a thin HTTP layer.
"""
import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .schemas import (
    AskRequest,
    AskResponse,
    SourceCitationResponse,
    HealthResponse,
    ErrorResponse,
)

logger = logging.getLogger("nyayaguide.api")

router = APIRouter()


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns service health status. Does not call OpenRouter or load heavy resources.",
    tags=["System"],
)
async def health_check():
    return HealthResponse(status="ok")


# ──────────────────────────────────────────────
# POST /api/ask
# ──────────────────────────────────────────────
@router.post(
    "/api/ask",
    response_model=AskResponse,
    summary="Ask a Legal Question",
    description=(
        "Submit a civic/legal rights question about Indian government acts and rules. "
        "The system retrieves relevant source documents, verifies relevance, and generates "
        "a grounded answer with programmatic source citations and suggested follow-ups. "
        "Out-of-domain questions are automatically abstained without calling the LLM."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Invalid or empty question"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        502: {"model": ErrorResponse, "description": "LLM service error (auth, rate limit, payment)"},
        504: {"model": ErrorResponse, "description": "LLM service timeout"},
    },
    tags=["RAG"],
)
async def ask_question(body: AskRequest, request: Request):
    """
    Delegates to the singleton NyayaRAGPipeline stored in app.state.
    """
    pipeline = request.app.state.pipeline

    try:
        rag_response = pipeline.ask(question=body.question)

        # Map SourceCitation → SourceCitationResponse (including source_url)
        sources = [
            SourceCitationResponse(
                document=s.document,
                category=s.category,
                page=s.page,
                legal_reference=s.legal_reference,
                title=s.title,
                source=s.source,
                source_url=s.source_url,
                chunk_id=s.chunk_id,
            )
            for s in rag_response.sources
        ]

        return AskResponse(
            question=rag_response.question,
            answer=rag_response.answer,
            sources=sources,
            is_abstention=rag_response.is_abstention,
            model_used=rag_response.model_used,
            top_score=rag_response.top_score,
            follow_up_questions=rag_response.follow_up_questions,
        )

    except ValueError as exc:
        # Bad input that slipped past Pydantic (e.g. empty after pipeline strip)
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    except PermissionError:
        # OpenRouter 401 — auth failure
        logger.error("OpenRouter authentication failed during /api/ask")
        return JSONResponse(
            status_code=502,
            content={"detail": "LLM service authentication failed. Please check server configuration."},
        )

    except TimeoutError:
        logger.error("OpenRouter request timed out during /api/ask")
        return JSONResponse(
            status_code=504,
            content={"detail": "LLM service request timed out. Please try again."},
        )

    except RuntimeError as exc:
        # OpenRouter rate-limit, payment, or other API errors
        safe_msg = str(exc)
        # Sanitize: mask potential API key patterns in server logs
        import re
        safe_msg = re.sub(r'sk-or-[A-Za-z0-9-]+', '***', safe_msg)
        safe_msg = re.sub(r'hf_[A-Za-z0-9]+', '***', safe_msg)
        # Strip anything that looks like a key
        if "rate limit" in safe_msg.lower():
            detail = "LLM service rate limit reached. Please wait and try again."
        elif "payment" in safe_msg.lower() or "credit" in safe_msg.lower():
            detail = "LLM service quota or payment issue. Please check server configuration."
        else:
            detail = "LLM service encountered an error. Please try again later."
        logger.error("RuntimeError in /api/ask: %s", safe_msg)
        return JSONResponse(status_code=502, content={"detail": detail})

    except Exception:
        # Catch-all: log full traceback server-side, return safe message to client
        logger.exception("Unexpected error in /api/ask")
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred. Please try again later."},
        )


# ──────────────────────────────────────────────
# Document Management & Knowledge Base APIs (Admin Protected)
# ──────────────────────────────────────────────
from typing import Optional, List
from fastapi import UploadFile, File, Form, Depends, HTTPException, status, BackgroundTasks
from starlette.concurrency import run_in_threadpool
from .auth import require_admin_key
from .schemas import (
    DocumentRecord,
    KnowledgeBaseStats,
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentDeleteResponse
)


@router.get(
    "/api/documents",
    response_model=List[DocumentRecord],
    summary="List Knowledge Base Documents",
    description="Returns all registered Government of India and administrator-uploaded documents with metadata, statuses, and chunk counts.",
    tags=["Knowledge Base Management"],
)
async def list_documents(
    category: Optional[str] = None,
    doc_status: Optional[str] = None,
    request: Request = None,
    admin_key: str = Depends(require_admin_key)
):
    """
    Returns full document registry from persistent SQLite store.
    """
    document_service = request.app.state.document_service
    return document_service.registry.list_documents(category=category, status=doc_status)


@router.get(
    "/api/documents/stats",
    response_model=KnowledgeBaseStats,
    summary="Knowledge Base Statistics",
    description="Returns aggregated statistics including total documents, pages, chunks, storage, and last update timestamp.",
    tags=["Knowledge Base Management"],
)
async def get_knowledge_base_stats(
    request: Request,
    admin_key: str = Depends(require_admin_key)
):
    """
    Returns aggregated document repository metrics.
    """
    document_service = request.app.state.document_service
    return document_service.registry.get_stats()


@router.get(
    "/api/documents/{document_id}",
    response_model=DocumentRecord,
    summary="Get Document Details",
    description="Returns detailed metadata, SHA-256 hash, and indexing information for a specific document.",
    responses={
        404: {"model": ErrorResponse, "description": "Document ID not found"}
    },
    tags=["Knowledge Base Management"],
)
async def get_document_details(
    document_id: str,
    request: Request,
    admin_key: str = Depends(require_admin_key)
):
    """
    Returns individual document details by ID.
    """
    document_service = request.app.state.document_service
    doc = document_service.registry.get_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found."
        )
    return doc


@router.get(
    "/api/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get Document Processing Status",
    description="Returns live processing status and extracted page/chunk counts for an ongoing or completed upload.",
    responses={
        404: {"model": ErrorResponse, "description": "Document ID not found"}
    },
    tags=["Knowledge Base Management"],
)
async def get_document_status(
    document_id: str,
    request: Request,
    admin_key: str = Depends(require_admin_key)
):
    """
    Returns live processing progress for a document.
    """
    document_service = request.app.state.document_service
    doc_status = document_service.get_document_status(document_id)
    if not doc_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found."
        )
    return doc_status


@router.post(
    "/api/documents/upload",
    response_model=DocumentUploadResponse,
    summary="Upload and Ingest Government PDF",
    description=(
        "Uploads a new Government of India legal PDF, extracts text, generates legal chunks with section references, "
        "computes BGE-small embeddings, incrementally updates the FAISS index, persists metadata, updates the SQLite registry, "
        "and hot-reloads the RAG pipeline for immediate retrieval."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format, size, or category"},
        409: {"model": ErrorResponse, "description": "Duplicate document (matching SHA-256 already exists)"},
        422: {"model": ErrorResponse, "description": "Validation failure or unextractable text"},
        500: {"model": ErrorResponse, "description": "Ingestion or vector indexing error"}
    },
    tags=["Knowledge Base Management"],
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Government legal PDF file (max 20 MB)"),
    category: str = Form(..., description="Legal category: RTI, CONSUMER, CIVIC, EDUCATION, TRANSPORT, ENVIRONMENT, OTHER"),
    title: Optional[str] = Form(None, description="Official title of the Act, Rules, or Gazette"),
    source: Optional[str] = Form(None, description="Issuing authority or publication source"),
    authority: Optional[str] = Form(None, description="Issuing government authority name"),
    source_url: Optional[str] = Form(None, description="Official government gazette/document URL"),
    admin_key: str = Depends(require_admin_key)
):
    document_service = request.app.state.document_service
    pipeline = request.app.state.pipeline

    try:
        file_bytes = await file.read()
        
        record = await run_in_threadpool(
            document_service.process_and_index_document,
            file_bytes=file_bytes,
            original_filename=file.filename,
            category=category,
            title=title,
            source=source,
            authority=authority,
            source_url=source_url,
            pipeline_instance=pipeline,
            background_tasks=background_tasks
        )

        return DocumentUploadResponse(
            document_id=record.document_id,
            original_file_name=record.original_file_name,
            status=record.status,
            uploaded_at=record.uploaded_at,
            message="Document successfully processed, indexed into FAISS, and available for queries.",
            page_count=record.page_count,
            chunk_count=record.chunk_count
        )

    except ValueError as exc:
        logger.warning("Upload validation failed: %s", exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    except FileExistsError as exc:
        logger.info("Duplicate document upload rejected: %s", exc)
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    except Exception as exc:
        logger.exception("Unexpected error during document ingestion: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Document processing failed: {str(exc)}"}
        )


@router.delete(
    "/api/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a Document",
    description=(
        "Safely removes a user-uploaded document from the SQLite registry, removes its chunks and vectors "
        "from the FAISS index, deletes the stored physical PDF, and hot-reloads the RAG pipeline. "
        "Deletion of baseline Government of India documents is strictly prohibited."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Baseline document deletion prohibited"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Deletion failed"}
    },
    tags=["Knowledge Base Management"],
)
async def delete_document(
    document_id: str,
    request: Request,
    admin_key: str = Depends(require_admin_key)
):
    """
    Executes safe, transactional document removal and FAISS index compaction.
    """
    document_service = request.app.state.document_service
    pipeline = request.app.state.pipeline

    try:
        result = document_service.delete_document(
            document_id=document_id,
            pipeline_instance=pipeline
        )
        return DocumentDeleteResponse(**result)

    except KeyError as exc:
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc).strip("'\"")}
        )

    except PermissionError as exc:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)}
        )

    except Exception as exc:
        logger.exception("Unexpected error deleting document %s: %s", document_id, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to delete document: {str(exc)}"}
        )


