"""
NyayaGuide AI — Document Registry & Knowledge Base Models
Pydantic schemas representing persistent document records, statuses, and repository statistics.
"""
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Lifecycle status of a document within the knowledge base registry."""
    UPLOADING = "UPLOADING"
    PROCESSING = "PROCESSING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


class DocumentRecord(BaseModel):
    """Represents a persistent record of a document in the SQLite registry."""
    document_id: str = Field(..., description="Unique document ID, e.g. DOC-20260822-0001")
    original_file_name: str = Field(..., description="Original name of the uploaded PDF file")
    stored_file_name: str = Field(..., description="Sanitized stored filename on the server")
    category: str = Field(..., description="Legal category, e.g. RTI, CONSUMER, CIVIC")
    title: str = Field(..., description="Official title of the document or bare act")
    source: str = Field(default="Government of India", description="Issuing authority or publication source")
    authority: str = Field(default="Government of India", description="Issuing authority name")
    source_url: Optional[str] = Field(default=None, description="Official online source URL if available")
    file_size_bytes: int = Field(default=0, description="Size of the PDF in bytes")
    page_count: int = Field(default=0, description="Total number of pages extracted")
    chunk_count: int = Field(default=0, description="Total number of RAG chunks generated")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADING, description="Current processing status")
    uploaded_at: str = Field(..., description="ISO 8601 formatted upload timestamp or 'Baseline'")
    indexed_at: Optional[str] = Field(default=None, description="ISO 8601 formatted index completion timestamp")
    content_hash: str = Field(..., description="SHA-256 hex digest of file contents for deduplication")
    version: int = Field(default=1, description="Document revision version")
    error_message: Optional[str] = Field(default=None, description="Safe error message if status is FAILED")
    is_baseline: bool = Field(default=False, description="True for the 4 initial baseline Government of India documents")


class KnowledgeBaseStats(BaseModel):
    """Aggregated statistics for the entire NyayaGuide AI knowledge base."""
    total_documents: int = Field(..., description="Total number of indexed and active documents")
    total_pages: int = Field(..., description="Total cumulative pages across all indexed documents")
    total_chunks: int = Field(..., description="Total cumulative RAG chunks in the FAISS vector index")
    total_file_size_bytes: int = Field(..., description="Total storage footprint of all documents in bytes")
    last_updated: str = Field(..., description="ISO timestamp of the most recent indexing action")


class DocumentUploadResponse(BaseModel):
    """Structured response after initiating a document upload."""
    document_id: str = Field(..., description="Assigned unique document identifier")
    original_file_name: str = Field(..., description="Original filename")
    status: DocumentStatus = Field(..., description="Current status of the document")
    uploaded_at: str = Field(..., description="Upload timestamp")
    message: str = Field(..., description="Human-readable success summary")
    page_count: Optional[int] = Field(default=None, description="Extracted page count if available")
    chunk_count: Optional[int] = Field(default=None, description="Generated chunk count if available")


class DocumentStatusResponse(BaseModel):
    """Status polling response for a specific document."""
    document_id: str = Field(..., description="Document identifier")
    original_file_name: str = Field(..., description="Original filename")
    status: DocumentStatus = Field(..., description="Current processing status")
    progress_stage: str = Field(..., description="Descriptive current stage (e.g. Extracting, Embedding, Completed)")
    page_count: int = Field(default=0, description="Pages parsed so far")
    chunk_count: int = Field(default=0, description="Chunks indexed so far")
    error_message: Optional[str] = Field(default=None, description="Detailed error description if status is FAILED")
    indexed_at: Optional[str] = Field(default=None, description="ISO timestamp when indexing completed")


class DocumentDeleteResponse(BaseModel):
    """Structured response after deleting a document from the knowledge base."""
    document_id: str = Field(..., description="Deleted document identifier")
    original_file_name: str = Field(..., description="Deleted filename")
    removed_chunks: int = Field(..., description="Number of vector chunks removed from FAISS")
    message: str = Field(..., description="Deletion confirmation message")
