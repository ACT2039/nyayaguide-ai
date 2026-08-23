"""
NyayaGuide AI — Document Management Service
Orchestrates dynamic document upload, validation, duplicate prevention, parsing, chunking,
embedding generation, transactional incremental FAISS indexing, metadata persistence, and RAG hot reload.
"""
import os
import re
import shutil
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger("nyayaguide.document_service")

from ..config import (
    DOCUMENTS_DIR,
    MAX_UPLOAD_SIZE_BYTES,
    ALLOWED_CATEGORIES,
    EMBEDDING_DIMENSION
)
from ..models.registry import DocumentRecord, DocumentStatus, DocumentUploadResponse, DocumentStatusResponse
from ..models.document import DocumentChunk, ParsedDocument
from ..ingestion.pdf_parser import PDFParser
from ..ingestion.chunker import LegalDocumentChunker
from ..ingestion.metadata import MetadataRegistry
from ..retrieval.embeddings import EmbeddingEngine
from ..retrieval.vector_store import FAISSVectorStore
from .document_registry import SQLiteDocumentRegistry
from .hf_storage_service import HuggingFaceStorageService


class DocumentService:
    """
    Unified service managing the document lifecycle and dynamic incremental RAG indexing.
    """

    def __init__(
        self,
        registry: Optional[SQLiteDocumentRegistry] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        storage_service: Optional[HuggingFaceStorageService] = None
    ):
        self.registry = registry or SQLiteDocumentRegistry()
        self.embedding_engine = embedding_engine or EmbeddingEngine.get_instance()
        self.vector_store = vector_store or FAISSVectorStore(dimension=EMBEDDING_DIMENSION)
        self.storage_service = storage_service or HuggingFaceStorageService()
        self.documents_dir = Path(DOCUMENTS_DIR)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        # Ingestion tools
        self.metadata_registry = MetadataRegistry()
        self.pdf_parser = PDFParser(metadata_registry=self.metadata_registry)
        self.chunker = LegalDocumentChunker()

    def validate_file(self, filename: str, file_bytes: bytes) -> None:
        """
        Validates file extension, magic bytes, size, and sanitization.
        Raises ValueError with clear, safe error message on invalid input.
        """
        if not filename or not filename.strip():
            raise ValueError("File name cannot be empty.")

        # Check extension
        ext = Path(filename).suffix.lower()
        if ext != ".pdf":
            raise ValueError(f"Invalid file type '{ext}'. Only PDF documents (.pdf) are permitted.")

        # Check maximum file size
        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            raise ValueError(f"File size exceeds maximum permitted limit of {max_mb} MB.")

        if len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        # Validate PDF magic bytes (%PDF-)
        if not file_bytes.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF format: file lacks official PDF header signature.")

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitizes the original filename to prevent path traversal and shell injection.
        """
        basename = os.path.basename(filename).strip()
        # Remove any path traversal patterns
        basename = re.sub(r'[\/\\:\*\?"<>\|]', '_', basename)
        # Avoid hidden filenames or consecutive dots
        basename = re.sub(r'^\.+', '', basename)
        if not basename.endswith('.pdf'):
            basename = f"{basename}.pdf"
        return basename or "uploaded_document.pdf"

    def compute_sha256(self, file_bytes: bytes) -> str:
        """Computes the SHA-256 hexadecimal digest of the file bytes."""
        return hashlib.sha256(file_bytes).hexdigest()

    def initiate_document_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
        category: str,
        title: Optional[str] = None,
        source: Optional[str] = None,
        authority: Optional[str] = None,
        source_url: Optional[str] = None,
        background_tasks: Optional[Any] = None,
        pipeline_instance: Optional[Any] = None
    ) -> DocumentRecord:
        """
        Fast synchronous validation, SHA-256 duplicate check, and disk storage.
        Creates initial SQLite record with status PROCESSING, queues background ingestion,
        and returns immediately (< 0.2s).
        """
        # Step 1: Validation
        self.validate_file(original_filename, file_bytes)

        category_clean = category.strip().upper()
        if category_clean not in ALLOWED_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Allowed categories: {', '.join(ALLOWED_CATEGORIES)}")

        # Step 2: SHA-256 Duplicate Check
        content_hash = self.compute_sha256(file_bytes)
        existing_doc = self.registry.get_by_hash(content_hash)
        if existing_doc:
            raise FileExistsError(
                f"Document already exists in the knowledge base (Document ID: {existing_doc.document_id}, Title: '{existing_doc.title}')."
            )

        # Step 3: Secure Disk Storage
        safe_name = self.sanitize_filename(original_filename)
        doc_id = self.registry.generate_document_id()
        stored_filename = f"{doc_id}_{safe_name}"
        stored_path = self.documents_dir / stored_filename

        try:
            with open(stored_path, "wb") as f:
                f.write(file_bytes)
        except Exception as e:
            raise IOError(f"Failed to write file to disk: {e}")

        # Construct initial title and metadata
        doc_title = title.strip() if title and title.strip() else safe_name.replace(".pdf", "").replace("_", " ")
        doc_source = source.strip() if source and source.strip() else "Government of India"
        doc_authority = authority.strip() if authority and authority.strip() else doc_source
        uploaded_at = datetime.now(timezone.utc).isoformat()

        record = DocumentRecord(
            document_id=doc_id,
            original_file_name=safe_name,
            stored_file_name=stored_filename,
            category=category_clean,
            title=doc_title,
            source=doc_source,
            authority=doc_authority,
            source_url=source_url.strip() if source_url and source_url.strip() else None,
            file_size_bytes=len(file_bytes),
            page_count=0,
            chunk_count=0,
            status=DocumentStatus.PROCESSING,
            uploaded_at=uploaded_at,
            indexed_at=None,
            content_hash=content_hash,
            version=1,
            error_message=None,
            is_baseline=False
        )

        # Register in SQLite
        self.registry.create_document(record)

        # Step 4: Queue or execute background ingestion
        if background_tasks is not None and hasattr(background_tasks, "add_task"):
            background_tasks.add_task(
                self.run_background_ingestion,
                doc_id=doc_id,
                stored_path=stored_path,
                original_filename=safe_name,
                category_clean=category_clean,
                doc_title=doc_title,
                doc_source=doc_source,
                pipeline_instance=pipeline_instance
            )
        else:
            # Synchronous execution fallback when background_tasks is not provided (e.g. unit tests)
            self.run_background_ingestion(
                doc_id=doc_id,
                stored_path=stored_path,
                original_filename=safe_name,
                category_clean=category_clean,
                doc_title=doc_title,
                doc_source=doc_source,
                pipeline_instance=pipeline_instance
            )
            # Re-fetch updated record to return status for synchronous test callers
            updated = self.registry.get_by_id(doc_id)
            if updated:
                if updated.status == DocumentStatus.FAILED:
                    raise RuntimeError(f"Document ingestion failed: {updated.error_message}")
                return updated

        return record

    def run_background_ingestion(
        self,
        doc_id: str,
        stored_path: Path,
        original_filename: str,
        category_clean: str,
        doc_title: str,
        doc_source: str,
        pipeline_instance: Optional[Any] = None
    ) -> None:
        """
        Executes heavy background PDF parsing, legal chunking, BGE embedding, FAISS indexing,
        RAG pipeline hot-reloading, and Hugging Face remote snapshot synchronization.
        Safely catches exceptions and sets DocumentStatus.FAILED on failure.
        """
        try:
            logger.info("Starting background ingestion for document '%s' (%s)...", doc_id, original_filename)
            self.registry.update_status(doc_id, DocumentStatus.PROCESSING)

            # Step 1: Parse PDF
            parsed_doc = self.pdf_parser.parse_pdf(stored_path, filename=original_filename)
            parsed_doc.category = category_clean
            parsed_doc.title = doc_title
            parsed_doc.source = doc_source

            if parsed_doc.total_pages == 0:
                raise ValueError("PDF contains 0 pages or could not be parsed.")

            empty_pages_count = sum(1 for p in parsed_doc.pages if p.is_empty)
            if empty_pages_count == parsed_doc.total_pages:
                raise ValueError("PDF appears to be scanned or contains no extractable text.")

            # Step 2: Legal Chunking
            chunks = self.chunker.chunk_document(parsed_doc)
            if not chunks:
                raise ValueError("No valid legal chunks could be extracted from this document.")

            self.registry.update_status(
                doc_id,
                DocumentStatus.EMBEDDING,
                page_count=parsed_doc.total_pages,
                chunk_count=len(chunks)
            )

            # Explicit memory cleanup before heavy BGE embedding
            total_pages_count = parsed_doc.total_pages
            del parsed_doc
            import gc
            gc.collect()

            # Step 3: BGE Embeddings (batch_size=2 for 512 MiB RAM limit)
            texts = [c.text for c in chunks]
            embeddings = self.embedding_engine.embed_documents(texts, batch_size=2)

            if len(embeddings) != len(chunks):
                raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)} chunks.")

            # Step 4: Transactional FAISS Incremental Indexing
            self.registry.update_status(doc_id, DocumentStatus.INDEXING)
            self.vector_store.load()
            self.vector_store.add_documents(chunks, embeddings)
            self.vector_store.save()

            # Step 5: Update SQLite status to INDEXED
            indexed_at = datetime.now(timezone.utc).isoformat()
            self.registry.update_status(
                doc_id,
                DocumentStatus.INDEXED,
                page_count=total_pages_count,
                chunk_count=len(chunks),
                indexed_at=indexed_at
            )

            # Step 6: Hot-reload in-memory RAG pipeline
            if pipeline_instance:
                pipeline_instance.reload_index()

            logger.info("Background ingestion completed successfully for document '%s'. Status: INDEXED.", doc_id)

            # Step 7: Synchronize Knowledge Base snapshot to Hugging Face
            if self.storage_service.is_available():
                try:
                    self.storage_service.sync_snapshot(new_pdf_path=stored_path)
                except Exception as hf_err:
                    logger.warning("Hugging Face snapshot sync notice for '%s': %s", doc_id, hf_err)

        except Exception as e:
            logger.error("Background ingestion failed for document '%s': %s", doc_id, e)
            safe_error = str(e)
            self.registry.update_status(
                doc_id,
                DocumentStatus.FAILED,
                error_message=safe_error
            )
            try:
                self.vector_store.load()
            except Exception:
                pass

    def process_and_index_document(
        self,
        file_bytes: bytes,
        original_filename: str,
        category: str,
        title: Optional[str] = None,
        source: Optional[str] = None,
        authority: Optional[str] = None,
        source_url: Optional[str] = None,
        pipeline_instance: Optional[Any] = None,
        background_tasks: Optional[Any] = None
    ) -> DocumentRecord:
        """
        Backward-compatible wrapper method around initiate_document_upload.
        """
        return self.initiate_document_upload(
            file_bytes=file_bytes,
            original_filename=original_filename,
            category=category,
            title=title,
            source=source,
            authority=authority,
            source_url=source_url,
            background_tasks=background_tasks,
            pipeline_instance=pipeline_instance
        )

    def get_document_status(self, document_id: str) -> Optional[DocumentStatusResponse]:
        """Returns the processing and indexing status of a document."""
        doc = self.registry.get_by_id(document_id)
        if not doc:
            return None

        stage_map = {
            DocumentStatus.UPLOADING: "Uploading PDF",
            DocumentStatus.PROCESSING: "Extracting & Cleaning Text",
            DocumentStatus.EMBEDDING: "Generating BGE Embeddings",
            DocumentStatus.INDEXING: "Indexing Vectors in FAISS",
            DocumentStatus.INDEXED: "Completed & Indexed",
            DocumentStatus.FAILED: "Processing Failed"
        }

        return DocumentStatusResponse(
            document_id=doc.document_id,
            original_file_name=doc.original_file_name,
            status=doc.status,
            progress_stage=stage_map.get(doc.status, "Processing"),
            page_count=doc.page_count,
            chunk_count=doc.chunk_count,
            error_message=doc.error_message,
            indexed_at=doc.indexed_at
        )

    def delete_document(
        self,
        document_id: str,
        pipeline_instance: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Safely and transactionally deletes a user-uploaded document:
        1. Validates document ID exists
        2. Prohibits deletion of baseline Government documents
        3. Removes document's chunks and vectors from FAISS & chunks_metadata.json
        4. Saves updated FAISS index and metadata to disk
        5. Removes physical PDF from documents storage
        6. Deletes record from SQLite registry
        7. Hot-reloads RAG pipeline in-memory
        """
        doc = self.registry.get_by_id(document_id)
        if not doc:
            raise KeyError(f"Document with ID '{document_id}' does not exist.")

        if doc.is_baseline:
            raise PermissionError(
                f"Document '{doc.original_file_name}' (ID: {doc.document_id}) is a protected baseline Government of India statute and cannot be deleted."
            )

        target_filename = doc.original_file_name

        try:
            # Step 1: Remove vectors and metadata from FAISS
            self.vector_store.load()
            removed_chunks = self.vector_store.remove_documents(target_filename)
            self.vector_store.save()

            # Step 2: Remove stored physical PDF file
            stored_file_path = self.documents_dir / doc.stored_file_name
            if stored_file_path.exists():
                stored_file_path.unlink()

            # Step 3: Remove record from SQLite registry
            self.registry.delete_document(document_id)

            # Step 3.5: Synchronize deletion to Hugging Face
            if self.storage_service.is_available():
                self.storage_service.sync_deletion(deleted_file_name=doc.stored_file_name)

            # Step 4: Hot-reload in-memory RAG pipeline if provided
            if pipeline_instance:
                pipeline_instance.reload_index()

            return {
                "document_id": document_id,
                "original_file_name": target_filename,
                "removed_chunks": removed_chunks,
                "message": f"Document '{target_filename}' was successfully removed from the knowledge base and FAISS vector index."
            }

        except Exception as e:
            # On failure, reload index from disk to restore state
            self.vector_store.load()
            raise RuntimeError(f"Failed to delete document '{document_id}': {str(e)}") from e

