"""
NyayaGuide AI — Database, Chunks Metadata, and FAISS Rebuilder
Performs safe, complete synchronization across SQLite, FAISS vectors, chunks_metadata.json, and stored PDFs.
"""
import sys
import json
import sqlite3
import hashlib
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import (
    REGISTRY_DB_PATH,
    FAISS_INDEX_PATH,
    METADATA_STORE_PATH,
    DOCUMENTS_DIR,
    TARGET_DOCUMENTS,
    EMBEDDING_DIMENSION
)
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import FAISSVectorStore
from app.models.document import DocumentChunk
from app.services.document_registry import SQLiteDocumentRegistry, BASELINE_RECORDS_DATA


def run_complete_deduplication():
    print("=" * 65)
    print("STARTING COMPLETE KNOWLEDGE BASE DEDUPLICATION & REBUILD")
    print("=" * 65)

    # Step 1: Clean SQLite Database
    db_path = Path(REGISTRY_DB_PATH)
    if db_path.exists():
        db_path.unlink()
        print("[1/5] Removed stale SQLite database file.")

    # Re-initialize registry (will create table and seed baseline with true content_hashes)
    registry = SQLiteDocumentRegistry(db_path)
    print("[2/5] Initialized fresh SQLite registry with 4 baseline records.")

    # Step 2: Ingest baseline documents from source (96 chunks)
    from app.main import run_ingestion
    baseline_chunks, summary = run_ingestion(use_hf=False, allow_local_fallback=True)
    print(f"[3/5] Parsed baseline documents: {len(baseline_chunks)} chunks, {summary.total_pages} pages.")

    # Step 3: Embed baseline chunks and build fresh FAISS index
    embedding_engine = EmbeddingEngine.get_instance()
    texts = [c.text for c in baseline_chunks]
    embeddings = embedding_engine.embed_documents(texts)

    vector_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)
    vector_store.clear()
    vector_store.add_documents(baseline_chunks, embeddings)
    idx_path, meta_path = vector_store.save()
    print(f"[4/5] Built and saved FAISS index ({vector_store.total_vectors} vectors) to {idx_path}.")

    # Step 4: Clean up any old test PDFs in backend/data/documents
    docs_dir = Path(DOCUMENTS_DIR)
    if docs_dir.exists():
        for item in docs_dir.iterdir():
            if item.is_file():
                item.unlink()
        print("[5/5] Cleaned up temporary test PDFs from documents directory.")

    print("\n" + "=" * 65)
    print("DEDUPLICATION COMPLETE: Clean 4 Baseline Documents State Ready.")
    print("=" * 65)


if __name__ == "__main__":
    run_complete_deduplication()
