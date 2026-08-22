"""
NyayaGuide AI — Semantic Retriever
Orchestrates embedding generation, FAISS indexing, and cosine semantic search.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from ..config import (
    DEFAULT_TOP_K,
    EMBEDDING_DIMENSION,
    FAISS_INDEX_PATH,
    METADATA_STORE_PATH,
    TARGET_DOCUMENTS
)
from ..models.document import DocumentChunk, RetrievalResult
from .embeddings import EmbeddingEngine
from .vector_store import FAISSVectorStore


class NyayaRetriever:
    """
    Unified retriever for NyayaGuide AI.
    Handles index building from parsed chunks and semantic similarity queries.
    """

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        vector_store: Optional[FAISSVectorStore] = None
    ):
        self.embedding_engine = embedding_engine or EmbeddingEngine.get_instance()
        self.vector_store = vector_store or FAISSVectorStore(dimension=EMBEDDING_DIMENSION)

    def build_index(
        self,
        chunks: Optional[List[DocumentChunk]] = None,
        save_to_disk: bool = True
    ) -> Dict[str, Any]:
        """
        Builds the FAISS vector index from document chunks:
        1. Ingests or receives chunks.
        2. Generates L2-normalized embeddings via BAAI/bge-small-en-v1.5.
        3. Populates FAISS IndexFlatIP.
        4. Saves index and metadata to disk.
        5. Returns execution statistics.
        """
        if chunks is None:
            from ..main import run_ingestion
            chunks, _ = run_ingestion()

        if not chunks:
            raise ValueError("No document chunks provided to build vector index.")

        # Extract text list for batch embedding
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_engine.embed_documents(texts)

        # Populate vector store
        self.vector_store.clear()
        self.vector_store.add_documents(chunks, embeddings)

        saved_index_path = None
        saved_meta_path = None
        if save_to_disk:
            saved_index_path, saved_meta_path = self.vector_store.save()

        unique_docs = len(set(c.document for c in chunks))
        stats = {
            "documents": unique_docs,
            "chunks": len(chunks),
            "embeddings_generated": len(embeddings),
            "embedding_dimension": self.vector_store.dimension,
            "faiss_vectors": self.vector_store.total_vectors,
            "index_path": str(saved_index_path) if saved_index_path else None,
            "metadata_path": str(saved_meta_path) if saved_meta_path else None
        }

        print("\n" + "=" * 65)
        print("FAISS VECTOR INDEX BUILD COMPLETE")
        print("=" * 65)
        print(f"Documents            : {stats['documents']}")
        print(f"Chunks               : {stats['chunks']}")
        print(f"Embeddings generated : {stats['embeddings_generated']}")
        print(f"Embedding dimension  : {stats['embedding_dimension']}")
        print(f"FAISS vectors        : {stats['faiss_vectors']}")
        if saved_index_path:
            idx_size = saved_index_path.stat().st_size
            meta_size = saved_meta_path.stat().st_size
            print(f"Storage Index File   : {saved_index_path.name} ({idx_size:,} bytes)")
            print(f"Storage Metadata File: {saved_meta_path.name} ({meta_size:,} bytes)")
        print("=" * 65)

        return stats

    def ensure_index_loaded(self) -> bool:
        """Loads index from disk if not currently populated in memory."""
        if self.vector_store.total_vectors > 0:
            return True

        if FAISS_INDEX_PATH.exists() and METADATA_STORE_PATH.exists():
            return self.vector_store.load(FAISS_INDEX_PATH, METADATA_STORE_PATH)

    def reload_index(self) -> bool:
        """Reloads the FAISS vector index and metadata store in-memory from disk."""
        if FAISS_INDEX_PATH.exists() and METADATA_STORE_PATH.exists():
            return self.vector_store.reload_from_disk(FAISS_INDEX_PATH, METADATA_STORE_PATH)
        return False

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieves the top-k most semantically relevant chunks for a user query.
        Returns: List of structured RetrievalResult objects.
        """
        self.ensure_index_loaded()

        query_vec = self.embedding_engine.embed_query(query)
        matches = self.vector_store.search(
            query_vector=query_vec,
            top_k=top_k,
            category_filter=category
        )

        results: List[RetrievalResult] = []
        for rank, (score, meta) in enumerate(matches, 1):
            results.append(RetrievalResult(
                rank=rank,
                score=round(score, 4),
                chunk_id=meta.get("chunk_id", ""),
                document=meta.get("document", ""),
                category=meta.get("category", ""),
                page=meta.get("page", 0),
                legal_reference=meta.get("legal_reference"),
                title=meta.get("title", ""),
                source=meta.get("source", ""),
                source_url=meta.get("source_url"),
                text=meta.get("text", "")
            ))

        return results


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    category: Optional[str] = None
) -> List[RetrievalResult]:
    """Convenience module-level retrieval function."""
    retriever = NyayaRetriever()
    return retriever.retrieve(query=query, top_k=top_k, category=category)
