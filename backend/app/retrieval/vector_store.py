"""
NyayaGuide AI — FAISS Vector Store & Metadata Mapper
Manages FAISS IndexFlatIP index and synchronous chunk metadata mapping.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

from ..config import (
    EMBEDDING_DIMENSION,
    FAISS_INDEX_PATH,
    METADATA_STORE_PATH,
    VECTOR_STORE_DIR
)
from ..models.document import DocumentChunk


class FAISSVectorStore:
    """
    Manages a FAISS IndexFlatIP (Inner Product / Cosine Similarity) vector store
    along with a synchronous JSON-based metadata mapping.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store: List[Dict[str, Any]] = []

    @property
    def total_vectors(self) -> int:
        """Return total number of vectors in the FAISS index."""
        return self.index.ntotal if self.index is not None else 0

    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """
        Adds normalized embeddings and synchronous chunk metadata to the vector store.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided but got {len(embeddings)} embeddings."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        # Ensure float32 and contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Add vectors to FAISS index
        self.index.add(vectors)

        # Add corresponding metadata records
        for chunk in chunks:
            self.metadata_store.append({
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "category": chunk.category,
                "page": chunk.page,
                "legal_reference": chunk.legal_reference,
                "title": chunk.title,
                "source": chunk.source,
                "source_url": chunk.source_url,
                "text": chunk.text,
                "word_count": chunk.word_count,
                "chunk_index": chunk.chunk_index
            })

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Performs cosine similarity search against the FAISS index.
        Returns: list of (similarity_score, metadata_dict)
        """
        if self.total_vectors == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

        # Request more candidates if filtering by category
        k_search = min(self.total_vectors, top_k * 3 if category_filter else top_k)
        scores, indices = self.index.search(query_vector, k_search)

        results: List[Tuple[float, Dict[str, Any]]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata_store):
                continue
            
            meta = self.metadata_store[idx]
            if category_filter and meta.get("category", "").upper() != category_filter.upper():
                continue

            results.append((float(score), meta))
            if len(results) >= top_k:
                break

        return results

    def save(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """
        Persist FAISS index binary and metadata JSON to disk.
        """
        idx_path = Path(index_path or FAISS_INDEX_PATH)
        meta_path = Path(metadata_path or METADATA_STORE_PATH)

        idx_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS binary index
        faiss.write_index(self.index, str(idx_path))

        # Save metadata mapping as structured UTF-8 JSON
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "total_vectors": self.total_vectors,
                "metadata": self.metadata_store
            }, f, indent=2, ensure_ascii=False)

        return idx_path, meta_path

    def load(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ) -> bool:
        """
        Load persisted FAISS index and metadata mapping from disk.
        """
        idx_path = Path(index_path or FAISS_INDEX_PATH)
        meta_path = Path(metadata_path or METADATA_STORE_PATH)

        if not idx_path.exists() or not meta_path.exists():
            return False

        # Load FAISS index
        self.index = faiss.read_index(str(idx_path))
        self.dimension = self.index.d

        # Load metadata mapping
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.metadata_store = data.get("metadata", [])

        return True

    def remove_documents(self, filename: str) -> int:
        """
        Removes all chunks and corresponding vector embeddings for a specified filename.
        Reconstructs the FAISS IndexFlatIP safely with the remaining canonical vectors.
        Returns the number of removed chunks/vectors.
        """
        if self.total_vectors == 0:
            return 0

        # Identify vector indices to keep and those to remove
        keep_indices: List[int] = []
        removed_count = 0

        for i, meta in enumerate(self.metadata_store):
            if meta.get("document", "").lower() == filename.lower():
                removed_count += 1
            else:
                keep_indices.append(i)

        if removed_count == 0:
            return 0

        # Reconstruct remaining vectors from current FAISS index
        new_index = faiss.IndexFlatIP(self.dimension)
        new_metadata: List[Dict[str, Any]] = []

        if keep_indices:
            # Reconstruct raw vectors by index from IndexFlatIP
            remaining_vectors = np.zeros((len(keep_indices), self.dimension), dtype=np.float32)
            for new_i, old_i in enumerate(keep_indices):
                remaining_vectors[new_i] = self.index.reconstruct(int(old_i))
                new_metadata.append(self.metadata_store[old_i])

            new_index.add(remaining_vectors)

        # Atomically swap in-memory index and metadata
        self.index = new_index
        self.metadata_store = new_metadata

        return removed_count

    def reload_from_disk(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ) -> bool:
        """
        Reload the FAISS index and metadata store from disk in-place.
        """
        return self.load(index_path=index_path, metadata_path=metadata_path)

    def clear(self) -> None:
        """Reset the FAISS index and metadata store."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = []
