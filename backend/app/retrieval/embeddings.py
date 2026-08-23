"""
NyayaGuide AI — Embedding Engine
Model: BAAI/bge-small-en-v1.5
Strategy: L2-normalized dense vector embeddings for exact cosine similarity search.
"""
import os
import gc
import ctypes
import threading
from typing import List, Optional, Any
import numpy as np

from ..config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION


def _trim_memory():
    """Forces Linux glibc memory allocator to release unmapped heap memory arenas back to the OS kernel."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class EmbeddingEngine:
    """
    Modular embedding engine for NyayaGuide AI.
    Loads BAAI/bge-small-en-v1.5 once and caches the model for reuse.
    
    Similarity Strategy:
    All document and query embeddings are normalized with L2 norm (||v||_2 = 1).
    When vectors are L2-normalized, their Inner Product (Dot Product) is mathematically
    identical to Cosine Similarity:
        cos(u, v) = (u . v) / (||u|| * ||v||) = u . v
    This allows FAISS IndexFlatIP (Inner Product) to compute exact cosine similarities.
    """

    _instance: Optional["EmbeddingEngine"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, query_instruction: bool = True):
        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSION
        self.query_instruction = query_instruction
        self._model: Optional[Any] = None

    @classmethod
    def get_instance(cls, model_name: str = EMBEDDING_MODEL_NAME) -> "EmbeddingEngine":
        """Get or create singleton instance of EmbeddingEngine to avoid repeated model loading."""
        if cls._instance is None or cls._instance.model_name != model_name:
            with cls._lock:
                if cls._instance is None or cls._instance.model_name != model_name:
                    cls._instance = cls(model_name=model_name)
        return cls._instance

    @property
    def model(self) -> Any:
        """Lazy load and cache the SentenceTransformer model on demand with CPU memory optimizations."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # Enforce strict single-threaded CPU memory environment variables
                    os.environ["OMP_NUM_THREADS"] = "1"
                    os.environ["MKL_NUM_THREADS"] = "1"
                    os.environ["OPENBLAS_NUM_THREADS"] = "1"
                    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
                    os.environ["NUMEXPR_NUM_THREADS"] = "1"
                    os.environ["TOKENIZERS_PARALLELISM"] = "false"

                    import torch
                    try:
                        torch.set_num_threads(1)
                        torch.set_num_interop_threads(1)
                    except Exception:
                        pass
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(
                        self.model_name,
                        device="cpu",
                        model_kwargs={"low_cpu_mem_usage": True}
                    )
                    _trim_memory()
        return self._model

    def embed_documents(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Generates L2-normalized float32 embeddings for a list of document chunks.
        Uses small batch size (4) and explicit glibc memory trimming for low-RAM (512 MiB) execution.
        Returns: np.ndarray of shape (len(texts), dimension), dtype float32
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        import torch

        with torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        _trim_memory()
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates L2-normalized float32 embedding for a user search query.
        Applies standard BGE query instruction if enabled for superior retrieval relevance.
        Returns: np.ndarray of shape (1, dimension), dtype float32
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        query_text = query.strip()
        if self.query_instruction and "bge" in self.model_name.lower():
            if not query_text.startswith("Represent this sentence"):
                query_text = f"Represent this sentence for searching relevant passages: {query_text}"

        import torch
        with torch.inference_mode():
            embedding = self.model.encode(
                [query_text],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        _trim_memory()
        return embedding.astype(np.float32)
