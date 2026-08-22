"""
Unit Tests for Embedding Engine (BAAI/bge-small-en-v1.5)
"""
import sys
import unittest
from pathlib import Path
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
from app.retrieval.embeddings import EmbeddingEngine


class TestEmbeddingEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = EmbeddingEngine.get_instance()

    def test_model_loading_and_dimension(self):
        self.assertIsNotNone(self.engine.model)
        self.assertEqual(self.engine.dimension, EMBEDDING_DIMENSION)
        self.assertEqual(self.engine.dimension, 384)

    def test_document_embedding_shape_and_dtype(self):
        texts = [
            "Section 6. Request for obtaining information under RTI Act.",
            "Section 35. Manner in which complaint shall be made to District Commission."
        ]
        embeddings = self.engine.embed_documents(texts)
        self.assertEqual(embeddings.shape, (2, 384))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_l2_normalization(self):
        texts = ["Consumer rights under the Consumer Protection Act, 2019."]
        vec = self.engine.embed_documents(texts)[0]
        norm = float(np.linalg.norm(vec))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_query_embedding_normalization(self):
        query = "How to file an RTI application?"
        query_vec = self.engine.embed_query(query)
        self.assertEqual(query_vec.shape, (1, 384))
        norm = float(np.linalg.norm(query_vec[0]))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_empty_query_handling(self):
        with self.assertRaises(ValueError):
            self.engine.embed_query("")


if __name__ == "__main__":
    unittest.main(verbosity=2)
