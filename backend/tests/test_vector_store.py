"""
Unit Tests for FAISS Vector Store and Metadata Mapping
"""
import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.models.document import DocumentChunk
from app.retrieval.vector_store import FAISSVectorStore


class TestFAISSVectorStore(unittest.TestCase):
    def setUp(self):
        self.dimension = 384
        self.store = FAISSVectorStore(dimension=self.dimension)

    def test_add_and_search_documents(self):
        # Create mock chunks
        chunks = [
            DocumentChunk(
                chunk_id="RTI_p1_c1",
                document="RTI_Act_2005.pdf",
                category="RTI",
                page=1,
                legal_reference="Section 6",
                title="The Right to Information Act, 2005",
                source="India Code",
                source_url="https://cic.gov.in",
                text="Information request procedure under RTI Act.",
                word_count=6,
                chunk_index=0
            ),
            DocumentChunk(
                chunk_id="CPA_p1_c1",
                document="Consumer_Protection_Act_2019.pdf",
                category="CONSUMER",
                page=1,
                legal_reference="Section 35",
                title="The Consumer Protection Act, 2019",
                source="Department of Consumer Affairs",
                source_url="https://egazette.gov.in",
                text="Consumer complaint procedure before District Commission.",
                word_count=7,
                chunk_index=1
            )
        ]

        # Create normalized orthogonal mock vectors
        v1 = np.zeros((1, 384), dtype=np.float32)
        v1[0, 0] = 1.0
        v2 = np.zeros((1, 384), dtype=np.float32)
        v2[0, 1] = 1.0
        embeddings = np.vstack([v1, v2])

        self.store.add_documents(chunks, embeddings)
        self.assertEqual(self.store.total_vectors, 2)

        # Search for vector close to v1
        results = self.store.search(v1, top_k=2)
        self.assertEqual(len(results), 2)
        top_score, top_meta = results[0]
        self.assertAlmostEqual(top_score, 1.0, places=4)
        self.assertEqual(top_meta["chunk_id"], "RTI_p1_c1")
        self.assertEqual(top_meta["category"], "RTI")

    def test_category_filtering(self):
        chunks = [
            DocumentChunk(
                chunk_id="RTI_1", document="RTI_Act_2005.pdf", category="RTI", page=1,
                legal_reference="Section 1", title="RTI Act", source="India Code",
                source_url=None, text="RTI Text", word_count=2, chunk_index=0
            ),
            DocumentChunk(
                chunk_id="CPA_1", document="Consumer_Protection_Act_2019.pdf", category="CONSUMER", page=1,
                legal_reference="Section 1", title="CP Act", source="Gov of India",
                source_url=None, text="Consumer Text", word_count=2, chunk_index=1
            )
        ]
        embeddings = np.zeros((2, 384), dtype=np.float32)
        embeddings[0, 0] = 1.0
        embeddings[1, 0] = 0.99
        self.store.add_documents(chunks, embeddings)

        query = np.zeros((1, 384), dtype=np.float32)
        query[0, 0] = 1.0

        # Filter for CONSUMER only
        results = self.store.search(query, top_k=2, category_filter="CONSUMER")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1]["category"], "CONSUMER")

    def test_persistence_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            idx_file = Path(tmp_dir) / "test_index.bin"
            meta_file = Path(tmp_dir) / "test_meta.json"

            chunk = DocumentChunk(
                chunk_id="Test_1", document="RTI_Act_2005.pdf", category="RTI", page=5,
                legal_reference="Section 4", title="RTI Act", source="India Code",
                source_url=None, text="Test legal content verbatim.", word_count=4, chunk_index=0
            )
            vec = np.zeros((1, 384), dtype=np.float32)
            vec[0, 10] = 1.0

            self.store.add_documents([chunk], vec)
            self.store.save(idx_file, meta_file)

            # Load into new store instance
            new_store = FAISSVectorStore(dimension=384)
            success = new_store.load(idx_file, meta_file)
            self.assertTrue(success)
            self.assertEqual(new_store.total_vectors, 1)
            self.assertEqual(new_store.metadata_store[0]["legal_reference"], "Section 4")
            self.assertEqual(new_store.metadata_store[0]["text"], "Test legal content verbatim.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
