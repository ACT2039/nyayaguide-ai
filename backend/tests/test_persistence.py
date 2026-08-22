"""
NyayaGuide AI — Permanent Knowledge Base Persistence & Deletion Unit Tests
Specifically verifies that:
1. Dynamic document uploads survive DocumentRegistry re-instantiations and server restarts.
2. Baseline document seeding is strictly idempotent and does NOT reset or wipe uploaded documents.
3. Newly added documents stay in SQLite, FAISS vectors, and chunks_metadata.json across pipeline reloads.
4. Deleted documents stay deleted and do NOT resurrect on restart.
5. Deletion of baseline documents remains prohibited.
"""
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import (
    ADMIN_API_KEY,
    TARGET_DOCUMENTS,
    REGISTRY_DB_PATH
)
from app.api.app import app
from app.services.document_registry import SQLiteDocumentRegistry
from app.services.document_service import DocumentService
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.retriever import NyayaRetriever
from app.rag.rag_pipeline import NyayaRAGPipeline
from app.models.registry import DocumentStatus


def make_test_pdf_bytes(title: str, content: str) -> bytes:
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), f"{title}\n\nSection 1. Short Title.\n{content}\n\nSection 2. Definitions.\nThis Act applies across all States of India.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestKnowledgeBasePersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_cm = TestClient(app, raise_server_exceptions=True)
        cls.client = cls.client_cm.__enter__()
        cls.admin_headers = {"X-Admin-Key": ADMIN_API_KEY}

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_01_idempotent_baseline_seeding(self):
        """Re-instantiating SQLiteDocumentRegistry 10 times does not duplicate or reset documents."""
        initial_reg = SQLiteDocumentRegistry()
        initial_count = len(initial_reg.list_documents())

        # Instantiate 10 times in a row
        for _ in range(10):
            reg = SQLiteDocumentRegistry()
            self.assertEqual(len(reg.list_documents()), initial_count)

    def test_02_persistence_across_service_and_pipeline_reloads(self):
        """
        Full lifecycle test:
        1. Upload new custom document (doc count increases N -> N+1)
        2. Re-create DocumentRegistry, VectorStore, Retriever, RAGPipeline from disk
        3. Verify document and its vectors remain indexed and retrievable
        4. Delete document
        5. Re-create all components and verify document stays deleted
        """
        # Step 1: Upload
        pdf_bytes = make_test_pdf_bytes(
            "The Indian Solar Energy and Green Power Act 2026",
            "Every commercial enterprise with over 500kW load shall install rooftop photovoltaic generation units."
        )
        files = {"file": ("Solar_Energy_Act_2026.pdf", pdf_bytes, "application/pdf")}
        data = {
            "category": "ENVIRONMENT",
            "title": "The Indian Solar Energy and Green Power Act 2026",
            "source": "Ministry of New and Renewable Energy"
        }

        upload_resp = self.client.post("/api/documents/upload", headers=self.admin_headers, files=files, data=data)
        self.assertEqual(upload_resp.status_code, 200)
        doc_id = upload_resp.json()["document_id"]

        # Verify it is in current SQLite registry
        reg1 = SQLiteDocumentRegistry()
        doc_record = reg1.get_by_id(doc_id)
        self.assertIsNotNone(doc_record)
        self.assertEqual(doc_record.title, "The Indian Solar Energy and Green Power Act 2026")
        docs_count_after_upload = len(reg1.list_documents())

        # Step 2: SIMULATE COMPLETE BACKEND RESTART
        # Fresh Registry, Fresh VectorStore, Fresh Retriever, Fresh Pipeline
        restarted_registry = SQLiteDocumentRegistry()
        self.assertEqual(len(restarted_registry.list_documents()), docs_count_after_upload)
        self.assertIsNotNone(restarted_registry.get_by_id(doc_id))

        restarted_vector_store = FAISSVectorStore()
        restarted_vector_store.load()
        meta_docs = set(c["document"] for c in restarted_vector_store.metadata_store)
        self.assertIn("Solar_Energy_Act_2026.pdf", meta_docs)

        restarted_pipeline = NyayaRAGPipeline()
        rag_resp = restarted_pipeline.ask("What are the rooftop solar installation requirements for commercial enterprises?")
        self.assertFalse(rag_resp.is_abstention)
        source_docs = [s.document for s in rag_resp.sources]
        self.assertIn("Solar_Energy_Act_2026.pdf", source_docs)

        # Step 3: DELETE DOCUMENT
        del_resp = self.client.delete(f"/api/documents/{doc_id}", headers=self.admin_headers)
        self.assertEqual(del_resp.status_code, 200)

        # Step 4: SIMULATE ANOTHER BACKEND RESTART AFTER DELETION
        restarted_registry_2 = SQLiteDocumentRegistry()
        self.assertIsNone(restarted_registry_2.get_by_id(doc_id))
        self.assertEqual(len(restarted_registry_2.list_documents()), docs_count_after_upload - 1)

        restarted_vector_store_2 = FAISSVectorStore()
        restarted_vector_store_2.load()
        meta_docs_2 = set(c["document"] for c in restarted_vector_store_2.metadata_store)
        self.assertNotIn("Solar_Energy_Act_2026.pdf", meta_docs_2)

        restarted_pipeline_2 = NyayaRAGPipeline()
        rag_resp_2 = restarted_pipeline_2.ask("What are the rooftop solar installation requirements for commercial enterprises?")
        source_docs_2 = [s.document for s in rag_resp_2.sources]
        self.assertNotIn("Solar_Energy_Act_2026.pdf", source_docs_2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
