"""
NyayaGuide AI — Document Management & Ingestion Integration Tests
Verifies SQLite registry, baseline seeding, PDF upload, duplicate SHA-256 detection,
incremental FAISS vector count growth, RAG hot-reload, file validation, admin auth protection,
and safe transactional document deletion.
"""
import io
import sys
import unittest
import tempfile
import hashlib
from pathlib import Path
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import (
    ADMIN_API_KEY,
    MAX_UPLOAD_SIZE_BYTES,
    TARGET_DOCUMENTS
)
from app.api.app import app
from app.services.document_registry import SQLiteDocumentRegistry
from app.services.document_service import DocumentService
from app.models.registry import DocumentStatus


def create_minimal_pdf_bytes(title: str, content: str) -> bytes:
    """Helper to generate valid minimal PDF bytes for testing."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), f"{title}\n\nSection 1. Short Title.\n{content}\n\nSection 2. Definitions.\nThis Act applies across India.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestDocumentManagement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_cm = TestClient(app, raise_server_exceptions=True)
        cls.client = cls.client_cm.__enter__()
        cls.admin_headers = {"X-Admin-Key": ADMIN_API_KEY}

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_01_baseline_registry_seeding(self):
        """Verify the 4 baseline Government of India documents are registered exactly once."""
        resp = self.client.get("/api/documents", headers=self.admin_headers)
        self.assertEqual(resp.status_code, 200)
        docs = resp.json()
        
        # Check baseline docs
        baseline_docs = [d for d in docs if d["is_baseline"]]
        self.assertEqual(len(baseline_docs), 4)

        filenames = [d["original_file_name"] for d in baseline_docs]
        for baseline_doc in TARGET_DOCUMENTS:
            self.assertIn(baseline_doc, filenames)

    def test_02_get_stats_baseline(self):
        """Verify statistics correctly aggregate baseline documents."""
        resp = self.client.get("/api/documents/stats", headers=self.admin_headers)
        self.assertEqual(resp.status_code, 200)
        stats = resp.json()
        self.assertGreaterEqual(stats["total_documents"], 4)
        self.assertGreaterEqual(stats["total_pages"], 89)
        self.assertGreaterEqual(stats["total_chunks"], 96)
        self.assertIn("last_updated", stats)

    def test_03_admin_auth_protection(self):
        """Verify admin routes reject requests without or with invalid X-Admin-Key."""
        # Missing key
        resp1 = self.client.get("/api/documents")
        self.assertEqual(resp1.status_code, 401)
        self.assertIn("Admin authentication required", resp1.json()["detail"])

        # Invalid key
        resp2 = self.client.get("/api/documents", headers={"X-Admin-Key": "wrong_secret_key"})
        self.assertEqual(resp2.status_code, 401)
        self.assertIn("Invalid administrative API key", resp2.json()["detail"])

    def test_04_valid_pdf_upload_and_incremental_index(self):
        """Verify dynamic upload, incremental FAISS addition, and immediate searchability."""
        # Initial stats
        stats_before = self.client.get("/api/documents/stats", headers=self.admin_headers).json()
        initial_chunks = stats_before["total_chunks"]

        # Create a unique test legal PDF
        pdf_bytes = create_minimal_pdf_bytes(
            "The Indian Legal Metrology and Consumer Standard Rules 2026",
            "Any manufacturer or seller violating statutory packaging requirements shall be liable for civil penalties."
        )

        files = {
            "file": ("Legal_Metrology_Rules_2026.pdf", pdf_bytes, "application/pdf")
        }
        data = {
            "category": "CONSUMER",
            "title": "Legal Metrology Rules 2026",
            "source": "Ministry of Consumer Affairs",
            "authority": "Legal Metrology Division",
            "source_url": "https://consumeraffairs.nic.in"
        }

        resp = self.client.post("/api/documents/upload", headers=self.admin_headers, files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        resp_data = resp.json()
        self.assertEqual(resp_data["status"], "INDEXED")
        self.assertGreater(resp_data["chunk_count"], 0)
        doc_id = resp_data["document_id"]

        # Verify document appears in GET /api/documents/{doc_id}
        detail_resp = self.client.get(f"/api/documents/{doc_id}", headers=self.admin_headers)
        self.assertEqual(detail_resp.status_code, 200)
        doc_detail = detail_resp.json()
        self.assertEqual(doc_detail["title"], "Legal Metrology Rules 2026")
        self.assertEqual(doc_detail["category"], "CONSUMER")

        # Verify status endpoint
        status_resp = self.client.get(f"/api/documents/{doc_id}/status", headers=self.admin_headers)
        self.assertEqual(status_resp.status_code, 200)
        self.assertEqual(status_resp.json()["status"], "INDEXED")

        # Verify stats updated incrementally
        stats_after = self.client.get("/api/documents/stats", headers=self.admin_headers).json()
        self.assertGreater(stats_after["total_chunks"], initial_chunks)

        # Verify immediate searchability in /api/ask
        ask_resp = self.client.post(
            "/api/ask",
            json={"question": "What are the penalties under Legal Metrology packaging requirements?"}
        )
        self.assertEqual(ask_resp.status_code, 200)
        ask_data = ask_resp.json()
        self.assertFalse(ask_data["is_abstention"])
        source_docs = [s["document"] for s in ask_data["sources"]]
        self.assertIn("Legal_Metrology_Rules_2026.pdf", source_docs)

    def test_05_duplicate_sha256_rejection(self):
        """Verify that uploading an identical file returns HTTP 409 Conflict."""
        pdf_bytes = create_minimal_pdf_bytes(
            "Duplicate Prevention Act 2026",
            "This document tests SHA-256 duplicate rejection."
        )

        files = {
            "file": ("Duplicate_Test_Doc.pdf", pdf_bytes, "application/pdf")
        }
        data = {"category": "CIVIC", "title": "Duplicate Doc"}

        # First upload
        resp1 = self.client.post("/api/documents/upload", headers=self.admin_headers, files=files, data=data)
        self.assertEqual(resp1.status_code, 200)

        # Duplicate upload
        files_dup = {
            "file": ("Different_Name_Same_Content.pdf", pdf_bytes, "application/pdf")
        }
        resp2 = self.client.post("/api/documents/upload", headers=self.admin_headers, files=files_dup, data=data)
        self.assertEqual(resp2.status_code, 409)
        self.assertIn("already exists", resp2.json()["detail"])

    def test_06_non_pdf_rejection(self):
        """Verify rejection of non-PDF uploads."""
        files = {
            "file": ("malicious_payload.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/octet-stream")
        }
        data = {"category": "RTI"}
        resp = self.client.post("/api/documents/upload", headers=self.admin_headers, files=files, data=data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Only PDF documents (.pdf) are permitted", resp.json()["detail"])

    def test_07_invalid_category_rejection(self):
        """Verify rejection of unallowed legal categories."""
        pdf_bytes = create_minimal_pdf_bytes("Random Doc", "Content")
        files = {"file": ("Random.pdf", pdf_bytes, "application/pdf")}
        data = {"category": "INVALID_UNKNOWN_CATEGORY"}
        resp = self.client.post("/api/documents/upload", headers=self.admin_headers, files=files, data=data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid category", resp.json()["detail"])

    def test_08_oversized_file_rejection(self):
        """Verify rejection of files exceeding MAX_UPLOAD_SIZE_BYTES."""
        fake_large_bytes = b"%PDF-1.4 " + (b"0" * (MAX_UPLOAD_SIZE_BYTES + 100))
        files = {"file": ("Oversized.pdf", fake_large_bytes, "application/pdf")}
        data = {"category": "RTI"}
        resp = self.client.post("/api/documents/upload", headers=self.admin_headers, files=files, data=data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("exceeds maximum permitted limit", resp.json()["detail"])

    def test_09_path_traversal_sanitization(self):
        """Verify that malicious directory traversal filenames are safely sanitized."""
        pdf_bytes = create_minimal_pdf_bytes("Traversal Test", "Testing traversal safety.")
        files = {
            "file": ("../../../../etc/passwd.pdf", pdf_bytes, "application/pdf")
        }
        data = {"category": "OTHER"}
        resp = self.client.post("/api/documents/upload", headers=self.admin_headers, files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("..", resp.json()["original_file_name"])

    def test_10_baseline_document_deletion_rejection(self):
        """Verify that deleting a baseline Government document returns HTTP 400 Bad Request."""
        resp = self.client.delete("/api/documents/DOC-BASELINE-0001", headers=self.admin_headers)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("protected baseline Government of India statute", resp.json()["detail"])

    def test_11_invalid_document_id_deletion(self):
        """Verify that deleting a non-existent document ID returns HTTP 404."""
        resp = self.client.delete("/api/documents/DOC-NON-EXISTENT-9999", headers=self.admin_headers)
        self.assertEqual(resp.status_code, 404)
        self.assertIn("does not exist", resp.json()["detail"])

    def test_12_delete_user_uploaded_document_and_verify_rag(self):
        """Verify deleting a user-uploaded document removes it from SQLite, FAISS, and RAG retrieval."""
        # 1. Upload a specific temporary document
        temp_pdf_bytes = create_minimal_pdf_bytes(
            "Ephemeral Civic Ordinance 2026",
            "This ephemeral civic ordinance stipulates unique temporary street parking rules in municipal zones."
        )
        files = {"file": ("Ephemeral_Ordinance_2026.pdf", temp_pdf_bytes, "application/pdf")}
        data = {"category": "CIVIC", "title": "Ephemeral Civic Ordinance 2026"}

        upload_resp = self.client.post("/api/documents/upload", headers=self.admin_headers, files=files, data=data)
        self.assertEqual(upload_resp.status_code, 200)
        doc_id = upload_resp.json()["document_id"]

        # 2. Verify searchability
        ask_resp1 = self.client.post(
            "/api/ask",
            json={"question": "What are the street parking rules under the Ephemeral Civic Ordinance?"}
        )
        self.assertEqual(ask_resp1.status_code, 200)
        source_docs1 = [s["document"] for s in ask_resp1.json()["sources"]]
        self.assertIn("Ephemeral_Ordinance_2026.pdf", source_docs1)

        # 3. Delete document
        del_resp = self.client.delete(f"/api/documents/{doc_id}", headers=self.admin_headers)
        self.assertEqual(del_resp.status_code, 200)
        del_data = del_resp.json()
        self.assertEqual(del_data["document_id"], doc_id)
        self.assertGreater(del_data["removed_chunks"], 0)

        # 4. Verify document no longer in SQLite
        get_resp = self.client.get(f"/api/documents/{doc_id}", headers=self.admin_headers)
        self.assertEqual(get_resp.status_code, 404)

        # 5. Verify document is no longer retrievable by RAG
        ask_resp2 = self.client.post(
            "/api/ask",
            json={"question": "What are the street parking rules under the Ephemeral Civic Ordinance?"}
        )
        self.assertEqual(ask_resp2.status_code, 200)
        source_docs2 = [s["document"] for s in ask_resp2.json()["sources"]]
        self.assertNotIn("Ephemeral_Ordinance_2026.pdf", source_docs2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
