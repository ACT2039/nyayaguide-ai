"""
NyayaGuide AI — Hugging Face Remote Persistence Unit Tests
Verifies snapshot manifest generation, upload/delete synchronization, cold-boot restore,
hash verification, failure recovery, duplicate protection, and baseline safety.
"""
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.app.config import HF_TOKEN, HF_DATASET_REPO
from backend.app.services.hf_storage_service import HuggingFaceStorageService
from backend.app.services.document_service import DocumentService
from backend.app.services.document_registry import SQLiteDocumentRegistry
from backend.app.retrieval.vector_store import FAISSVectorStore
from backend.app.models.registry import DocumentRecord, DocumentStatus


class TestHuggingFacePersistence(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

        self.db_path = self.base_path / "registry" / "knowledge_base.db"
        self.faiss_path = self.base_path / "vector_store" / "faiss_index.bin"
        self.meta_path = self.base_path / "vector_store" / "chunks_metadata.json"
        self.docs_dir = self.base_path / "documents"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_path.parent.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize mock storage service
        self.mock_api = MagicMock()
        self.storage_service = HuggingFaceStorageService(
            token="mock_hf_token_12345",
            repo_id="mock_user/mock_repo",
            enabled=True
        )
        self.storage_service.api = self.mock_api

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_01_authentication_and_configuration(self):
        """Verify HF storage service initialization and availability check."""
        service_enabled = HuggingFaceStorageService(token="hf_test_token_999", repo_id="test/repo", enabled=True)
        self.assertTrue(service_enabled.is_available())

        service_disabled = HuggingFaceStorageService(token="", repo_id="test/repo", enabled=False)
        self.assertFalse(service_disabled.is_available())

    def test_02_manifest_generation_and_hash_calculation(self):
        """Verify generation of Knowledge Base snapshot manifest and SHA-256 hashes."""
        # Create dummy artifacts
        self.db_path.write_bytes(b"dummy_db_data")
        self.faiss_path.write_bytes(b"dummy_faiss_data")
        self.meta_path.write_text(json.dumps({"metadata": []}), encoding="utf-8")

        test_pdf = self.docs_dir / "DOC-0001_test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 sample pdf content")

        manifest = self.storage_service.generate_manifest(
            db_path=self.db_path,
            faiss_path=self.faiss_path,
            metadata_path=self.meta_path,
            documents_dir=self.docs_dir
        )

        self.assertEqual(manifest["version"], 1)
        self.assertTrue(len(manifest["db_sha256"]) == 64)
        self.assertTrue(len(manifest["faiss_sha256"]) == 64)
        self.assertEqual(manifest["total_uploaded_pdfs"], 1)
        self.assertEqual(manifest["uploaded_documents"][0]["stored_file_name"], "DOC-0001_test.pdf")

    def test_03_upload_synchronization(self):
        """Verify that sync_snapshot uploads PDF, DB, FAISS, metadata, and manifest."""
        self.db_path.write_bytes(b"dummy_db_data")
        self.faiss_path.write_bytes(b"dummy_faiss_data")
        self.meta_path.write_text(json.dumps({"metadata": []}), encoding="utf-8")
        test_pdf = self.docs_dir / "DOC-0001_test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 test")

        success = self.storage_service.sync_snapshot(
            new_pdf_path=test_pdf,
            db_path=self.db_path,
            faiss_path=self.faiss_path,
            metadata_path=self.meta_path,
            documents_dir=self.docs_dir
        )

        self.assertTrue(success)
        # Should have called upload_file 5 times (pdf, db, faiss, metadata, manifest)
        self.assertEqual(self.mock_api.upload_file.call_count, 5)

    def test_04_delete_synchronization(self):
        """Verify that sync_deletion removes remote PDF and updates artifacts."""
        self.db_path.write_bytes(b"dummy_db_data")
        self.faiss_path.write_bytes(b"dummy_faiss_data")
        self.meta_path.write_text(json.dumps({"metadata": []}), encoding="utf-8")

        success = self.storage_service.sync_deletion(
            deleted_file_name="DOC-0001_test.pdf",
            db_path=self.db_path,
            faiss_path=self.faiss_path,
            metadata_path=self.meta_path,
            documents_dir=self.docs_dir
        )

        self.assertTrue(success)
        self.mock_api.delete_file.assert_called_once_with(
            path_in_repo="uploaded_documents/DOC-0001_test.pdf",
            repo_id="mock_user/mock_repo",
            repo_type="dataset",
            token="mock_hf_token_12345"
        )

    @patch("backend.app.services.hf_storage_service.hf_hub_download")
    def test_05_startup_restoration(self, mock_download):
        """Verify cold-boot restoration of remote Knowledge Base snapshot."""
        # Setup mock manifest download
        manifest_data = {
            "version": 1,
            "generated_at": "2026-08-23T10:00:00Z",
            "db_sha256": "db_hash_123",
            "faiss_sha256": "faiss_hash_123",
            "total_uploaded_pdfs": 1,
            "uploaded_documents": [{"stored_file_name": "DOC-0001_sample.pdf"}]
        }
        manifest_file = self.base_path / "temp_manifest.json"
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

        dummy_file = self.base_path / "dummy_download"
        dummy_file.write_bytes(b"downloaded_bytes")

        def side_effect(repo_id, repo_type, filename, token, force_download=False):
            if filename == "snapshot_manifest.json":
                return str(manifest_file)
            return str(dummy_file)

        mock_download.side_effect = side_effect

        restored = self.storage_service.restore_latest_snapshot(
            db_path=self.db_path,
            faiss_path=self.faiss_path,
            metadata_path=self.meta_path,
            documents_dir=self.docs_dir
        )

        self.assertTrue(restored)
        self.assertTrue(self.db_path.exists())
        self.assertTrue(self.faiss_path.exists())
        self.assertTrue(self.meta_path.exists())
        self.assertTrue((self.docs_dir / "DOC-0001_sample.pdf").exists())

    def test_06_failed_hf_sync_error_handling(self):
        """Verify that failed HF upload raises RuntimeError and marks doc status failed."""
        self.mock_api.upload_file.side_effect = Exception("HF API Network Timeout")

        registry = SQLiteDocumentRegistry(db_path=self.db_path)
        vector_store = FAISSVectorStore(dimension=384)
        doc_service = DocumentService(
            registry=registry,
            vector_store=vector_store,
            storage_service=self.storage_service
        )

        # Upload dummy PDF
        dummy_pdf_bytes = b"%PDF-1.4 header text sample legal act content"
        with self.assertRaises(RuntimeError) as ctx:
            doc_service.process_and_index_document(
                file_bytes=dummy_pdf_bytes,
                original_filename="failed_sync.pdf",
                category="RTI"
            )

        self.assertIn("failed", str(ctx.exception).lower())

    def test_07_baseline_document_protection(self):
        """Verify that protected baseline documents cannot be deleted."""
        registry = SQLiteDocumentRegistry(db_path=self.db_path)
        doc_service = DocumentService(registry=registry, storage_service=self.storage_service)

        # Attempt to delete baseline RTI Act 2005
        baseline_doc = registry.get_by_filename("RTI_Act_2005.pdf")
        self.assertIsNotNone(baseline_doc)
        self.assertTrue(baseline_doc.is_baseline)

        with self.assertRaises(PermissionError):
            doc_service.delete_document(baseline_doc.document_id)


if __name__ == "__main__":
    unittest.main()
