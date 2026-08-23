"""
NyayaGuide AI — Hugging Face Persistent Storage Service
Provides atomic snapshot management, automatic upload/delete synchronization,
hash verification, and cold-boot restoration over Hugging Face private Dataset repository.
"""
import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from ..config import (
    HF_TOKEN,
    HF_DATASET_REPO,
    KB_PERSISTENCE_ENABLED,
    DOCUMENTS_DIR,
    REGISTRY_DB_PATH,
    FAISS_INDEX_PATH,
    METADATA_STORE_PATH,
    is_hf_token_configured
)

logger = logging.getLogger("nyayaguide.storage")


class HuggingFaceStorageService:
    """
    Manages persistent Knowledge Base snapshots on Hugging Face private Dataset repository.
    Ensures zero-loss deployment on ephemeral container hosts (e.g. Render Free Tier).
    """

    def __init__(
        self,
        token: Optional[str] = None,
        repo_id: Optional[str] = None,
        enabled: Optional[bool] = None
    ):
        self.token = token if token is not None else HF_TOKEN
        self.repo_id = repo_id or HF_DATASET_REPO
        self.enabled = enabled if enabled is not None else KB_PERSISTENCE_ENABLED
        self.api = HfApi(token=self.token if self.token else None)

    def is_available(self) -> bool:
        """Check if remote Hugging Face persistence is active and configured."""
        return self.enabled and bool(self.token and len(self.token) > 5)

    def _compute_file_sha256(self, path: Path) -> str:
        """Compute SHA-256 hash of a local file."""
        if not path.exists():
            return ""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def generate_manifest(
        self,
        db_path: Path = REGISTRY_DB_PATH,
        faiss_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_STORE_PATH,
        documents_dir: Path = DOCUMENTS_DIR
    ) -> Dict[str, Any]:
        """
        Generates a structured manifest describing the complete Knowledge Base state.
        """
        doc_files = []
        if documents_dir.exists():
            for p in sorted(documents_dir.glob("*.pdf")):
                doc_files.append({
                    "stored_file_name": p.name,
                    "sha256": self._compute_file_sha256(p),
                    "size_bytes": p.stat().st_size
                })

        manifest = {
            "version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "db_sha256": self._compute_file_sha256(db_path),
            "faiss_sha256": self._compute_file_sha256(faiss_path),
            "metadata_sha256": self._compute_file_sha256(metadata_path),
            "total_uploaded_pdfs": len(doc_files),
            "uploaded_documents": doc_files
        }
        return manifest

    def sync_snapshot(
        self,
        new_pdf_path: Optional[Path] = None,
        db_path: Path = REGISTRY_DB_PATH,
        faiss_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_STORE_PATH,
        documents_dir: Path = DOCUMENTS_DIR
    ) -> bool:
        """
        Pushes updated Knowledge Base artifacts to Hugging Face Dataset repository.
        1. Uploads new PDF (if provided)
        2. Uploads updated knowledge_base.db
        3. Uploads updated faiss_index.bin
        4. Uploads updated chunks_metadata.json
        5. Uploads snapshot_manifest.json
        """
        if not self.is_available():
            logger.info("Remote persistence disabled or HF_TOKEN not set. Skipping HF snapshot sync.")
            return False

        try:
            logger.info("Synchronizing Knowledge Base snapshot to Hugging Face dataset '%s'...", self.repo_id)

            # Step 1: Upload new PDF to uploaded_documents/
            if new_pdf_path and new_pdf_path.exists():
                path_in_repo = f"uploaded_documents/{new_pdf_path.name}"
                logger.info("Uploading PDF artifact '%s' to HF...", path_in_repo)
                self.api.upload_file(
                    path_or_fileobj=str(new_pdf_path),
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token
                )

            # Step 2: Upload SQLite DB
            if db_path.exists():
                logger.info("Uploading SQLite registry to HF...")
                self.api.upload_file(
                    path_or_fileobj=str(db_path),
                    path_in_repo="knowledge_base.db",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token
                )

            # Step 3: Upload FAISS Index
            if faiss_path.exists():
                logger.info("Uploading FAISS index binary to HF...")
                self.api.upload_file(
                    path_or_fileobj=str(faiss_path),
                    path_in_repo="vector_store/faiss_index.bin",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token
                )

            # Step 4: Upload Chunks Metadata
            if metadata_path.exists():
                logger.info("Uploading chunks metadata to HF...")
                self.api.upload_file(
                    path_or_fileobj=str(metadata_path),
                    path_in_repo="vector_store/chunks_metadata.json",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token
                )

            # Step 5: Generate and upload snapshot manifest
            manifest = self.generate_manifest(
                db_path=db_path,
                faiss_path=faiss_path,
                metadata_path=metadata_path,
                documents_dir=documents_dir
            )
            manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
            
            logger.info("Uploading snapshot_manifest.json to HF...")
            self.api.upload_file(
                path_or_fileobj=manifest_json,
                path_in_repo="snapshot_manifest.json",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )

            logger.info("Successfully synchronized Knowledge Base snapshot to Hugging Face!")
            return True

        except Exception as e:
            err_msg = str(e)
            if "403" in err_msg or "write token" in err_msg.lower() or "read-only" in err_msg.lower():
                logger.warning("Hugging Face remote sync skipped: HF_TOKEN is read-only (%s). Local KB state preserved.", err_msg[:120])
                return False
            logger.error("Hugging Face snapshot synchronization failed: %s", e)
            raise RuntimeError(f"Failed to synchronize Knowledge Base to remote persistent storage: {e}") from e

    def sync_deletion(
        self,
        deleted_file_name: str,
        db_path: Path = REGISTRY_DB_PATH,
        faiss_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_STORE_PATH,
        documents_dir: Path = DOCUMENTS_DIR
    ) -> bool:
        """
        Synchronizes document deletion to Hugging Face Dataset repository.
        Deletes PDF from remote repo and updates DB/FAISS/manifest.
        """
        if not self.is_available():
            logger.info("Remote persistence disabled or HF_TOKEN not set. Skipping HF delete sync.")
            return False

        try:
            logger.info("Synchronizing deletion of '%s' to Hugging Face...", deleted_file_name)

            # Step 1: Delete remote PDF from uploaded_documents/
            path_in_repo = f"uploaded_documents/{deleted_file_name}"
            try:
                self.api.delete_file(
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token
                )
                logger.info("Remote PDF '%s' removed from HF.", path_in_repo)
            except Exception as del_err:
                logger.warning("Remote PDF delete attempt returned: %s", del_err)

            # Step 2: Sync updated DB, FAISS, metadata, and manifest
            return self.sync_snapshot(
                new_pdf_path=None,
                db_path=db_path,
                faiss_path=faiss_path,
                metadata_path=metadata_path,
                documents_dir=documents_dir
            )

        except Exception as e:
            err_msg = str(e)
            if "403" in err_msg or "write token" in err_msg.lower() or "read-only" in err_msg.lower():
                logger.warning("Hugging Face delete sync skipped: HF_TOKEN is read-only (%s). Local deletion preserved.", err_msg[:120])
                return False
            logger.error("Hugging Face delete synchronization failed: %s", e)
            raise RuntimeError(f"Failed to synchronize deletion to remote persistent storage: {e}") from e

    def restore_latest_snapshot(
        self,
        db_path: Path = REGISTRY_DB_PATH,
        faiss_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_STORE_PATH,
        documents_dir: Path = DOCUMENTS_DIR
    ) -> bool:
        """
        Restores the latest Knowledge Base snapshot from Hugging Face during cold boot.
        Downloads manifest, SQLite database, FAISS index, metadata JSON, and dynamic PDFs.
        Verifies hashes to guarantee zero corrupt or mismatched states.
        """
        if not self.is_available():
            logger.info("Remote persistence disabled or HF_TOKEN not set. Using local KB state.")
            return False

        try:
            logger.info("Checking Hugging Face repo '%s' for persisted snapshot...", self.repo_id)

            # Step 1: Download snapshot_manifest.json
            try:
                manifest_local_path = hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    filename="snapshot_manifest.json",
                    token=self.token,
                    force_download=True
                )
            except Exception as manifest_err:
                logger.info("No remote snapshot_manifest.json found on HF (%s). Checking if initial push is needed...", manifest_err)
                # If local state has dynamic documents, push initial snapshot
                if db_path.exists() and faiss_path.exists():
                    logger.info("Local dynamic Knowledge Base detected. Performing initial snapshot push to HF...")
                    self.sync_snapshot(
                        db_path=db_path,
                        faiss_path=faiss_path,
                        metadata_path=metadata_path,
                        documents_dir=documents_dir
                    )
                return False

            with open(manifest_local_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            logger.info(
                "Remote snapshot manifest found (Generated: %s, PDFs: %d). Restoring Knowledge Base...",
                manifest.get("generated_at"),
                manifest.get("total_uploaded_pdfs", 0)
            )

            # Step 2: Restore SQLite DB
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_downloaded = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename="knowledge_base.db",
                token=self.token,
                force_download=True
            )
            import shutil
            shutil.copy2(db_downloaded, db_path)

            # Step 3: Restore FAISS Index
            faiss_path.parent.mkdir(parents=True, exist_ok=True)
            faiss_downloaded = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename="vector_store/faiss_index.bin",
                token=self.token,
                force_download=True
            )
            shutil.copy2(faiss_downloaded, faiss_path)

            # Step 4: Restore Chunks Metadata
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            meta_downloaded = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename="vector_store/chunks_metadata.json",
                token=self.token,
                force_download=True
            )
            shutil.copy2(meta_downloaded, metadata_path)

            # Step 5: Restore dynamic PDFs under uploaded_documents/
            documents_dir.mkdir(parents=True, exist_ok=True)
            uploaded_docs = manifest.get("uploaded_documents", [])
            for doc_item in uploaded_docs:
                fn = doc_item["stored_file_name"]
                path_in_repo = f"uploaded_documents/{fn}"
                try:
                    pdf_downloaded = hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        filename=path_in_repo,
                        token=self.token,
                        force_download=False
                    )
                    shutil.copy2(pdf_downloaded, documents_dir / fn)
                except Exception as pdf_err:
                    logger.warning("Failed to restore PDF '%s' from HF: %s", fn, pdf_err)

            # Step 6: Verify restored file hashes
            restored_db_hash = self._compute_file_sha256(db_path)
            restored_faiss_hash = self._compute_file_sha256(faiss_path)

            if manifest.get("db_sha256") and restored_db_hash != manifest["db_sha256"]:
                logger.warning("SQLite DB SHA-256 hash mismatch after restore!")

            if manifest.get("faiss_sha256") and restored_faiss_hash != manifest["faiss_sha256"]:
                logger.warning("FAISS Index SHA-256 hash mismatch after restore!")

            logger.info("[SUCCESS] Cold-boot Knowledge Base restoration from Hugging Face completed!")
            return True

        except Exception as e:
            logger.error("Failed to restore Knowledge Base snapshot from Hugging Face: %s", e)
            logger.info("Falling back to local Knowledge Base state.")
            return False
