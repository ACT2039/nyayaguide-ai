import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

from ..config import (
    HF_TOKEN,
    HF_DATASET_REPO,
    CACHE_DIR,
    TARGET_DOCUMENTS,
    LOCAL_DOC_DIRS,
    is_hf_token_configured
)


class HuggingFaceDatasetLoader:
    """
    Manages secure authentication and retrieval of PDF documents
    from the private Hugging Face dataset repository.
    """

    def __init__(self, token: Optional[str] = None, repo_id: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.token = token if token is not None else HF_TOKEN
        self.repo_id = repo_id or HF_DATASET_REPO
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api = HfApi(token=self.token if self.token else None)

    def verify_connection(self) -> Tuple[bool, str, List[str]]:
        """
        Verify connection and authentication to the Hugging Face private repository.
        Returns: (success: bool, message: str, available_files: list[str])
        """
        if not self.token:
            return (
                False,
                "HF_TOKEN environment variable is not set. Please set your Hugging Face read token in .env or environment.",
                []
            )

        try:
            # Check repo metadata
            repo_info = self.api.dataset_info(repo_id=self.repo_id, token=self.token)
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset", token=self.token)
            
            # Filter for PDF files or root files
            available_files = [f for f in files if not f.startswith(".")]
            
            # Verify target documents
            missing_targets = [doc for doc in TARGET_DOCUMENTS if doc not in available_files]
            if missing_targets:
                return (
                    True,
                    f"Connected to {self.repo_id} (Private: {repo_info.private}), but missing target files: {missing_targets}",
                    available_files
                )

            return (
                True,
                f"Connected successfully to private dataset '{self.repo_id}' (All 4 target PDFs verified)",
                available_files
            )

        except RepositoryNotFoundError:
            return (
                False,
                f"Repository '{self.repo_id}' was not found or your HF_TOKEN does not have read access to this private dataset.",
                []
            )
        except GatedRepoError:
            return (
                False,
                f"Repository '{self.repo_id}' is gated and requires accepted terms before access.",
                []
            )
        except Exception as e:
            return (
                False,
                f"Hugging Face connection failed ({type(e).__name__}): {e}",
                []
            )

    def download_document(self, filename: str) -> Path:
        """
        Download or retrieve a cached PDF document from the Hugging Face dataset.
        Returns the local Path to the downloaded file.
        """
        if not self.token:
            raise ValueError(
                f"Cannot download '{filename}' from Hugging Face: HF_TOKEN is not configured."
            )

        try:
            local_file = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=filename,
                token=self.token,
                cache_dir=str(self.cache_dir),
                force_download=False
            )
            return Path(local_file)
        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve '{filename}' from Hugging Face dataset '{self.repo_id}': {e}"
            ) from e

    def get_document_path(self, filename: str, allow_local_fallback: bool = True) -> Path:
        """
        Retrieve document path:
        1. From Hugging Face dataset if HF_TOKEN is available.
        2. Fallback to local copy if available and allow_local_fallback is True.
        """
        if is_hf_token_configured():
            try:
                return self.download_document(filename)
            except Exception as hf_err:
                if not allow_local_fallback:
                    raise hf_err
                print(f"HF retrieval for {filename} failed ({hf_err}), checking local fallback...")

        if allow_local_fallback:
            for folder in LOCAL_DOC_DIRS:
                candidate = folder / filename
                if candidate.exists() and candidate.stat().st_size > 1000:
                    return candidate

        raise FileNotFoundError(
            f"Document '{filename}' could not be retrieved from Hugging Face and was not found locally."
        )

    def clear_cache(self) -> int:
        """
        Cleans the temporary cache directory to free up disk space.
        Returns bytes freed.
        """
        if not self.cache_dir.exists():
            return 0
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("**/*") if f.is_file())
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return total_size

    def get_cache_info(self) -> dict:
        """Returns details on the current cache usage."""
        if not self.cache_dir.exists():
            return {"cache_dir": str(self.cache_dir), "files_count": 0, "size_bytes": 0}
        
        cached_files = [f for f in self.cache_dir.glob("**/*") if f.is_file()]
        total_size = sum(f.stat().st_size for f in cached_files)
        return {
            "cache_dir": str(self.cache_dir),
            "files_count": len(cached_files),
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2)
        }
