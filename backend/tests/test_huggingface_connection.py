"""
Hugging Face Connection & Authentication Test for NyayaGuide AI.
Tests access to the private dataset: charantejarangi123/govt_documents_rag_project
"""
import sys
import os
from pathlib import Path

# Reconfigure stdout for UTF-8 compatibility on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import (
    HF_TOKEN,
    HF_DATASET_REPO,
    TARGET_DOCUMENTS,
    is_hf_token_configured,
    get_safe_hf_token_status
)
from app.ingestion.hf_loader import HuggingFaceDatasetLoader


def test_huggingface_connection():
    print("=" * 60)
    print("NyayaGuide AI -- Hugging Face Connection Verification")
    print("=" * 60)
    print(f"Target Repository : {HF_DATASET_REPO}")
    print(f"Authentication    : {get_safe_hf_token_status()}")
    print("-" * 60)

    if not is_hf_token_configured():
        print("\n[ERROR] HF_TOKEN is missing or not configured.")
        print("Please configure your Hugging Face read access token:")
        print("  1. Create a .env file (copied from .env.example)")
        print("  2. Add: HF_TOKEN=hf_your_token_here")
        print("  3. Or set it in your environment: export HF_TOKEN=... or $env:HF_TOKEN=...")
        print("\nStopping further ingestion steps until authentication is configured.")
        return False

    loader = HuggingFaceDatasetLoader()
    success, message, files = loader.verify_connection()

    if not success:
        print(f"\n[AUTHENTICATION FAILED]")
        print(f"Reason: {message}")
        print("\nPlease check that:")
        print("  1. Your HF_TOKEN is valid and has read permissions.")
        print(f"  2. Your Hugging Face account has been granted access to private dataset '{HF_DATASET_REPO}'.")
        return False

    print(f"\nConnected successfully")
    print(f"Repository: {HF_DATASET_REPO}")
    print(f"\nAvailable files ({len(files)}):")
    for f in sorted(files):
        status_marker = "[TARGET PDF]" if f in TARGET_DOCUMENTS else "[FILE]"
        print(f"  - {status_marker} {f}")

    # Check that all 4 target documents are available
    missing = [doc for doc in TARGET_DOCUMENTS if doc not in files]
    print("-" * 60)
    if missing:
        print(f"[WARNING] Missing required target files in dataset: {missing}")
        return False
    else:
        print(f"[SUCCESS] All 4 target legal documents are verified and accessible:")
        for doc in TARGET_DOCUMENTS:
            print(f"  [OK] {doc}")
        print("=" * 60)
        return True


if __name__ == "__main__":
    passed = test_huggingface_connection()
    sys.exit(0 if passed else 1)
