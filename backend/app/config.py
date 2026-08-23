import os
from pathlib import Path
from dotenv import load_dotenv

# Search and load .env from current directory, parent directories, or project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Try loading from possible .env locations
for env_path in [
    PROJECT_ROOT / ".env",
    WORKSPACE_ROOT / ".env",
    PROJECT_ROOT / "nyayaguide_knowledge_base" / ".env",
    Path.cwd() / ".env"
]:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break
else:
    load_dotenv()  # Default search

# Hugging Face Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "charantejarangi123/govt_documents_rag_project")

# Metadata file location
METADATA_FILE = PROJECT_ROOT / "metadata" / "documents.json"
if not METADATA_FILE.exists() and (PROJECT_ROOT / "nyayaguide_knowledge_base" / "metadata" / "documents.json").exists():
    METADATA_FILE = PROJECT_ROOT / "nyayaguide_knowledge_base" / "metadata" / "documents.json"
elif not METADATA_FILE.exists() and (WORKSPACE_ROOT / "metadata" / "documents.json").exists():
    METADATA_FILE = WORKSPACE_ROOT / "metadata" / "documents.json"

# Cache & Temporary download directory (kept separate from source code)
CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"

# Fallback local paths if offline or testing locally
LOCAL_DOC_DIRS = [
    PROJECT_ROOT / "rti",
    PROJECT_ROOT / "consumer",
    PROJECT_ROOT / "nyayaguide_knowledge_base" / "rti",
    PROJECT_ROOT / "nyayaguide_knowledge_base" / "consumer",
    WORKSPACE_ROOT / "rti",
    WORKSPACE_ROOT / "consumer",
    WORKSPACE_ROOT / "nyayaguide_knowledge_base" / "rti",
    WORKSPACE_ROOT / "nyayaguide_knowledge_base" / "consumer"
]

# Required Source PDFs
TARGET_DOCUMENTS = [
    "RTI_Act_2005.pdf",
    "RTI_Rules_2012.pdf",
    "Consumer_Protection_Act_2019.pdf",
    "Consumer_Commission_and_General_Rules_2020.pdf"
]

# Chunking Parameters (tuned for legal sections & clauses)
TARGET_CHUNK_SIZE_WORDS = 350    # Approx. 450-700 tokens
MAX_CHUNK_SIZE_WORDS = 600       # Approx. 800-1000 tokens
CHUNK_OVERLAP_WORDS = 60         # Modest overlap to maintain legal context continuity

# Phase 2: Embedding & Vector Store Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSION = 384
VECTOR_STORE_DIR = PROJECT_ROOT / "backend" / "data" / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_STORE_PATH = VECTOR_STORE_DIR / "chunks_metadata.json"
DEFAULT_TOP_K = 5

# Phase 3: OpenRouter LLM & RAG Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "liquid/lfm-2.5-2.6b:free").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions").strip()
OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "60"))

# Groundedness & Abstention Threshold
# For BAAI/bge-small-en-v1.5 with normalized embeddings, legal queries score >= 0.53,
# while out-of-domain queries (weather, coding, etc.) score <= 0.48.
MIN_RELEVANCE_THRESHOLD = float(os.getenv("MIN_RELEVANCE_THRESHOLD", "0.50"))
ABSTENTION_MESSAGE = "I could not find sufficient information in the current knowledge base to answer this reliably."

# Phase 4: FastAPI API Configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "").strip()
_default_origins = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
_configured_origins = os.getenv("CORS_ALLOWED_ORIGINS", _default_origins)
if FRONTEND_ORIGIN and FRONTEND_ORIGIN not in _configured_origins:
    CORS_ALLOWED_ORIGINS = f"{_configured_origins},{FRONTEND_ORIGIN}"
else:
    CORS_ALLOWED_ORIGINS = _configured_origins

MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))

# Phase 7: Knowledge Base Management & Dynamic Document Ingestion
DOCUMENTS_DIR = PROJECT_ROOT / "backend" / "data" / "documents"
REGISTRY_DB_PATH = PROJECT_ROOT / "backend" / "data" / "registry" / "knowledge_base.db"
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(20 * 1024 * 1024)))  # 20 MB default
ALLOWED_CATEGORIES = [
    "RTI",
    "CONSUMER",
    "CIVIC",
    "EDUCATION",
    "TRANSPORT",
    "ENVIRONMENT",
    "OTHER"
]
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "nyayaguide_admin_secret_2026").strip()
KB_PERSISTENCE_ENABLED = os.getenv("KB_PERSISTENCE_ENABLED", "true" if bool(HF_TOKEN and len(HF_TOKEN) > 5) else "false").lower() in ("true", "1", "yes")


def is_hf_token_configured() -> bool:
    """Check if HF_TOKEN is provided without exposing the value."""
    return bool(HF_TOKEN and len(HF_TOKEN) > 5)


def get_safe_hf_token_status() -> str:
    """Return safe status description of the HF token configuration."""
    if not HF_TOKEN:
        return "Not Set (HF_TOKEN is empty)"
    return f"Configured ({len(HF_TOKEN)} chars, starts with {HF_TOKEN[:4]}...)"


def is_openrouter_configured() -> bool:
    """Check if OPENROUTER_API_KEY is provided without exposing the value."""
    return bool(OPENROUTER_API_KEY and len(OPENROUTER_API_KEY) > 10)


def get_safe_openrouter_status() -> str:
    """Return safe status description of the OpenRouter configuration."""
    if not OPENROUTER_API_KEY:
        return "Not Set (OPENROUTER_API_KEY is empty)"
    return f"Configured ({len(OPENROUTER_API_KEY)} chars, model: {OPENROUTER_MODEL})"


def is_admin_key_valid(provided_key: Optional[str]) -> bool:
    """Validate administrative API key using constant-time comparison."""
    import secrets
    if not provided_key or not ADMIN_API_KEY:
        return False
    return secrets.compare_digest(provided_key.strip(), ADMIN_API_KEY)

