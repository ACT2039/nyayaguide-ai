"""
NyayaGuide AI — FastAPI Application
Creates the FastAPI app, configures CORS, and initializes the singleton RAG pipeline.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import CORS_ALLOWED_ORIGINS, OPENROUTER_MODEL
from ..rag.rag_pipeline import NyayaRAGPipeline
from .routes import router

logger = logging.getLogger("nyayaguide.api")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Application lifespan: initialize heavy resources once at startup, clean up at shutdown.
    1. Restores latest persistent Knowledge Base snapshot from Hugging Face (if available).
    2. Initializes the NyayaRAGPipeline (embedding model, FAISS index, metadata, OpenRouter client).
    3. Initializes DocumentService with remote persistence.
    """
    logger.info("Initializing NyayaGuide AI RAG Pipeline & Remote Persistence...")

    from ..services.hf_storage_service import HuggingFaceStorageService
    storage_service = HuggingFaceStorageService()
    if storage_service.is_available():
        try:
            restored = storage_service.restore_latest_snapshot()
            if restored:
                logger.info("Cold-boot Knowledge Base snapshot restored successfully from Hugging Face.")
        except Exception as restore_err:
            logger.warning("Cold-boot snapshot restoration from Hugging Face notice: %s", restore_err)

    pipeline = NyayaRAGPipeline()
    application.state.pipeline = pipeline

    from ..services.document_service import DocumentService
    document_service = DocumentService(storage_service=storage_service)
    application.state.document_service = document_service

    # Warm up singleton EmbeddingEngine in background thread so model is ready before first upload/query
    import asyncio
    from starlette.concurrency import run_in_threadpool
    from ..retrieval.embeddings import EmbeddingEngine

    async def _warmup_bge():
        try:
            engine = EmbeddingEngine.get_instance()
            await run_in_threadpool(lambda: engine.model)
            logger.info("EmbeddingEngine BGE model pre-warmup completed.")
        except Exception as w_err:
            logger.warning("BGE model warmup notice: %s", w_err)

    asyncio.create_task(_warmup_bge())

    logger.info(
        "RAG Pipeline & Document Management ready. Model: %s, FAISS vectors loaded, SQLite registry synced.",
        OPENROUTER_MODEL,
    )
    yield
    # Shutdown: nothing heavy to clean up
    logger.info("NyayaGuide AI API shutting down.")


app = FastAPI(
    title="NyayaGuide AI",
    description=(
        "Source-grounded civic rights assistant for Indian government acts and rules. "
        "Uses RAG (Retrieval-Augmented Generation) over official Government of India "
        "legal documents including the RTI Act 2005, RTI Rules 2012, "
        "Consumer Protection Act 2019, and Consumer Commission Rules 2020."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ──────────────────────────────────────────────
# CORS Configuration
# ──────────────────────────────────────────────
origins = [o.strip() for o in CORS_ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-Admin-Key", "Authorization"],
)

# ──────────────────────────────────────────────
# Include API routes
# ──────────────────────────────────────────────
app.include_router(router)
