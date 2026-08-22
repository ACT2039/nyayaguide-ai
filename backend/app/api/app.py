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
    The NyayaRAGPipeline (embedding model, FAISS index, metadata, OpenRouter client)
    is loaded exactly once and shared across all requests via app.state.
    """
    logger.info("Initializing NyayaGuide AI RAG Pipeline (singleton)...")
    pipeline = NyayaRAGPipeline()
    application.state.pipeline = pipeline

    from ..services.document_service import DocumentService
    document_service = DocumentService()
    application.state.document_service = document_service

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
