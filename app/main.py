import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router as rag_router
from app.rag.pipeline import rag_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Pure RAG server...")
    
    # Auto-ingest uploads folder on startup if documents exist
    if settings.UPLOADS_DIR.exists():
        try:
            count = await rag_pipeline.ingest_folder(settings.UPLOADS_DIR)
            logger.info(f"Startup ingestion finished: {count} new chunk(s) indexed.")
        except Exception as e:
            logger.warning(f"Could not auto-ingest documents on startup: {e}")
            
    yield
    logger.info("Shutting down Pure RAG server.")


app = FastAPI(
    title="Pure RAG API",
    description="High-performance, minimal RAG implementation.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)


@app.get("/")
def root():
    return {
        "service": "Pure RAG Service",
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    existing_ids = await rag_pipeline.store.get_existing_ids()
    return {
        "status": "ok",
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "total_chunks_indexed": len(existing_ids),
        "embedding_model": settings.EMBEDDING_MODEL,
        "generation_model": settings.GENERATION_MODEL
    }
