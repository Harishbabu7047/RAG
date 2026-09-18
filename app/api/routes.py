import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.rag.pipeline import rag_pipeline
from app.api.schemas import IngestFolderRequest, QueryRequest, QueryResponse, IngestResponse

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(payload: IngestFolderRequest | None = None):
    """Ingests all documents from the specified folder, defaulting to the uploads/ directory."""
    folder = Path(payload.folder_path) if payload and payload.folder_path else settings.UPLOADS_DIR
    
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    try:
        count = await rag_pipeline.ingest_folder(folder)
        return IngestResponse(
            status="success",
            chunks_ingested=count,
            message=f"Ingested {count} chunk(s) from {folder}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF, TXT, or MD document into the uploads folder and ingests it."""
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: .pdf, .txt, .md")

    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d")
    target_dir = settings.UPLOADS_DIR / today_str
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / file.filename

    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        count = await rag_pipeline.ingest_file(destination)
        return IngestResponse(
            status="success",
            chunks_ingested=count,
            message=f"Uploaded and indexed '{file.filename}' ({count} chunks)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_rag(payload: QueryRequest):
    """Retrieves relevant chunks and generates a grounded answer."""
    try:
        result = await rag_pipeline.answer(question=payload.question, top_k=payload.top_k)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            chunks=result["chunks"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_query(payload: QueryRequest):
    """Streams generated answer tokens using Server-Sent Events."""
    async def token_generator():
        try:
            async for token in rag_pipeline.answer_stream(payload.question, top_k=payload.top_k):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR: {str(e)}]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@router.get("/stats")
async def get_stats():
    """Returns vector store statistics."""
    existing_ids = await rag_pipeline.store.get_existing_ids()
    return {
        "total_chunks": len(existing_ids),
        "embedding_model": settings.EMBEDDING_MODEL,
        "generation_model": settings.GENERATION_MODEL
    }
