# api/routes.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: Routes are the "endpoints" of your API.
# Each route is a URL + HTTP method that clients call.
#
# Our API exposes 4 endpoints:
#
#   GET  /status    → health check + document store info
#   POST /ask       → ask a question, get a grounded answer
#   POST /ingest    → load new documents into the pipeline
#   GET  /documents → list all documents in the store
#
# FastAPI uses Python decorators (@router.get, @router.post)
# to register route handlers. The type annotations on function
# parameters tell FastAPI how to parse and validate inputs.
# ─────────────────────────────────────────────────────────────

import pathlib
import shutil
from fastapi import APIRouter, Request, HTTPException, UploadFile, File

from .models import (
    AskRequest, AskResponse, SourceModel,
    StatusResponse, IngestRequest, IngestResponse, UploadResponse,
)
from .dependencies import build_pipeline, get_pipeline, get_store, get_chunks
from src.ingestion import load_document
from src.chunking import chunk_document
from src.embedding import embed_chunks

DOCUMENTS_DIR = pathlib.Path("documents")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".text"}

router = APIRouter()


# ── GET /status ────────────────────────────────────────────────
@router.get(
    "/status",
    response_model=StatusResponse,
    summary="API health check and store info",
)
async def get_status(request: Request):
    """
    Returns the current state of the document store.
    Use this to verify the API is running and documents are loaded.
    """
    store = get_store(request)
    stats = store.get_stats()

    return StatusResponse(
        status="ready",
        total_chunks=stats.get("total_chunks", 0),
        documents=stats.get("source_files", []),
        num_documents=stats.get("num_documents", 0),
    )


# ── POST /ask ──────────────────────────────────────────────────
@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question from your documents",
)
async def ask_question(body: AskRequest, request: Request):
    """
    The core RAG endpoint. Accepts a question, retrieves relevant
    chunks, and returns a grounded answer with source citations.

    Modes:
    - hybrid:    dense + BM25 keyword search (best default)
    - dense:     semantic vector search only
    - multi_doc: synthesize answer across all documents
    """
    pipeline = get_pipeline(request)
    store    = get_store(request)

    if store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents in store. POST to /ingest first."
        )

    try:
        # Route to the right pipeline method based on mode
        if body.mode == "multi_doc":
            response = pipeline.ask_across_documents(question=body.question)
        elif body.source_file:
            response = pipeline.ask_document(
                question=body.question,
                source_file=body.source_file,
            )
        else:
            response = pipeline.ask(question=body.question)

        # Convert RAGResponse → AskResponse (Pydantic model)
        sources = [
            SourceModel(
                source_file=s.get("source_file", "unknown"),
                score=s.get("score", 0.0),
                chunk_index=s.get("chunk_index", 0),
                preview=s.get("preview", ""),
            )
            for s in response.sources
        ]

        return AskResponse(
            question=response.question,
            answer=response.answer,
            sources=sources,
            mode=body.mode,
            chunks_retrieved=len(sources),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /ingest ───────────────────────────────────────────────
@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Load documents from disk into the pipeline",
)
async def ingest_documents(body: IngestRequest, request: Request):
    """
    Re-runs the ingestion pipeline (Phases 1–3) on the documents folder.
    Call this after adding new documents to the documents/ folder.

    CONCEPT: This endpoint makes the system "live" — you can add
    documents while the API is running and update the store without
    restarting the server.
    """
    docs_path = pathlib.Path(body.documents_dir)
    if not docs_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Documents directory not found: {body.documents_dir}"
        )

    try:
        # Rebuild pipeline with new documents
        pipeline, store, chunks = build_pipeline(body.documents_dir)

        # Update app state so future requests use the new pipeline
        request.app.state.pipeline = pipeline
        request.app.state.store    = store
        request.app.state.chunks   = chunks

        stats = store.get_stats()

        return IngestResponse(
            success=True,
            documents_loaded=stats.get("num_documents", 0),
            chunks_created=len(chunks),
            message=f"Successfully ingested {stats.get('num_documents', 0)} "
                    f"document(s) into {len(chunks)} chunks.",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /upload ───────────────────────────────────────────────
@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a document file and add it to the vector store",
)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF, TXT, or DOCX file directly from the UI.

    CONCEPT: This endpoint does incremental ingestion —
    it processes ONLY the new file and adds its chunks to the
    existing ChromaDB store, without touching other documents.

    WHY incremental vs full rebuild?
    Full rebuild (build_pipeline) re-embeds ALL documents.
    Incremental only embeds the NEW file → much faster.
    ChromaDB upsert ensures no duplicates if the same file
    is uploaded twice.

    FLOW:
      1. Validate file type
      2. Save file to documents/ folder
      3. Load + clean text (Phase 1)
      4. Chunk (Phase 2)
      5. Embed new chunks (Phase 3)
      6. Upsert into ChromaDB (Phase 3)
      7. Rebuild BM25 index in pipeline (Phase 5)
      8. Return stats
    """
    # ── Step 1: Validate file type ─────────────────────────────
    suffix = pathlib.Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # ── Step 2: Save to documents/ folder ──────────────────────
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    save_path = DOCUMENTS_DIR / file.filename

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        await file.close()

    # ── Step 3: Load + clean (Phase 1) ─────────────────────────
    doc = load_document(save_path)
    if not doc.is_valid:
        save_path.unlink(missing_ok=True)   # clean up bad file
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract text from '{file.filename}': {doc.error}"
        )

    # ── Step 4: Chunk (Phase 2) ─────────────────────────────────
    new_chunks = chunk_document(
        doc,
        strategy="sentence",
        max_tokens=200,
        overlap_sentences=1,
    )
    if not new_chunks:
        raise HTTPException(
            status_code=422,
            detail=f"Document '{file.filename}' produced no chunks after processing."
        )

    # ── Step 5 & 6: Embed + store (Phase 3) ────────────────────
    store = get_store(request)
    embedded = embed_chunks(new_chunks, batch_size=20, show_progress=False)
    store.add_chunks(embedded)

    # ── Step 7: Update in-memory state ─────────────────────────
    # Add new chunks to app.state.chunks so BM25 index stays current
    existing_chunks = get_chunks(request)
    all_chunks = existing_chunks + embedded

    # Rebuild the pipeline's hybrid retriever with the updated chunk list
    # (BM25 index must include new chunks to search them by keyword)
    pipeline = get_pipeline(request)
    from src.rag.bm25 import BM25Index
    pipeline.retriever.bm25      = BM25Index(all_chunks)
    pipeline.retriever.chunks    = all_chunks
    request.app.state.chunks     = all_chunks

    print(f"✅ Uploaded '{file.filename}': {len(new_chunks)} new chunks added")

    return UploadResponse(
        success=True,
        file_name=file.filename,
        chunks_created=len(new_chunks),
        message=(
            f"'{file.filename}' ingested successfully. "
            f"{len(new_chunks)} chunks added to the vector store. "
            f"Total chunks now: {store.count()}."
        ),
    )


# ── GET /documents ─────────────────────────────────────────────
@router.get(
    "/documents",
    summary="List all documents in the store",
)
async def list_documents(request: Request):
    """
    Returns a list of all documents currently in the vector store.
    """
    store = get_store(request)
    stats = store.get_stats()
    return {
        "documents": stats.get("source_files", []),
        "total_chunks": stats.get("total_chunks", 0),
    }