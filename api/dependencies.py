# api/dependencies.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: Dependency Injection in FastAPI
#
# The RAG pipeline is expensive to initialize — it builds the
# BM25 index, connects to ChromaDB, and loads the Gemini client.
# We do NOT want to rebuild it on every API request.
#
# FastAPI's dependency injection system (Depends) lets us create
# ONE pipeline instance at startup and share it across ALL requests.
# This is the standard pattern for stateful resources in APIs.
#
# STARTUP FLOW:
#   1. FastAPI starts → lifespan() runs
#   2. lifespan() initializes the pipeline → stores in app.state
#   3. Every request calls get_pipeline() → returns same instance
#   4. FastAPI shuts down → lifespan() cleanup runs
# ─────────────────────────────────────────────────────────────

import pathlib
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.ingestion import load_documents_from_folder
from src.chunking import chunk_documents
from src.embedding import init_gemini, embed_chunks, VectorStore
from src.rag import AdvancedRAGPipeline


def build_pipeline(documents_dir: str = "documents") -> AdvancedRAGPipeline:
    """
    Run the full Phase 1–5 pipeline and return a ready-to-use
    AdvancedRAGPipeline. Called once at API startup.

    WHY rebuild on startup instead of using the persisted ChromaDB?
    ChromaDB persists embeddings across runs — we don't re-embed.
    But the BM25 index lives in memory and must be rebuilt each time.
    The rebuild is fast (< 1 second for thousands of chunks).
    """
    print("\n🔧 Initializing RAG pipeline for API...")

    # Phase 1: Load documents
    docs_path = pathlib.Path(documents_dir)
    if not docs_path.exists():
        docs_path.mkdir()

    documents = load_documents_from_folder(documents_dir)
    valid_docs = [d for d in documents if d.is_valid]

    if not valid_docs:
        print("⚠️  No valid documents found. Upload documents to get started.")

    # Phase 2: Chunk
    chunks = chunk_documents(
        valid_docs,
        strategy="sentence",
        max_tokens=200,
        overlap_sentences=1,
    ) if valid_docs else []

    # Phase 3: Embed + store (ChromaDB upsert is idempotent — safe to re-run)
    init_gemini()
    store = VectorStore(persist_directory="./chroma_db", collection_name="documents")

    if chunks:
        embedded = embed_chunks(chunks, batch_size=20, show_progress=True)
        store.add_chunks(embedded)

    # Phase 5: Advanced pipeline
    pipeline = AdvancedRAGPipeline(
        vector_store=store,
        chunks=chunks,
        top_k=5,
        model="gemini-2.5-flash",
        dense_weight=0.6,
        sparse_weight=0.4,
    )

    print("✅ RAG pipeline ready\n")
    return pipeline, store, chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Code before 'yield' runs at startup.
    Code after 'yield' runs at shutdown.
    """
    pipeline, store, chunks = build_pipeline()

    # Store on app.state so all routes can access them
    app.state.pipeline = pipeline
    app.state.store    = store
    app.state.chunks   = chunks

    yield  # ← API is live here, handling requests

    # Cleanup (nothing needed for ChromaDB/Gemini)
    print("👋 API shutting down")


def get_pipeline(request) -> AdvancedRAGPipeline:
    """FastAPI dependency — returns the shared pipeline instance."""
    return request.app.state.pipeline


def get_store(request) -> VectorStore:
    """FastAPI dependency — returns the shared vector store."""
    return request.app.state.store


def get_chunks(request) -> list:
    """FastAPI dependency — returns the current chunk list."""
    return request.app.state.chunks