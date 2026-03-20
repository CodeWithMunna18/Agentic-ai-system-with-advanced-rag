# app.py
# ─────────────────────────────────────────────────────────────
# FastAPI application entry point.
# Run with: uvicorn app:app --reload --port 8000
#
# Once running, visit:
#   http://localhost:8000/docs   → interactive Swagger UI
#   http://localhost:8000/status → quick health check
# ─────────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import lifespan
from api.routes import router


# Create the FastAPI app with lifespan (handles startup/shutdown)
app = FastAPI(
    title="Document Intelligence Platform",
    description=(
        "A RAG-powered document Q&A API built with Gemini and ChromaDB.\n\n"
        "**Endpoints:**\n"
        "- `GET /status` — health check\n"
        "- `POST /ask` — ask a question from your documents\n"
        "- `POST /ingest` — load new documents\n"
        "- `GET /documents` — list loaded documents\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Streamlit UI (running on a different port) to call this API
# In production, replace "*" with your actual frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routes
app.include_router(router)


# Root redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Document Intelligence API", "docs": "/docs"}