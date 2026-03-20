# src/embedding/embedder.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: An "embedder" converts text → vector using Gemini.
#
# WHY Gemini text-embedding-004?
#   - 768-dimensional vectors (good balance of quality vs speed)
#   - Supports "task types" — tells the model HOW the embedding
#     will be used, which improves relevance significantly:
#
#   RETRIEVAL_DOCUMENT  → use when embedding chunks to STORE
#   RETRIEVAL_QUERY     → use when embedding a user's QUESTION
#   SEMANTIC_SIMILARITY → use when comparing two pieces of text
#
#   This matters! The same sentence embedded with DOCUMENT vs QUERY
#   task types produces slightly different vectors, optimized for
#   the retrieval direction (query → document).
#
# BATCHING: Gemini allows embedding multiple texts in one API call.
#   We batch chunks to reduce API calls and respect rate limits.
# ─────────────────────────────────────────────────────────────

import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from ..chunking.chunk import Chunk

# Load API key from .env
load_dotenv()

# Module-level client — initialized once via init_gemini()
_client: genai.Client | None = None


# ── Setup ──────────────────────────────────────────────────────
def init_gemini() -> genai.Client:
    """
    Initialize the new google-genai client with your API key.
    Call this once at startup before any embedding calls.
    """
    global _client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY not set!\n"
            "Open your .env file and replace 'your_api_key_here' with your real key.\n"
            "Get one free at: https://aistudio.google.com"
        )
    _client = genai.Client(api_key=api_key)
    print("✅ Gemini API initialized (google-genai SDK)")
    return _client


def _get_client() -> genai.Client:
    """Return the initialized client, or raise a clear error."""
    if _client is None:
        raise RuntimeError("Call init_gemini() before embedding.")
    return _client


# ── Single text embedding ──────────────────────────────────────
def embed_text(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    """
    Embed a single piece of text using Gemini text-embedding-004.

    Args:
        text:      the text to embed
        task_type: how this embedding will be used
                   "RETRIEVAL_DOCUMENT" — for chunks being stored
                   "RETRIEVAL_QUERY"    — for user questions
                   "SEMANTIC_SIMILARITY"— for comparing two texts

    Returns:
        A list of 768 floats representing the semantic meaning of the text.
    """
    client = _get_client()
    response = client.models.embed_content(
        model="gemini-embedding-001",          # no "models/" prefix in new SDK
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return response.embeddings[0].values


def embed_query(question: str) -> list[float]:
    """
    Shortcut for embedding a user's search query.
    Uses RETRIEVAL_QUERY task type — optimized for finding relevant documents.
    """
    return embed_text(question, task_type="RETRIEVAL_QUERY")


# ── Batch embedding (efficient for many chunks) ────────────────
def embed_chunks(
    chunks: list[Chunk],
    batch_size: int = 20,
    delay_between_batches: float = 0.5,
    show_progress: bool = True,
) -> list[Chunk]:
    """
    Embed a list of Chunk objects in batches.

    WHY batching?
    Making 17 individual API calls vs 1 batch call:
    - 17 calls: ~17 seconds, 17x network overhead
    - 1 batch:  ~1 second, 1x network overhead
    Batching is always faster and uses fewer API quota units.

    This function MUTATES the chunks in-place by setting chunk.embedding,
    then returns the same list.
    """
    if not chunks:
        print("⚠️  No chunks to embed.")
        return []

    client = _get_client()
    total = len(chunks)
    embedded_count = 0
    failed_count = 0

    print(f"🔢 Embedding {total} chunks in batches of {batch_size}...")

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        batch_texts = [chunk.text for chunk in batch]

        try:
            # New SDK: pass a list of strings → get a list of embeddings back
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch_texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )

            # response.embeddings is a list of ContentEmbedding objects
            for chunk, embedding_obj in zip(batch, response.embeddings):
                chunk.embedding = embedding_obj.values

            embedded_count += len(batch)

            if show_progress:
                pct = (embedded_count / total) * 100
                print(f"   Batch {batch_start // batch_size + 1}: "
                      f"{embedded_count}/{total} chunks embedded ({pct:.0f}%)")

        except Exception as e:
            print(f"   ❌ Batch failed: {e}")
            failed_count += len(batch)

        if batch_start + batch_size < total:
            time.sleep(delay_between_batches)

    print(f"\n✅ Embedding complete: {embedded_count} succeeded, {failed_count} failed")
    return chunks


# ── Similarity helper ──────────────────────────────────────────
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Result range: 0.0 (unrelated) to 1.0 (identical meaning)

    CONCEPT: Measures the ANGLE between two vectors, not their distance.
    Similar meaning → small angle → high cosine score.

    Formula: cos(θ) = (A · B) / (|A| × |B|)
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = sum(a * a for a in vec_a) ** 0.5
    magnitude_b = sum(b * b for b in vec_b) ** 0.5
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)