# src/rag/retriever.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: The Retriever is the bridge between the vector store
# and the LLM. Its job:
#   1. Take a user's question
#   2. Embed it with RETRIEVAL_QUERY task type
#   3. Search ChromaDB for the most relevant chunks
#   4. Filter out low-confidence results (relevance threshold)
#   5. Return ranked, enriched results ready for prompt building
#
# WHY a separate retriever module?
# The retriever has its own concerns — threshold tuning, result
# deduplication, re-ranking. Keeping it separate from the LLM
# call makes each part independently testable and tunable.
# ─────────────────────────────────────────────────────────────

from dataclasses import dataclass
from ..embedding.embedder import embed_query
from ..embedding.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    """
    A search result — a chunk with its relevance score attached.
    This is what flows into the prompt builder.
    """
    text: str
    score: float          # cosine similarity: 0.0–1.0
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int

    def __repr__(self):
        return (f"RetrievedChunk(score={self.score:.4f}, "
                f"source='{self.source_file}', "
                f"index={self.chunk_index})")


class Retriever:
    """
    Handles the full retrieval pipeline:
    question → embedding → vector search → filtered ranked results
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        relevance_threshold: float = 0.5,
    ):
        """
        Args:
            vector_store:        the ChromaDB store from Phase 3
            top_k:               max number of chunks to retrieve
            relevance_threshold: minimum similarity score to include
                                 0.7+ = high confidence
                                 0.5+ = moderate (good default)
                                 0.3+ = loose (use when docs are sparse)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self,
        question: str,
        source_file: str | None = None,   # optional: filter by document
    ) -> list[RetrievedChunk]:
        """
        Main retrieval method.

        Steps:
        1. Embed the question with RETRIEVAL_QUERY task type
        2. Search ChromaDB for top_k similar chunks
        3. Filter by relevance_threshold
        4. Return as RetrievedChunk objects

        WHY RETRIEVAL_QUERY (not RETRIEVAL_DOCUMENT)?
        The embedding model is trained to make query vectors "point toward"
        document vectors. Using the wrong task type reduces retrieval quality.
        """
        # Step 1: Embed the question
        query_vec = embed_query(question)

        # Step 2: Search vector store
        raw_results = self.vector_store.search(
            query_embedding=query_vec,
            top_k=self.top_k,
            source_file=source_file,
        )

        # Step 3: Filter and convert to RetrievedChunk
        results = []
        for hit in raw_results:
            if hit["score"] < self.relevance_threshold:
                continue                     # skip low-confidence results

            results.append(RetrievedChunk(
                text=hit["text"],
                score=hit["score"],
                source_file=hit["metadata"].get("source_file", "unknown"),
                chunk_index=hit["metadata"].get("chunk_index", 0),
                start_char=hit["metadata"].get("start_char", 0),
                end_char=hit["metadata"].get("end_char", 0),
            ))

        return results   # already sorted by score desc from vector_store