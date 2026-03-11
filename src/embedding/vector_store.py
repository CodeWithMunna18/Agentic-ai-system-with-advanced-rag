# src/embedding/vector_store.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: A Vector Store is a database optimized for storing
# and searching vectors (embeddings).
#
# WHY ChromaDB?
#   - Runs locally (no server needed, no cost)
#   - Stores embeddings + text + metadata together
#   - Supports fast similarity search out of the box
#   - Easy to inspect and understand (great for learning)
#   - Production-ready: same API scales to millions of vectors
#
# HOW ChromaDB works internally:
#   1. You give it: text + embedding + metadata + an ID
#   2. It stores them in a persistent local database (a folder)
#   3. You give it a query embedding
#   4. It computes cosine similarity against ALL stored vectors
#   5. Returns the top-k most similar ones
#
# In Phase 4, step 3-5 is what powers "retrieval" in RAG.
# ─────────────────────────────────────────────────────────────

import chromadb
from chromadb.config import Settings
from ..chunking.chunk import Chunk


# ── VectorStore class ──────────────────────────────────────────
class VectorStore:
    """
    Wraps ChromaDB to provide a simple, RAG-focused interface.

    Responsibilities:
    - Store embedded chunks persistently on disk
    - Search for the most relevant chunks given a query vector
    - Provide inspection tools (count, peek, stats)
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
    ):
        """
        Args:
            persist_directory: folder where ChromaDB saves its data
                               (creates it if it doesn't exist)
            collection_name:   name for this group of documents
                               (like a table in a regular database)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Create the ChromaDB client with persistence
        # This creates/opens a local database in persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the collection
        # A "collection" is like a table — groups related documents
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # cosine is best for text embeddings (matches our similarity metric)
            metadata={"hnsw:space": "cosine"},
        )

        print(f"📁 VectorStore ready: '{collection_name}' "
              f"({self.collection.count()} existing chunks)")


    # ── Store chunks ───────────────────────────────────────────
    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Store embedded chunks in ChromaDB.

        ChromaDB needs 4 things per item:
        1. ids        — unique string identifier
        2. embeddings — the vector (list of floats)
        3. documents  — the original text (stored for retrieval)
        4. metadatas  — any extra info (source file, position, etc.)

        WHY store metadata?
        In Phase 5, you'll filter results like:
        "only return chunks from finance_report.pdf"
        "only return chunks from the first half of the document"
        Metadata makes this possible.
        """
        # Filter out chunks without embeddings
        ready = [c for c in chunks if c.embedding is not None]
        skipped = len(chunks) - len(ready)

        if skipped:
            print(f"⚠️  Skipped {skipped} chunks without embeddings")

        if not ready:
            print("❌ No chunks with embeddings to store.")
            return 0

        # Prepare the 4 lists ChromaDB expects
        ids = [chunk.chunk_id for chunk in ready]

        embeddings = [chunk.embedding for chunk in ready]

        documents = [chunk.text for chunk in ready]

        metadatas = [
            {
                # Source info — critical for citations in Phase 5
                "source_file":   chunk.source_file,
                "source_path":   chunk.source_path,

                # Position info — useful for context window ordering
                "chunk_index":   chunk.chunk_index,
                "start_char":    chunk.start_char,
                "end_char":      chunk.end_char,

                # Chunking info — useful for debugging
                "strategy":      chunk.strategy,
                "word_count":    chunk.word_count,
                "est_tokens":    chunk.estimated_tokens,
            }
            for chunk in ready
        ]

        # ChromaDB's add() handles duplicates by raising an error.
        # upsert() is safer — it updates if ID exists, inserts if not.
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"✅ Stored {len(ready)} chunks → '{self.collection_name}'")
        return len(ready)


    # ── Search (the core of RAG retrieval) ────────────────────
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_file: str | None = None,   # optional filter by document
    ) -> list[dict]:
        """
        Find the top_k most similar chunks to a query embedding.

        CONCEPT: This is the "R" in RAG — Retrieval.
        We give ChromaDB our query vector, it computes cosine similarity
        against every stored chunk vector, and returns the closest ones.

        Args:
            query_embedding: vector from embed_query(user_question)
            top_k:           how many results to return (typically 3–5)
            source_file:     if set, only search within this document

        Returns:
            List of dicts, each containing:
            - text:       the chunk text
            - score:      similarity score (0–1, higher = more relevant)
            - metadata:   source file, position, etc.
            - chunk_id:   the unique chunk identifier
        """
        # Build optional metadata filter
        where = {"source_file": source_file} if source_file else None

        results = self.collection.query(
            query_embeddings=[query_embedding],  # note: list of queries
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns results nested in lists (supports multi-query)
        # We unwrap the first query's results
        hits = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine "distance" = 1 - similarity
            # Convert back to similarity score (1 = identical, 0 = unrelated)
            similarity_score = 1 - distance

            hits.append({
                "text":      doc,
                "score":     round(similarity_score, 4),
                "metadata":  meta,
                "chunk_id":  meta.get("source_file", "unknown"),
            })

        # Sort by score descending (most relevant first)
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits


    # ── Inspection helpers ─────────────────────────────────────
    def count(self) -> int:
        """How many chunks are stored?"""
        return self.collection.count()

    def peek(self, n: int = 3) -> list[dict]:
        """
        Return the first n stored items — useful for debugging.
        Lets you verify what's actually in the database.
        """
        result = self.collection.peek(limit=n)
        items = []
        for i, (doc_id, doc, meta) in enumerate(zip(
            result["ids"],
            result["documents"],
            result["metadatas"],
        )):
            items.append({
                "id":       doc_id,
                "text":     doc[:150] + "..." if len(doc) > 150 else doc,
                "metadata": meta,
            })
        return items

    def clear(self) -> None:
        """Delete all chunks from this collection. Use with care!"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"🗑️  Cleared collection '{self.collection_name}'")

    def get_stats(self) -> dict:
        """Summary stats about what's stored."""
        count = self.collection.count()
        if count == 0:
            return {"total_chunks": 0}

        sample = self.collection.peek(limit=count)
        source_files = set(m["source_file"] for m in sample["metadatas"])

        return {
            "total_chunks":  count,
            "source_files":  list(source_files),
            "num_documents": len(source_files),
        }