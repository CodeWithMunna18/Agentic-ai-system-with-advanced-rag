# src/rag/pipeline.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: The RAGPipeline is the top-level orchestrator.
# It wires together ALL phases into a single clean interface:
#
#   pipeline.ask("your question") → RAGResponse
#
# This is what your API (Phase 6) and UI (Phase 7) will call.
# By the time we get there, this single method is the only
# thing they need to know about.
#
# FULL FLOW inside pipeline.ask():
#   1. Retriever  → embed question + search ChromaDB
#   2. PromptBuilder → format chunks into RAG prompt
#   3. Generator  → send prompt to Gemini, get answer
#   4. Return     → RAGResponse with answer + citations
# ─────────────────────────────────────────────────────────────

from ..embedding.vector_store import VectorStore
from .retriever import Retriever
from .prompt_builder import build_rag_prompt, preview_prompt
from .generator import Generator, RAGResponse


class RAGPipeline:
    """
    The complete Retrieval-Augmented Generation pipeline.
    Initialize once, call .ask() as many times as you want.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        relevance_threshold: float = 0.5,
        model: str = "gemini-1.5-flash",
    ):
        """
        Args:
            vector_store:        ChromaDB store from Phase 3
            top_k:               max chunks to retrieve per query
            relevance_threshold: min similarity score (0.0–1.0)
            model:               Gemini model for generation
        """
        self.retriever = Retriever(
            vector_store=vector_store,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
        )
        self.generator = Generator(model=model)
        self.top_k = top_k

        print(f"🚀 RAG Pipeline ready")
        print(f"   Model      : {model}")
        print(f"   Top-k      : {top_k}")
        print(f"   Min score  : {relevance_threshold}")
        print(f"   Store size : {vector_store.count()} chunks\n")

    def ask(
        self,
        question: str,
        source_file: str | None = None,
        show_prompt: bool = False,
        show_sources: bool = True,
    ) -> RAGResponse:
        """
        Ask a question and get a grounded answer from your documents.

        Args:
            question:     the question to answer
            source_file:  if set, only search within this document
            show_prompt:  print the full prompt (useful for learning/debugging)
            show_sources: print source chunks alongside the answer

        Returns:
            RAGResponse with .answer, .sources, and .display() method
        """
        print(f"\n🔍 Retrieving context for: \"{question}\"")

        # ── Step 1: Retrieve relevant chunks ──────────────────
        chunks = self.retriever.retrieve(question, source_file=source_file)

        if not chunks:
            print("  ⚠️  No relevant chunks found above threshold")
        else:
            print(f"  ✅ Retrieved {len(chunks)} chunk(s):")
            for c in chunks:
                print(f"     Score {c.score:.4f} | {c.source_file} | chunk #{c.chunk_index}")

        # ── Step 2: Build the RAG prompt ───────────────────────
        prompt = build_rag_prompt(question, chunks)

        # Optionally show the full prompt for learning purposes
        if show_prompt:
            preview_prompt(prompt)

        # ── Step 3: Generate answer with Gemini ────────────────
        print(f"\n🤖 Generating answer with Gemini...")
        response = self.generator.generate(
            prompt=prompt,
            question=question,
            retrieved_chunks=chunks,
        )

        # ── Step 4: Display and return ─────────────────────────
        if show_sources:
            response.display()

        return response