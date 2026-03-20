# src/rag/advanced_pipeline.py
# ─────────────────────────────────────────────────────────────
# Phase 5: Advanced RAG Pipeline
#
# Extends Phase 4's RAGPipeline with:
#   1. Hybrid search (dense + BM25 via HybridRetriever)
#   2. Metadata filtering (scope queries to specific documents)
#   3. Multi-document reasoning (compare/synthesize across docs)
# ─────────────────────────────────────────────────────────────

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from ..embedding.vector_store import VectorStore
from ..chunking.chunk import Chunk
from .hybrid_retriever import HybridRetriever
from .retriever import RetrievedChunk
from .prompt_builder import build_rag_prompt, preview_prompt
from .generator import Generator, RAGResponse

load_dotenv()


# ── Multi-document prompt (different from single-doc RAG prompt) ──
def build_multidoc_prompt(
    question: str,
    chunks_by_source: dict[str, list[RetrievedChunk]],
) -> str:
    """
    Build a prompt specifically for multi-document reasoning.

    WHY a different prompt?
    Single-doc RAG: "answer from this context"
    Multi-doc RAG:  "compare what EACH document says, then synthesize"

    The model needs explicit instruction to:
    - Attribute claims to specific sources
    - Note agreements and contradictions between documents
    - Synthesize a combined answer rather than just listing
    """
    source_blocks = []
    for source_name, chunks in chunks_by_source.items():
        combined_text = "\n".join(c.text for c in chunks)
        source_blocks.append(
            f"=== Document: {source_name} ===\n{combined_text}"
        )

    all_context = "\n\n".join(source_blocks)
    num_docs = len(chunks_by_source)

    return f"""You are a document intelligence assistant analyzing {num_docs} document(s).

TASK: Answer the question by synthesizing information across all provided documents.

RULES:
- Attribute each claim to its source document using [Doc: filename] notation
- If documents agree on a point, state that explicitly
- If documents contradict each other, highlight the contradiction
- If only one document addresses the question, say so
- Do not add information from outside the provided documents

DOCUMENTS:
{all_context}

QUESTION: {question}

SYNTHESIZED ANSWER:"""


class AdvancedRAGPipeline:
    """
    Production-grade RAG pipeline with hybrid search,
    metadata filtering, and multi-document reasoning.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        chunks: list[Chunk],               # needed for BM25 index
        top_k: int = 5,
        model: str = "gemini-2.0-flash",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        # Hybrid retriever combines dense + BM25
        self.retriever = HybridRetriever(
            vector_store=vector_store,
            chunks=chunks,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        self.generator = Generator(model=model)
        self.vector_store = vector_store
        self.chunks = chunks
        self.top_k = top_k

        print(f"\n🚀 Advanced RAG Pipeline ready")
        print(f"   Model         : {model}")
        print(f"   Search mode   : Hybrid (dense={dense_weight} + BM25={sparse_weight})")
        print(f"   Top-k         : {top_k}")
        print(f"   Chunks indexed: {len(chunks)}\n")

    # ── 1. Standard hybrid ask ─────────────────────────────────
    def ask(
        self,
        question: str,
        source_file: str | None = None,
        show_prompt: bool = False,
    ) -> RAGResponse:
        """
        Ask a question using hybrid retrieval.
        Optionally scope to a single document with source_file.
        """
        scope = f" [scoped to: {source_file}]" if source_file else ""
        print(f"\n🔍 Hybrid search{scope}: \"{question}\"")

        chunks = self.retriever.retrieve(question, source_file=source_file)

        if not chunks:
            print("  ⚠️  No relevant chunks found")
        else:
            print(f"  ✅ Retrieved {len(chunks)} chunk(s) via hybrid search:")
            for c in chunks:
                print(f"     RRF {c.score:.6f} | {c.source_file} | chunk #{c.chunk_index}")

        prompt = build_rag_prompt(question, chunks)
        if show_prompt:
            preview_prompt(prompt)

        print(f"\n🤖 Generating answer...")
        response = self.generator.generate(
            prompt=prompt,
            question=question,
            retrieved_chunks=chunks,
        )
        response.display()
        return response

    # ── 2. Metadata-filtered ask ───────────────────────────────
    def ask_document(
        self,
        question: str,
        source_file: str,
    ) -> RAGResponse:
        """
        METADATA FILTERING: scope retrieval to a specific document.

        WHY is this useful?
        Multi-document setups often need targeted queries:
        "What does the Q3 report say about revenue?"
        → should only search Q3_report.pdf, not all documents
        """
        print(f"\n📄 Document-scoped query")
        print(f"   Source  : {source_file}")
        print(f"   Question: \"{question}\"")
        return self.ask(question, source_file=source_file)

    # ── 3. Multi-document reasoning ────────────────────────────
    def ask_across_documents(
        self,
        question: str,
        source_files: list[str] | None = None,
    ) -> RAGResponse:
        """
        MULTI-DOCUMENT REASONING: retrieve from each document
        separately, then synthesize a combined answer.

        WHY retrieve separately per document?
        If you retrieve globally, one document might dominate the
        top-k results. Per-document retrieval guarantees each source
        contributes context, enabling true cross-document synthesis.

        Args:
            question:     the question to answer
            source_files: list of filenames to reason across.
                          If None, uses all documents in the store.
        """
        print(f"\n📚 Multi-document reasoning: \"{question}\"")

        # Determine which documents to reason across
        if source_files is None:
            stats = self.vector_store.get_stats()
            source_files = stats.get("source_files", [])

        if not source_files:
            print("  ⚠️  No documents found in vector store")
            return RAGResponse(question=question, answer="No documents available.")

        print(f"   Reasoning across {len(source_files)} document(s): {source_files}")

        # Retrieve top chunks from EACH document separately
        chunks_by_source: dict[str, list[RetrievedChunk]] = {}
        for src in source_files:
            doc_chunks = self.retriever.retrieve(question, source_file=src)
            if doc_chunks:
                chunks_by_source[src] = doc_chunks[:3]  # top 3 per doc
                print(f"   ✅ {src}: {len(doc_chunks)} chunks retrieved")
            else:
                print(f"   ⚠️  {src}: no relevant chunks found")

        if not chunks_by_source:
            return RAGResponse(
                question=question,
                answer="No relevant information found across the provided documents."
            )

        # Build multi-doc prompt and generate
        prompt = build_multidoc_prompt(question, chunks_by_source)
        all_chunks = [c for chunks in chunks_by_source.values() for c in chunks]

        print(f"\n🤖 Synthesizing answer across {len(chunks_by_source)} document(s)...")
        response = self.generator.generate(
            prompt=prompt,
            question=question,
            retrieved_chunks=all_chunks,
        )
        response.display()
        return response