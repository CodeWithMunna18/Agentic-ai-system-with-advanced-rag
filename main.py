# main.py
# ─────────────────────────────────────────────────────────────
# Runs Phase 1 (ingestion) → Phase 2 (chunking) in sequence.
# Each phase feeds its output directly into the next.
# ─────────────────────────────────────────────────────────────

import pathlib
from src.ingestion import load_document, load_documents_from_folder
from src.chunking import chunk_document, chunk_documents, compare_strategies
from src.embedding import init_gemini, embed_query, embed_chunks, cosine_similarity, VectorStore
from src.rag import RAGPipeline, AdvancedRAGPipeline


# ── Ensure sample documents exist ─────────────────────────────
def create_sample_documents():
    docs_dir = pathlib.Path("documents")
    docs_dir.mkdir(exist_ok=True)

    # Document 1 — RAG concepts (used since Phase 1)
    sample1 = docs_dir / "rag_intro.txt"
    if not sample1.exists():
        sample1.write_text(
            "What is Retrieval-Augmented Generation?\n\n"
            "Retrieval-Augmented Generation, or RAG, is an AI framework that enhances "
            "large language models by giving them access to external knowledge. "
            "Instead of relying solely on information learned during training, a RAG "
            "system first retrieves relevant documents from a knowledge base, then uses "
            "those documents as context when generating a response.\n\n"
            "Why RAG Matters\n\n"
            "Large language models are powerful but have significant limitations. "
            "They can hallucinate — confidently stating facts that are simply wrong. "
            "Their knowledge has a cutoff date, making them unaware of recent events. "
            "They cannot access private or proprietary data unless it was in training.\n\n"
            "RAG solves all three problems. By retrieving from a live knowledge base, "
            "the model always has access to current, accurate, and domain-specific "
            "information at query time.\n\n"
            "How RAG Works Step by Step\n\n"
            "The process begins with document ingestion. Raw documents are loaded, "
            "cleaned, and split into smaller chunks. Each chunk is then converted into "
            "a vector embedding — a list of numbers that captures the semantic meaning "
            "of the text. These embeddings are stored in a vector database.\n\n"
            "When a user asks a question, the question is also converted to an embedding. "
            "The system then searches the vector database for chunks whose embeddings are "
            "mathematically closest to the question embedding. This is called similarity search.\n\n"
            "The top matching chunks are retrieved and injected into the prompt sent to "
            "the language model. The model reads this context and generates an answer "
            "grounded in the retrieved information — not just its training data.\n\n"
            "Key Components of a RAG System\n\n"
            "A complete RAG system has five core components. First, the document store "
            "holds the raw source documents. Second, the chunker splits documents into "
            "retrievable pieces. Third, the embedding model converts text to vectors. "
            "Fourth, the vector database stores and indexes those vectors for fast search. "
            "Fifth, the language model generates the final answer using retrieved context.\n\n"
            "Chunking is the Most Underrated Step\n\n"
            "Most developers focus on the language model and overlook chunking. "
            "But chunking strategy directly determines retrieval quality. "
            "Chunks that are too large contain noise that confuses the embedding model. "
            "Chunks that are too small lose necessary context for answering questions. "
            "The overlap between adjacent chunks ensures that ideas near chunk boundaries "
            "are never lost. Getting chunking right is the difference between a RAG "
            "system that works and one that merely appears to work.",
            encoding="utf-8"
        )
        print(f"📝 Created sample: documents/rag_intro.txt")

    # Document 2 — Vector databases (new, for multi-doc demo)
    sample2 = docs_dir / "vector_databases.txt"
    if not sample2.exists():
        sample2.write_text(
            "Understanding Vector Databases\n\n"
            "A vector database is a specialized storage system designed to store, "
            "index, and search high-dimensional vectors efficiently. Unlike traditional "
            "databases that search by exact value or range, vector databases search by "
            "similarity — finding vectors that are mathematically close to a query vector.\n\n"
            "Why Vector Databases Exist\n\n"
            "Modern AI models represent everything as vectors: text, images, audio, "
            "and video. A sentence like 'the cat sat on the mat' becomes a list of "
            "768 numbers after passing through an embedding model. Finding similar "
            "sentences means finding vectors with small angular distance — a problem "
            "traditional SQL databases handle very poorly.\n\n"
            "Vector databases use specialized index structures like HNSW (Hierarchical "
            "Navigable Small World graphs) to make this search fast even across millions "
            "of vectors. HNSW builds a multi-layer graph where each node connects to its "
            "nearest neighbors, enabling approximate nearest neighbor search in "
            "milliseconds.\n\n"
            "ChromaDB vs Other Vector Databases\n\n"
            "ChromaDB is an open-source, embedded vector database designed for simplicity. "
            "It runs in-process with your Python application — no separate server needed. "
            "ChromaDB is ideal for prototypes, local development, and small-to-medium "
            "datasets up to a few million vectors.\n\n"
            "Pinecone is a managed cloud vector database offering serverless scaling. "
            "It handles billions of vectors with sub-10ms query latency and requires "
            "no infrastructure management. Pinecone is the production choice when you "
            "need scale beyond what local solutions offer.\n\n"
            "Weaviate and Qdrant are open-source alternatives that can run self-hosted "
            "or in the cloud. Both support hybrid search (vector + keyword) natively "
            "and offer more advanced filtering than ChromaDB.\n\n"
            "Similarity Metrics\n\n"
            "Vector databases support multiple distance metrics. Cosine similarity "
            "measures the angle between vectors — the standard choice for text embeddings "
            "because it is invariant to vector magnitude. Euclidean distance measures "
            "absolute distance in space — better for image embeddings. Dot product "
            "is the fastest to compute and works well when vectors are normalized.\n\n"
            "The Role of Vector Databases in RAG\n\n"
            "In a RAG system, the vector database is the memory. Every document chunk "
            "is stored as a vector. At query time, the user's question is embedded and "
            "the database returns the chunks whose vectors are closest. The quality of "
            "this retrieval step determines the quality of the final answer. A slow or "
            "inaccurate vector database creates a bottleneck that no language model "
            "can compensate for.",
            encoding="utf-8"
        )
        print(f"📝 Created sample: documents/vector_databases.txt\n")


# ══════════════════════════════════════════════════════════════
# PHASE 1 — Document Ingestion
# ══════════════════════════════════════════════════════════════
def run_phase1() -> list:
    print("=" * 60)
    print("  PHASE 1 — Document Ingestion")
    print("=" * 60)

    documents = load_documents_from_folder("documents/")
    valid = [d for d in documents if d.is_valid]

    if valid:
        print("\n── Sample stats from first valid document ──────────────")
        d = valid[0]
        for k, v in d.stats.items():
            print(f"   {k:<22}: {v}")

    return valid


# ══════════════════════════════════════════════════════════════
# PHASE 2 — Chunking
# ══════════════════════════════════════════════════════════════
def run_phase2(documents: list) -> list:
    print("\n" + "=" * 60)
    print("  PHASE 2 — Chunking")
    print("=" * 60)

    if not documents:
        print("❌ No valid documents from Phase 1. Cannot chunk.")
        return []

    # ── DEMO A: Compare all strategies on the first document ──
    print("\n── Demo A: Strategy Comparison ─────────────────────────")
    compare_strategies(documents[0])

    # ── DEMO B: Chunk all documents with sentence strategy ────
    print("\n── Demo B: Sentence Chunking (recommended for RAG) ─────")
    all_chunks = chunk_documents(
        documents,
        strategy="sentence",
        max_tokens=200,
        overlap_sentences=1,
    )

    # ── DEMO C: Inspect individual chunks ─────────────────────
    print("\n── Demo C: Inspect first 3 chunks ──────────────────────")
    for chunk in all_chunks[:3]:
        print(f"\n  {chunk}")
        print(f"  Position : chars {chunk.start_char}–{chunk.end_char}")
        print(f"  Preview  : \"{chunk.preview(120)}\"")

    # ── DEMO D: Show the overlap between chunk 0 and chunk 1 ──
    if len(all_chunks) >= 2:
        print("\n── Demo D: Overlap Visualization ────────────────────────")
        print("  Last 80 chars of Chunk 0:")
        print(f"  \"...{all_chunks[0].text[-80:]}\"")
        print("\n  First 80 chars of Chunk 1:")
        print(f"  \"{all_chunks[1].text[:80]}...\"")
        print("\n  ☝️  Notice the repeated text — that's the overlap working!")

    return all_chunks


# ══════════════════════════════════════════════════════════════
# PHASE 3 — Embeddings + Vector Store
# ══════════════════════════════════════════════════════════════
def run_phase3(chunks: list) -> VectorStore:
    print("\n" + "=" * 60)
    print("  PHASE 3 — Gemini Embeddings + ChromaDB")
    print("=" * 60)

    if not chunks:
        print("❌ No chunks from Phase 2. Cannot embed.")
        return None

    # Step 1: Initialize Gemini
    init_gemini()

    # Step 2: Embed all chunks via Gemini API
    print(f"\n── Step 1: Embed {len(chunks)} chunks ──────────────────────")
    embedded_chunks = embed_chunks(chunks, batch_size=20)

    # Step 3: Inspect one embedding so you understand what it is
    sample = embedded_chunks[0]
    print(f"\n── Step 2: What does an embedding look like? ───────────")
    print(f"  Chunk text : \"{sample.preview(80)}\"")
    print(f"  Vector dims: {len(sample.embedding)}")
    print(f"  First 8 values: {[round(x, 4) for x in sample.embedding[:8]]}")
    print(f"  → Each number encodes a dimension of semantic meaning")

    # Step 4: Demonstrate cosine similarity BEFORE storing
    print(f"\n── Step 3: Cosine Similarity Demo ──────────────────────")
    if len(embedded_chunks) >= 2:
        score_adjacent = cosine_similarity(
            embedded_chunks[0].embedding,
            embedded_chunks[1].embedding
        )
        score_far = cosine_similarity(
            embedded_chunks[0].embedding,
            embedded_chunks[-1].embedding
        )
        print(f"  Chunk 0 vs Chunk 1 (adjacent): {score_adjacent:.4f}  ← should be high")
        print(f"  Chunk 0 vs Chunk {len(embedded_chunks)-1} (distant): {score_far:.4f}  ← lower")
        print(f"  → Adjacent chunks share overlap text → higher similarity ✅")

    # Step 5: Store in ChromaDB
    print(f"\n── Step 4: Store in ChromaDB ────────────────────────────")
    store = VectorStore(
        persist_directory="./chroma_db",
        collection_name="documents",
    )
    store.add_chunks(embedded_chunks)

    # Step 6: Inspect what's stored
    print(f"\n── Step 5: Verify storage ───────────────────────────────")
    stats = store.get_stats()
    print(f"  Total chunks stored : {stats['total_chunks']}")
    print(f"  Source documents    : {stats['source_files']}")

    # Step 7: Run a test search to prove retrieval works
    print(f"\n── Step 6: Test Similarity Search ──────────────────────")
    test_question = "What is RAG and how does it work?"
    print(f"  Query: \"{test_question}\"")

    query_vec = embed_query(test_question)
    results = store.search(query_vec, top_k=3)

    print(f"\n  Top 3 most relevant chunks:")
    for i, hit in enumerate(results):
        print(f"\n  [{i+1}] Score: {hit['score']:.4f}  |  "
              f"Source: {hit['metadata']['source_file']}")
        print(f"       \"{hit['text'][:120]}...\"")

    print(f"\n  ☝️  Higher score = more semantically similar to the query")

    return store


# ══════════════════════════════════════════════════════════════
# PHASE 4 — Full RAG Q&A Pipeline
# ══════════════════════════════════════════════════════════════
def run_phase4(store: VectorStore) -> None:
    print("\n" + "=" * 60)
    print("  PHASE 4 — Retrieval + Generation (Full RAG Loop)")
    print("=" * 60)

    if store is None or store.count() == 0:
        print("❌ No chunks in vector store. Run Phase 3 first.")
        return

    # Initialize the complete pipeline
    pipeline = RAGPipeline(
        vector_store=store,
        top_k=5,
        relevance_threshold=0.5,
        model="gemini-2.5-flash",
    )

    # ── Demo A: Basic Q&A ─────────────────────────────────────
    print("── Demo A: Basic Q&A ────────────────────────────────")
    pipeline.ask("What is RAG and why does it matter?")

    # ── Demo B: Specific detail question ──────────────────────
    print("\n── Demo B: Specific detail question ─────────────────")
    pipeline.ask("What are the five core components of a RAG system?")

    # ── Demo C: Show the prompt (learning mode) ───────────────
    print("\n── Demo C: Full prompt visibility (learning mode) ───")
    pipeline.ask(
        "Why is chunking so important in RAG?",
        show_prompt=True,    # ← prints exactly what Gemini receives
    )

    # ── Demo D: Out-of-scope question ─────────────────────────
    print("\n── Demo D: Out-of-scope question (hallucination test)")
    pipeline.ask(
        "What is the capital of France?",   # not in our documents
    )


# ══════════════════════════════════════════════════════════════
# PHASE 5 — Advanced RAG
# ══════════════════════════════════════════════════════════════
def run_phase5(store: VectorStore, chunks: list) -> None:
    print("\n" + "=" * 60)
    print("  PHASE 5 — Advanced RAG (Hybrid + Filtering + Multi-doc)")
    print("=" * 60)

    pipeline = AdvancedRAGPipeline(
        vector_store=store,
        chunks=chunks,
        top_k=5,
        model="gemini-2.5-flash",
        dense_weight=0.6,
        sparse_weight=0.4,
    )

    # ── Demo A: Hybrid search (keyword that pure dense would miss) ─
    print("\n── Demo A: Hybrid search — keyword-heavy query ──────────")
    print("   (BM25 shines here — exact term 'BM25' in the query)")
    pipeline.ask("What does BM25 stand for and how does it score documents?")

    # ── Demo B: Metadata filtering — scope to one document ────────
    print("\n── Demo B: Metadata filtering — scoped to one document ──")
    pipeline.ask_document(
        question="What are the limitations of this approach?",
        source_file="rag_intro.txt",
    )

    # ── Demo C: Multi-document reasoning ──────────────────────────
    print("\n── Demo C: Multi-document reasoning ─────────────────────")
    print("   (synthesizes across rag_intro.txt + vector_databases.txt)")
    pipeline.ask_across_documents(
        question="How do vector databases and RAG work together, "
                 "and what role does similarity search play in both?",
    )

    # ── Demo D: Compare hybrid vs dense on same question ──────────
    print("\n── Demo D: Hybrid vs Dense comparison ───────────────────")
    question = "What is HNSW and why is it used?"
    print(f"   Question: \"{question}\"")
    print(f"\n   [Hybrid retrieval]")
    hybrid_chunks = pipeline.retriever.retrieve(question)
    print(f"   Sources: {[c.source_file for c in hybrid_chunks]}")
    print(f"   Scores:  {[c.score for c in hybrid_chunks]}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    create_sample_documents()

    documents = run_phase1()
    chunks    = run_phase2(documents)
    store     = run_phase3(chunks)
    run_phase4(store)
    run_phase5(store, chunks)

    print("\n" + "=" * 60)
    print("  ✅ Phase 5 Complete — Production-grade RAG!")
    print("  Next: Phase 6 — FastAPI + Streamlit UI")
    print("=" * 60)