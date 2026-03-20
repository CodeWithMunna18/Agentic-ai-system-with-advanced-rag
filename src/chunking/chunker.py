# src/chunking/chunker.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: The Chunker is the orchestrator for Phase 2 —
# just like Loader was the orchestrator for Phase 1.
#
# It accepts a Document (or list of Documents) and a strategy,
# then returns a flat list of Chunks ready for Phase 3 (embedding).
#
# It also provides comparison utilities so you can SEE the
# difference between strategies on your own documents.
# ─────────────────────────────────────────────────────────────

from ..ingestion.loader import Document
from .chunk import Chunk
from .strategies import fixed_size_chunking, sentence_chunking, paragraph_chunking


# ── Strategy registry (same dispatch pattern as extractors.py) ─
STRATEGIES = {
    "fixed":     fixed_size_chunking,
    "sentence":  sentence_chunking,
    "paragraph": paragraph_chunking,
}


def chunk_document(
    document: Document,
    strategy: str = "sentence",
    **kwargs                        # passed directly to the strategy function
) -> list[Chunk]:
    """
    Chunk a single Document using the specified strategy.

    Args:
        document:  a valid Document from Phase 1
        strategy:  one of "fixed", "sentence", "paragraph"
        **kwargs:  strategy-specific settings e.g. chunk_size=500, overlap=100

    Returns:
        List of Chunk objects ready for embedding in Phase 3
    """
    if not document.is_valid:
        print(f"⚠️  Skipping invalid document: {document.file_name} — {document.error}")
        return []

    if strategy not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {available}")

    strategy_fn = STRATEGIES[strategy]
    chunks = strategy_fn(document, **kwargs)

    print(f"✂️  '{document.file_name}' → {len(chunks)} chunks (strategy='{strategy}')")
    return chunks


def chunk_documents(
    documents: list[Document],
    strategy: str = "sentence",
    **kwargs
) -> list[Chunk]:
    """
    Chunk a list of Documents. Returns ALL chunks from ALL documents
    as a single flat list — this is what Phase 3 will receive.
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, strategy=strategy, **kwargs)
        all_chunks.extend(chunks)

    print(f"\n📦 Total chunks produced: {len(all_chunks)}")
    return all_chunks


# ── Comparison utility ─────────────────────────────────────────

def compare_strategies(document: Document) -> None:
    """
    Run all three strategies on the same document and print a
    side-by-side comparison. Use this to understand tradeoffs.
    """
    print("\n" + "═" * 60)
    print(f"  CHUNKING STRATEGY COMPARISON")
    print(f"  Document: {document.file_name}")
    print(f"  Total words: {document.stats.get('words', 0)}")
    print(f"  Estimated tokens: {document.stats.get('estimated_tokens', 0)}")
    print("═" * 60)

    configs = {
        "fixed":     {"chunk_size": 500, "overlap": 100},
        "sentence":  {"max_tokens": 200, "overlap_sentences": 1},
        "paragraph": {"max_tokens": 300, "overlap_paragraphs": 1},
    }

    results = {}
    for name, kwargs in configs.items():
        chunks = STRATEGIES[name](document, **kwargs)
        results[name] = chunks

        if not chunks:
            print(f"\n  [{name.upper()}] — No chunks produced")
            continue

        sizes = [c.word_count for c in chunks]
        avg = sum(sizes) / len(sizes)
        min_s = min(sizes)
        max_s = max(sizes)

        print(f"\n  [{name.upper()}]")
        print(f"  ├─ Total chunks : {len(chunks)}")
        print(f"  ├─ Avg words    : {avg:.0f}")
        print(f"  ├─ Min words    : {min_s}")
        print(f"  ├─ Max words    : {max_s}")
        print(f"  └─ First chunk preview:")
        print(f"     \"{chunks[0].preview(120)}\"")

    print("\n" + "═" * 60)
    print("  💡 RECOMMENDATION:")
    print("  • 'sentence' is best for most RAG use cases")
    print("  • 'paragraph' if document has strong structure")
    print("  • 'fixed' only if speed > quality matters")
    print("═" * 60)

    return results