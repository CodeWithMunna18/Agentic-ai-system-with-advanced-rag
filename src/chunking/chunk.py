# src/chunking/chunk.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: A Chunk is a small, self-contained piece of a document.
#
# Think of it like index cards in a library:
#   - Document = the full book
#   - Chunk     = one index card summarizing a section
#
# Each chunk carries:
#   1. The text itself
#   2. WHERE it came from (source doc, position)
#   3. Metadata useful for filtering later (Phase 5)
#
# We design this dataclass now so Phases 3–6 can build on it
# without changing the structure.
# ─────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """
    A single chunk of text extracted from a Document.
    This is the atomic unit that gets embedded and stored in the vector DB.
    """

    # ── Core content ──────────────────────────────────────────
    text: str                        # the actual chunk text
    chunk_id: str                    # unique ID e.g. "research_paper.pdf::chunk_003"

    # ── Source tracking (crucial for RAG citations) ───────────
    source_file: str                 # e.g. "research_paper.pdf"
    source_path: str                 # full file path

    # ── Position info (helps with context & debugging) ────────
    chunk_index: int                 # 0-based position within the document
    start_char: int                  # character offset where chunk starts in original doc
    end_char: int                    # character offset where chunk ends

    # ── Chunking strategy metadata ────────────────────────────
    strategy: str = "unknown"        # "fixed", "sentence", "paragraph"
    chunk_size: int = 0              # target chunk size used
    overlap_size: int = 0            # overlap size used

    # ── Stats (computed after creation) ───────────────────────
    word_count: int = 0
    char_count: int = 0
    estimated_tokens: int = 0

    # ── Optional: will be filled in Phase 3 ──────────────────
    embedding: Optional[list[float]] = field(default=None, repr=False)

    def __post_init__(self):
        """Auto-compute stats when chunk is created."""
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())
        self.estimated_tokens = self.char_count // 4

    def __repr__(self):
        return (
            f"Chunk(id='{self.chunk_id}', "
            f"words={self.word_count}, "
            f"tokens=~{self.estimated_tokens}, "
            f"strategy='{self.strategy}')"
        )

    def preview(self, chars: int = 100) -> str:
        """Returns a short preview of the chunk text — useful for debugging."""
        if len(self.text) <= chars:
            return self.text
        return self.text[:chars] + "..."