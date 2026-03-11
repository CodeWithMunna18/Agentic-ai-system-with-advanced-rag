# src/chunking/strategies.py
# ─────────────────────────────────────────────────────────────
# THREE CHUNKING STRATEGIES — understand each before using them
#
# Strategy 1: FIXED SIZE
#   Split by character count. Simple, predictable.
#   ✅ Fast, works on any text
#   ❌ Can cut mid-sentence, losing meaning
#   Best for: structured data, logs, code
#
# Strategy 2: SENTENCE-AWARE
#   Split on sentence boundaries (periods, ?, !).
#   ✅ Each chunk is grammatically complete
#   ❌ Sentence lengths vary wildly → uneven chunk sizes
#   Best for: articles, reports, books
#
# Strategy 3: PARAGRAPH-AWARE
#   Split on double newlines (natural paragraph breaks).
#   ✅ Preserves logical document structure
#   ❌ Paragraphs can be very long or very short
#   Best for: PDFs, DOCX files, structured documents
#
# WHICH TO USE?
#   In practice, sentence-aware is the best default for RAG.
#   We'll make it easy to compare all three in main.py.
# ─────────────────────────────────────────────────────────────

import re
import uuid
from typing import Callable
from ..ingestion.loader import Document
from .chunk import Chunk


# ══════════════════════════════════════════════════════════════
# STRATEGY 1 — FIXED SIZE CHUNKING
# ══════════════════════════════════════════════════════════════

def fixed_size_chunking(
    document: Document,
    chunk_size: int = 500,       # target size in characters
    overlap: int = 100,          # overlap in characters
) -> list[Chunk]:
    """
    CONCEPT: Slide a window of `chunk_size` characters across the text.
    After each chunk, step back `overlap` characters before starting the next.

    Visual example (chunk_size=10, overlap=3):
    Text:    "Hello world this is RAG"
    Chunk 1: "Hello worl"          (chars 0–10)
    Chunk 2:       "rld this i"    (chars 7–17)  ← stepped back 3
    Chunk 3:             "s is RAG" (chars 14–22)

    WHY overlap? The word "world" near the boundary appears in BOTH
    chunk 1 and chunk 2, so a search for "world" will find relevant context
    in either chunk.
    """
    text = document.clean_text
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Don't go past the end of the document
        end = min(end, len(text))

        chunk_text = text[start:end].strip()

        # Skip chunks that are just whitespace after stripping
        if chunk_text:
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{document.file_name}::fixed_{chunk_index:04d}",
                source_file=document.file_name,
                source_path=document.file_path,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
                strategy="fixed",
                chunk_size=chunk_size,
                overlap_size=overlap,
            )
            chunks.append(chunk)
            chunk_index += 1

        # Move forward by (chunk_size - overlap)
        # This creates the overlapping window effect
        step = chunk_size - overlap
        start += step

        # Safety: if overlap >= chunk_size we'd loop forever
        if step <= 0:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )

    return chunks


# ══════════════════════════════════════════════════════════════
# STRATEGY 2 — SENTENCE-AWARE CHUNKING
# ══════════════════════════════════════════════════════════════

def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.

    WHY not just split on '.'?
    Because "Dr. Smith visited the U.S.A. in 2024." has 4 dots
    but is ONE sentence. We use a smarter pattern that looks for
    sentence-ending punctuation followed by a capital letter or end of string.
    """
    # Split on . ? ! followed by whitespace + capital letter (or end of text)
    # The lookahead (?=[A-Z]|\s*$) avoids splitting "Dr. Smith" or "U.S.A."
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)

    # Remove empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]


def sentence_chunking(
    document: Document,
    max_tokens: int = 200,         # approximate max tokens per chunk (1 token ≈ 4 chars)
    overlap_sentences: int = 1,    # how many sentences to repeat in the next chunk
) -> list[Chunk]:
    """
    CONCEPT: Build chunks sentence by sentence.
    Keep adding sentences to the current chunk until we'd exceed max_tokens.
    Then start a new chunk, but "carry over" the last N sentences as overlap.

    Example (max_tokens=20, overlap_sentences=1):
    Sentences: [S1, S2, S3, S4, S5]
    Chunk 1:   S1 + S2 + S3    (hits token limit)
    Chunk 2:        S3 + S4 + S5  ← S3 is carried over as overlap
    Chunk 3:             S5 + ...
    """
    sentences = _split_into_sentences(document.clean_text)

    if not sentences:
        return []

    max_chars = max_tokens * 4   # convert token estimate to characters
    chunks = []
    chunk_index = 0
    i = 0                        # current sentence index

    while i < len(sentences):
        current_sentences = []
        current_length = 0
        start_sentence = i

        # Fill up the current chunk with sentences
        while i < len(sentences):
            sentence = sentences[i]
            sentence_len = len(sentence)

            # If adding this sentence exceeds limit AND we already have content,
            # stop here — don't add partial sentences
            if current_length + sentence_len > max_chars and current_sentences:
                break

            current_sentences.append(sentence)
            current_length += sentence_len + 1  # +1 for space
            i += 1

        # Build the chunk text
        chunk_text = " ".join(current_sentences).strip()

        # Find character positions in the original text
        start_char = document.clean_text.find(current_sentences[0]) if current_sentences else 0
        end_char = start_char + len(chunk_text)

        if chunk_text:
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{document.file_name}::sent_{chunk_index:04d}",
                source_file=document.file_name,
                source_path=document.file_path,
                chunk_index=chunk_index,
                start_char=max(0, start_char),
                end_char=end_char,
                strategy="sentence",
                chunk_size=max_tokens,
                overlap_size=overlap_sentences,
            )
            chunks.append(chunk)
            chunk_index += 1

        # Step back `overlap_sentences` to create overlap
        # We move i back so the next chunk starts from overlap_sentences ago
        if overlap_sentences > 0 and i < len(sentences):
            i = max(start_sentence + 1, i - overlap_sentences)

    return chunks


# ══════════════════════════════════════════════════════════════
# STRATEGY 3 — PARAGRAPH-AWARE CHUNKING
# ══════════════════════════════════════════════════════════════

def paragraph_chunking(
    document: Document,
    max_tokens: int = 300,          # max tokens per chunk
    overlap_paragraphs: int = 1,    # how many paragraphs to carry over
) -> list[Chunk]:
    """
    CONCEPT: Use natural paragraph breaks (double newlines) as boundaries.
    This respects the author's intended logical structure.

    After splitting on paragraphs, we merge small paragraphs together
    (up to max_tokens) and carry over the last N paragraphs as overlap.

    WHY merge small paragraphs?
    A paragraph like "See Figure 3." is 3 words — useless as a standalone chunk.
    We merge it with adjacent paragraphs to create meaningful chunk sizes.
    """
    # Split on double newlines (paragraph boundaries)
    raw_paragraphs = re.split(r'\n\n+', document.clean_text)

    # Clean and filter empty paragraphs
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    if not paragraphs:
        return []

    max_chars = max_tokens * 4
    chunks = []
    chunk_index = 0
    i = 0

    while i < len(paragraphs):
        current_paragraphs = []
        current_length = 0
        start_para = i

        # Merge paragraphs until we hit the size limit
        while i < len(paragraphs):
            para = paragraphs[i]
            para_len = len(para)

            if current_length + para_len > max_chars and current_paragraphs:
                break

            current_paragraphs.append(para)
            current_length += para_len
            i += 1

        chunk_text = "\n\n".join(current_paragraphs).strip()

        # Approximate char position
        start_char = document.clean_text.find(current_paragraphs[0]) if current_paragraphs else 0
        end_char = start_char + len(chunk_text)

        if chunk_text:
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{document.file_name}::para_{chunk_index:04d}",
                source_file=document.file_name,
                source_path=document.file_path,
                chunk_index=chunk_index,
                start_char=max(0, start_char),
                end_char=end_char,
                strategy="paragraph",
                chunk_size=max_tokens,
                overlap_size=overlap_paragraphs,
            )
            chunks.append(chunk)
            chunk_index += 1

        # Overlap: carry back N paragraphs
        if overlap_paragraphs > 0 and i < len(paragraphs):
            i = max(start_para + 1, i - overlap_paragraphs)

    return chunks