# cleaner.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: Raw extracted text is messy — extra spaces, weird
# characters, repeated newlines, headers/footers noise.
#
# WHY does this matter for RAG?
# When we split text into chunks later, dirty text creates chunks
# like "   \n\n   Page 4   \n\n" which waste space in the vector
# store and confuse the embedding model. Clean input → better search.
# ─────────────────────────────────────────────────────────────

import re


def clean_text(raw_text: str) -> str:
    """
    Runs the raw text through a cleaning pipeline.
    Each step is a separate function so you can turn them on/off
    and understand exactly what each one does.
    """
    text = raw_text

    # Step 1: Normalize line endings
    # Windows uses \r\n, Mac used to use \r — standardize to \n
    text = _normalize_line_endings(text)

    # Step 2: Remove non-printable / control characters
    # PDFs sometimes embed hidden control chars that break tokenizers
    text = _remove_control_characters(text)

    # Step 3: Collapse excessive whitespace within a line
    # "hello   world" → "hello world"
    text = _collapse_spaces(text)

    # Step 4: Collapse excessive blank lines
    # 5 empty lines in a row → 2 empty lines (preserve paragraph breaks)
    text = _collapse_blank_lines(text)

    # Step 5: Strip leading/trailing whitespace from the whole doc
    text = text.strip()

    return text


# ── Individual Cleaning Steps ──────────────────────────────────

def _normalize_line_endings(text: str) -> str:
    """Standardize all line endings to Unix-style \n"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _remove_control_characters(text: str) -> str:
    """
    Remove ASCII control characters (0x00–0x1F) EXCEPT:
    - \n (newline) — we need this for line breaks
    - \t (tab)     — useful for preserving table-like structure
    """
    # \x00-\x08 and \x0b-\x0c and \x0e-\x1f are the ones we remove
    # \x09 = tab, \x0a = newline — we keep those
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def _collapse_spaces(text: str) -> str:
    """Replace multiple spaces/tabs on a single line with one space"""
    # Replace any run of spaces or tabs (but NOT newlines) with a single space
    return re.sub(r"[ \t]+", " ", text)


def _collapse_blank_lines(text: str) -> str:
    """
    Replace 3+ consecutive blank lines with just 2.
    We keep 2 to preserve paragraph structure (important for chunking later).
    """
    return re.sub(r"\n{3,}", "\n\n", text)


# ── Stats helper — useful for debugging ───────────────────────

def get_text_stats(text: str) -> dict:
    """
    Returns basic stats about the text.
    Useful to understand what you're working with before chunking.
    """
    lines = text.split("\n")
    words = text.split()

    return {
        "characters": len(text),
        "words": len(words),
        "lines": len(lines),
        "non_empty_lines": len([l for l in lines if l.strip()]),
        # Rough estimate: LLMs count tokens ~ chars/4
        "estimated_tokens": len(text) // 4,
    }