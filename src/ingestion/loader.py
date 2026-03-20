# loader.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: The Loader is the "orchestrator" for Phase 1.
# It doesn't know HOW to extract text — it delegates that to
# extractors.py. It doesn't know HOW to clean — delegates to
# cleaner.py. Its only job: accept a file or folder, return
# a list of structured document objects.
#
# This is the "Orchestrator Pattern" — a key design in RAG systems
# where each component has one job and talks to others.
# ─────────────────────────────────────────────────────────────

import pathlib
from dataclasses import dataclass, field
from typing import Optional

from .extractors import extract_text, SUPPORTED_EXTENSIONS
from .cleaner import clean_text, get_text_stats


# ── Document Dataclass ─────────────────────────────────────────
# CONCEPT: We wrap raw text in a structured object so we can
# carry metadata (filename, path, stats) alongside the text.
# In Phase 3, we'll add chunk_id, embedding, etc. to this structure.

@dataclass
class Document:
    """
    Represents one loaded & cleaned document.
    This is the core data structure that flows through the entire pipeline.
    """
    file_name: str                    # e.g. "research_paper.pdf"
    file_path: str                    # full path on disk
    file_type: str                    # e.g. ".pdf"
    raw_text: str                     # text straight from extractor (before cleaning)
    clean_text: str                   # text after cleaning pipeline
    stats: dict = field(default_factory=dict)  # word count, tokens, etc.
    error: Optional[str] = None       # if loading failed, reason is stored here

    @property
    def is_valid(self) -> bool:
        """A document is valid if it has text and no error."""
        return self.error is None and bool(self.clean_text.strip())

    def __repr__(self):
        status = "✅" if self.is_valid else "❌"
        words = self.stats.get("words", 0)
        tokens = self.stats.get("estimated_tokens", 0)
        return f"{status} Document('{self.file_name}', {words} words, ~{tokens} tokens)"


# ── Core Loading Functions ─────────────────────────────────────

def load_document(file_path: str | pathlib.Path) -> Document:
    """
    Load a single document from a file path.
    Returns a Document object — even if loading fails (error is stored inside).
    We never raise exceptions here so the pipeline can continue with other files.
    """
    path = pathlib.Path(file_path)

    # Base document shell — only metadata here, NO raw_text/clean_text
    # (those are passed separately below to avoid "multiple values" error)
    doc_base = {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "file_type": path.suffix.lower(),
    }

    # Guard: does the file exist?
    if not path.exists():
        return Document(**doc_base, raw_text="", clean_text="", error=f"File not found: {path}")

    # Guard: is the file type supported?
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return Document(**doc_base, raw_text="", clean_text="", error=f"Unsupported type: {path.suffix}")

    # Guard: is the file empty?
    if path.stat().st_size == 0:
        return Document(**doc_base, raw_text="", clean_text="", error="File is empty (0 bytes)")

    try:
        # Step 1: Extract raw text using the right extractor
        raw = extract_text(path)

        # Step 2: Clean the raw text
        cleaned = clean_text(raw)

        # Step 3: Compute stats on the CLEAN text
        stats = get_text_stats(cleaned)

        return Document(
            **doc_base,
            raw_text=raw,
            clean_text=cleaned,
            stats=stats,
        )

    except Exception as e:
        # Catch anything unexpected — store the error, don't crash
        return Document(**doc_base, raw_text="", clean_text="", error=str(e))


def load_documents_from_folder(
    folder_path: str | pathlib.Path,
    recursive: bool = False
) -> list[Document]:
    """
    Load all supported documents from a folder.

    Args:
        folder_path: path to the folder
        recursive:   if True, also searches subfolders (Phase 5 will use this)

    Returns:
        List of Document objects (both valid and failed ones, so you can inspect errors)
    """
    folder = pathlib.Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    # Find all files — recursive or top-level only
    pattern = "**/*" if recursive else "*"
    all_files = [f for f in folder.glob(pattern) if f.is_file()]

    # Filter to only supported file types
    supported_files = [
        f for f in all_files
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not supported_files:
        print(f"⚠️  No supported files found in '{folder}'")
        print(f"   Supported types: {', '.join(SUPPORTED_EXTENSIONS)}")
        return []

    print(f"📂 Found {len(supported_files)} supported file(s) in '{folder.name}/'")

    # Load each file — collect all results
    documents = []
    for file_path in supported_files:
        print(f"   Loading: {file_path.name}...", end=" ")
        doc = load_document(file_path)
        print(repr(doc))
        documents.append(doc)

    # Summary
    valid = [d for d in documents if d.is_valid]
    failed = [d for d in documents if not d.is_valid]
    print(f"\n📊 Loaded {len(valid)} valid | {len(failed)} failed")

    return documents