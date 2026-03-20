# extractors.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: An "extractor" knows how to pull raw text out of ONE
# specific file type. We keep each extractor separate so adding
# a new file type later is just adding a new function — not
# rewriting everything. This is the "Single Responsibility" idea.
# ─────────────────────────────────────────────────────────────

import pathlib
from pypdf import PdfReader
from docx import Document


# ── PDF Extractor ──────────────────────────────────────────────
def extract_from_pdf(file_path: pathlib.Path) -> str:
    """
    WHY pypdf?
    PDFs store text in a weird internal format. pypdf handles the
    decoding so we get plain strings. Each PDF has 'pages' and each
    page has its own text block — we join them all together.
    """
    reader = PdfReader(file_path)

    pages_text = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()

        # Some PDF pages are images (scanned docs) — extract_text()
        # returns None or empty string for those. We skip them for now.
        # (Phase 6 will add OCR support for scanned pages.)
        if text and text.strip():
            pages_text.append(text)
        else:
            print(f"  ⚠️  Page {page_number + 1} appears to be an image/scanned — skipping")

    # Join all pages with a newline separator so page breaks are preserved
    return "\n".join(pages_text)
# print(extract_from_pdf("c:/Users/munna.b.kumar/Learning/project/Agent AI/doc_inteligence/documents/sample.pdf"))

# ── TXT Extractor ──────────────────────────────────────────────
def extract_from_txt(file_path: pathlib.Path) -> str:
    """
    Plain text files are simple — just read them.
    We try UTF-8 first (standard), fall back to latin-1 if the file
    has special characters from older systems.
    """
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"  ⚠️  UTF-8 failed for {file_path.name}, trying latin-1...")
        return file_path.read_text(encoding="latin-1")


# ── DOCX Extractor ─────────────────────────────────────────────
def extract_from_docx(file_path: pathlib.Path) -> str:
    """
    DOCX files are actually ZIP archives containing XML.
    python-docx parses that XML and gives us 'paragraphs'.
    Each paragraph is a logical text block (heading, sentence, list item, etc.)
    """
    doc = Document(file_path)

    # A paragraph with no text is just an empty line — we skip those
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

    return "\n".join(paragraphs)


# ── Router — picks the right extractor automatically ───────────
# CONCEPT: A dispatch table (dict mapping → function) is cleaner
# than a long if/elif chain. Adding a new format = one new line.
EXTRACTORS = {
    ".pdf":  extract_from_pdf,
    ".txt":  extract_from_txt,
    ".text": extract_from_txt,   # same extractor, different extension
    ".docx": extract_from_docx,
}

SUPPORTED_EXTENSIONS = set(EXTRACTORS.keys())


def extract_text(file_path: pathlib.Path) -> str:
    """
    Main entry point. Given any supported file, returns raw text.
    Raises a clear error if the file type isn't supported yet.
    """
    suffix = file_path.suffix.lower()

    if suffix not in EXTRACTORS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"Unsupported file type: '{suffix}'\n"
            f"Currently supported: {supported}"
        )

    extractor_fn = EXTRACTORS[suffix]
    return extractor_fn(file_path)