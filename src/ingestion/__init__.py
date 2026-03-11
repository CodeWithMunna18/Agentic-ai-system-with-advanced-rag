# Make load_document importable directly from src.ingestion
from .loader import load_document, load_documents_from_folder, Document
from .extractors import SUPPORTED_EXTENSIONS