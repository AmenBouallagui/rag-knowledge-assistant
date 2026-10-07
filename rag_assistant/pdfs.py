"""Extract text and split within pages to preserve precise citations."""

from dataclasses import dataclass
from pathlib import Path
import warnings

from pypdf import PdfReader


@dataclass
class Chunk:
    source: str
    page: int
    text: str


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Use overlapping character windows; prefer a nearby whitespace boundary."""
    if not 0 <= overlap < chunk_size <= 4000:
        raise ValueError("Require 0 <= overlap < chunk_size <= 4000 characters.")
    text = " ".join(text.split())
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start + overlap:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(text):
            break
        start = end - overlap
    return pieces


def load_chunks(folder: Path, chunk_size: int = 1200, overlap: int = 200) -> list[Chunk]:
    # Validate even when there are no files or no text.
    split_text("", chunk_size, overlap)
    if not folder.is_dir():
        raise ValueError(f"PDF directory does not exist: {folder}")
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    if not files:
        raise ValueError(f"No PDFs found in {folder}.")
    chunks = []
    for path in files:
        source = path.relative_to(folder).as_posix()
        try:
            reader = PdfReader(path)
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("password-protected PDF")
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    warnings.warn(f"Skipping {source}, page {page_number}: no text (OCR may be needed).")
                chunks.extend(Chunk(source, page_number, text)
                              for text in split_text(text, chunk_size, overlap))
        except Exception as exc:
            raise ValueError(f"Could not read {source}: {exc}") from exc
    if not chunks:
        raise ValueError("No extractable text found. Scanned PDFs require OCR first.")
    return chunks
