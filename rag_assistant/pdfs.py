"""Extract text and split within pages to preserve precise citations."""

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

from pypdf import PdfReader


@dataclass
class Chunk:
    source: str
    page: int
    text: str


def _logical_units(text: str, chunk_size: int):
    for line in text.splitlines():
        # The bullet glyph is unambiguous, even without spaces. For ASCII
        # markers, require whitespace so hyphenated words stay together.
        for part in re.split(r"(?=•)|(?<=\s)(?=[*-]\s)", line):
            unit = " ".join(part.split())
            while len(unit) > chunk_size:
                window = unit[:chunk_size + 1]
                sentence_ends = [match.end() for match in re.finditer(r'[.!?]["”’)]*(?= )', window)]
                end = sentence_ends[-1] if sentence_ends else window.rfind(" ")
                if end <= 0:
                    end = chunk_size  # A single unbroken token exceeds the limit.
                yield unit[:end].rstrip()
                unit = unit[end:].lstrip()
            if unit:
                yield unit


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Pack page text into chunks, preferring complete lines and bullets."""
    if not 0 <= overlap < chunk_size <= 4000:
        raise ValueError("Require 0 <= overlap < chunk_size <= 4000 characters.")

    chunks = []
    current = []
    length = 0
    for unit in _logical_units(text, chunk_size):
        if current and length + 1 + len(unit) > chunk_size:
            chunks.append("\n".join(current))
            trailing = []
            length = 0
            for previous in reversed(current):
                candidate = length + len(previous) + bool(trailing)
                if candidate + 1 + len(unit) > chunk_size:
                    break
                if abs(overlap - candidate) >= abs(overlap - length):
                    break  # The larger suffix would be farther from the target.
                trailing.append(previous)
                length = candidate
            current = trailing[::-1]
        length += len(unit) + bool(current)
        current.append(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


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
