"""
file_parser.py — PDF, DOCX, TXT parsing and text chunking utilities.
"""

from __future__ import annotations

import re
from typing import List


def parse_file(file_bytes: bytes, filename: str) -> str:
    """Parse uploaded file bytes into a plain-text string.

    Supported formats:
        • .pdf  — extracted via PyMuPDF (fitz)
        • .docx — extracted via python-docx
        • .txt  — decoded as UTF-8

    Raises:
        ValueError: for unsupported file types or parse failures.
    """
    lower = filename.lower()

    if lower.endswith(".pdf"):
        return _parse_pdf(file_bytes)
    elif lower.endswith(".docx"):
        return _parse_docx(file_bytes)
    elif lower.endswith(".txt"):
        return _parse_txt(file_bytes)
    else:
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "unknown"
        raise ValueError(
            f"Unsupported file type '.{ext}'. "
            "Only PDF, DOCX, and TXT files are accepted."
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_pdf(file_bytes: bytes) -> str:
    """Extract text from all pages of a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages: List[str] = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install pymupdf"
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF: {exc}") from exc


def _parse_docx(file_bytes: bytes) -> str:
    """Extract text from every paragraph in a DOCX file."""
    try:
        import io
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        raise RuntimeError(
            "python-docx is not installed. Run: pip install python-docx"
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse DOCX: {exc}") from exc


def _parse_txt(file_bytes: bytes) -> str:
    """Decode a plain-text file as UTF-8 (with fallback to latin-1)."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """Split *text* into overlapping word-level chunks.

    Strategy:
      1. Normalise whitespace and clean the text.
      2. Split into sentences using ". " as primary delimiter.
      3. Accumulate sentences into chunks of at most *chunk_size* words.
      4. Apply *overlap* words of context from the previous chunk.
      5. Drop chunks shorter than 50 characters.

    Args:
        text:       Raw document text.
        chunk_size: Maximum number of words per chunk (approx. token count).
        overlap:    Number of words carried over from the previous chunk.

    Returns:
        A list of text chunk strings.
    """
    # --- 1. Clean ---------------------------------------------------------
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # --- 2. Split into sentences ------------------------------------------
    # Use a regex that splits on ". ", "! ", "? " while keeping punctuation.
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    raw_sentences = sentence_pattern.split(text)

    # Tokenise each sentence into words
    sentences: List[List[str]] = [s.split() for s in raw_sentences if s.strip()]

    if not sentences:
        return []

    # --- 3. Build chunks --------------------------------------------------
    chunks: List[str] = []
    current_words: List[str] = []

    for sentence_words in sentences:
        # If adding this sentence would overflow, flush current chunk first
        if current_words and len(current_words) + len(sentence_words) > chunk_size:
            chunk_text_str = " ".join(current_words)
            if len(chunk_text_str) >= 50:
                chunks.append(chunk_text_str)
            # --- 4. Overlap: carry last *overlap* words into next chunk ----
            current_words = current_words[-overlap:] if overlap else []

        current_words.extend(sentence_words)

    # Flush the final accumulated words
    if current_words:
        chunk_text_str = " ".join(current_words)
        if len(chunk_text_str) >= 50:
            chunks.append(chunk_text_str)

    # --- 5. Filter short chunks ------------------------------------------
    chunks = [c.strip() for c in chunks if len(c.strip()) >= 50]

    return chunks
