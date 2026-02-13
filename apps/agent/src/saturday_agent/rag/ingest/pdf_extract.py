from __future__ import annotations

import re
from pathlib import Path
from typing import List

try:
    from pypdf import PdfReader
except ModuleNotFoundError:
    PdfReader = None  # type: ignore[assignment]


class PdfExtractionError(RuntimeError):
    """Raised when a PDF cannot be parsed into text."""


def _clean_page_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(file_path: str) -> List[str]:
    if PdfReader is None:
        raise PdfExtractionError(
            "pypdf is required for rag.ingest_pdf. Install it with `pip install pypdf`."
        )

    source = Path(str(file_path or "")).expanduser()
    if not source.exists() or not source.is_file():
        raise PdfExtractionError(f"PDF file was not found: {source}")

    try:
        reader = PdfReader(str(source))
    except Exception as exc:
        raise PdfExtractionError(f"Failed to open PDF '{source}': {exc}") from exc

    pages: List[str] = []
    for page_index, page in enumerate(reader.pages):
        try:
            extracted = page.extract_text() or ""
        except Exception as exc:
            raise PdfExtractionError(
                f"Failed to extract text from page {page_index + 1} of '{source}': {exc}"
            ) from exc

        cleaned = _clean_page_text(extracted)
        if cleaned:
            pages.append(cleaned)

    if not pages:
        raise PdfExtractionError(
            f"No extractable text was found in '{source}'. "
            "The PDF may contain scanned images only."
        )
    return pages


def extract_pdf_text(file_path: str) -> str:
    return "\n\n".join(extract_pdf_pages(file_path))
