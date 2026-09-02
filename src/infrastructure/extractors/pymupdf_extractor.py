"""
PyMuPDF-based legal PDF extractor.

Implements ExtractorPort using fitz (PyMuPDF) for high-performance
text extraction with header/footer clipping.
"""

from pathlib import Path

import fitz  # PyMuPDF
import regex as re

from src.config.logging import get_logger
from src.domain.exceptions import ExtractionError
from src.domain.ports.extractor_port import ExtractorPort

logger = get_logger(__name__)

# Minimum text-to-page ratio to consider a page as text-selectable
_SCANNED_THRESHOLD = 10  # characters per page


class PyMuPDFExtractor(ExtractorPort):
    """
    Extract text from legal PDFs using PyMuPDF.

    Clips header/footer margins to remove page numbers and
    running titles. Detects scanned (image-only) PDFs.
    """

    def __init__(
        self,
        header_margin_pt: float = 50.0,
        footer_margin_pt: float = 50.0,
    ) -> None:
        """
        Args:
            header_margin_pt: Points to clip from top of each page (72pt = 1 inch).
            footer_margin_pt: Points to clip from bottom of each page.
        """
        self._header_margin = header_margin_pt
        self._footer_margin = footer_margin_pt

    def extract(
        self,
        pdf_path: Path,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> str:
        """Extract clean text from a legal PDF with header/footer clipping."""
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise ExtractionError(str(pdf_path), str(e))

        total_pages = len(doc)
        last_page = end_page if end_page else total_pages
        full_text: list[str] = []

        logger.info(
            "Extracting PDF",
            path=str(pdf_path),
            pages=f"{start_page}-{last_page}",
            total_pages=total_pages,
        )

        for page_num in range(start_page - 1, last_page):
            page = doc[page_num]
            rect = page.rect

            # Clip header and footer regions
            body_rect = fitz.Rect(
                rect.x0,
                rect.y0 + self._header_margin,
                rect.x1,
                rect.y1 - self._footer_margin,
            )

            page_text = page.get_text("text", clip=body_rect)

            # Strip repetitive translation headers
            page_text = re.sub(
                r"Tentative English Translation[^\n]*\n"
                r"(?:The original in Khmer[^\n]*\n)?"
                r"(?:Discussion w/ WB & ADB[^\n]*\n)?",
                "",
                page_text,
                flags=re.IGNORECASE,
            )

            # Remove index-style dot leaders (e.g., "Article 120 ............ 45")
            page_text = re.sub(r"\.{4,}\s*\d+", "", page_text)

            # Normalize excessive line breaks
            page_text = re.sub(r"\n{3,}", "\n\n", page_text)

            full_text.append(page_text)

        doc.close()

        result = "\n\n".join(full_text)
        logger.info("Extraction complete", characters=len(result))
        return result

    def is_scanned(self, pdf_path: Path) -> bool:
        """Detect whether a PDF is image-based (scanned)."""
        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            return False

        total_chars = 0
        sample_pages = min(len(doc), 5)  # Check first 5 pages

        for page_num in range(sample_pages):
            page = doc[page_num]
            total_chars += len(page.get_text("text").strip())

        doc.close()

        avg_chars_per_page = total_chars / max(sample_pages, 1)
        is_scanned = avg_chars_per_page < _SCANNED_THRESHOLD

        logger.info(
            "Scanned detection",
            path=str(pdf_path),
            avg_chars_per_page=avg_chars_per_page,
            is_scanned=is_scanned,
        )
        return is_scanned
