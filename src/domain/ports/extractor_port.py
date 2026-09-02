"""Abstract interface for PDF text extractors."""

from abc import ABC, abstractmethod
from pathlib import Path


class ExtractorPort(ABC):
    """
    Port for extracting raw text from legal PDF documents.

    Implementations handle PDF parsing, header/footer stripping,
    and OCR fallback for scanned documents.
    """

    @abstractmethod
    def extract(
        self,
        pdf_path: Path,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> str:
        """
        Extract clean text from a legal PDF.

        Args:
            pdf_path: Path to the PDF file.
            start_page: First page to extract (1-indexed).
            end_page: Last page to extract (inclusive). None = all pages.

        Returns:
            Extracted text with headers/footers removed.

        Raises:
            ExtractionError: If the PDF cannot be read or parsed.
        """
        ...

    @abstractmethod
    def is_scanned(self, pdf_path: Path) -> bool:
        """
        Detect whether a PDF is image-based (scanned) rather than text-selectable.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            True if the PDF requires OCR for text extraction.
        """
        ...
