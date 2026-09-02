"""
OCR Pipeline for Scanned Legal PDFs.

Converts scanned image PDFs (like commercial_enterprises_2005_en.pdf)
to searchable text using pdf2image + pytesseract.

Usage:
    python -m src.pipeline.ocr
    python -m src.pipeline.ocr --input data/01_raw/en/commercial_enterprises_2005_en.pdf
    python -m src.pipeline.ocr --lang eng --dpi 300

Prerequisites:
    pip install pdf2image pytesseract
    # Windows: Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
    # Add Tesseract to PATH or set TESSERACT_CMD in .env
"""

import argparse
import os
import re
from pathlib import Path

from src.config.logging import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "01_raw" / "en"
EXTRACTED_DIR = BASE_DIR / "data" / "02_extracted"


def _check_dependencies() -> bool:
    """Check that pdf2image and pytesseract are installed."""
    try:
        import pdf2image  # noqa: F401
        import pytesseract  # noqa: F401
        return True
    except ImportError as e:
        logger.error(
            "Missing OCR dependencies. Install with: pip install pdf2image pytesseract",
            missing=str(e),
        )
        return False


def _set_tesseract_cmd() -> None:
    """Set the Tesseract executable path from environment or common defaults."""
    import pytesseract

    tesseract_cmd = os.environ.get("TESSERACT_CMD", "")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return

    # Common Windows installation paths
    windows_defaults = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in windows_defaults:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info("Found Tesseract", path=path)
            return


def ocr_pdf(
    pdf_path: Path,
    lang: str = "eng",
    dpi: int = 300,
    first_page: int = 1,
    last_page: int | None = None,
) -> str:
    """
    Extract text from a scanned PDF using OCR.

    Args:
        pdf_path: Path to the scanned PDF.
        lang: Tesseract language code ('eng', 'khm', 'eng+khm').
        dpi: Image resolution. 300 is recommended for most legal documents.
        first_page: First page to process (1-indexed).
        last_page: Last page to process. None = all pages.

    Returns:
        Concatenated extracted text for all pages.

    Raises:
        ImportError: If pdf2image or pytesseract is not installed.
        RuntimeError: If Tesseract binary is not found.
    """
    if not _check_dependencies():
        raise ImportError("pdf2image and pytesseract are required for OCR.")

    from pdf2image import convert_from_path
    import pytesseract

    _set_tesseract_cmd()

    logger.info(
        "Starting OCR",
        pdf=pdf_path.name,
        lang=lang,
        dpi=dpi,
        first_page=first_page,
        last_page=last_page or "all",
    )

    # Convert PDF pages to images
    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
    )
    logger.info("Converted PDF to images", total_pages=len(images))

    extracted_pages: list[str] = []
    for page_num, image in enumerate(images, start=first_page):
        if page_num % 10 == 0 or page_num == first_page:
            logger.info("OCR processing page", page=page_num, total=len(images) + first_page - 1)

        page_text = pytesseract.image_to_string(image, lang=lang)
        page_text = _clean_ocr_page(page_text)
        if page_text.strip():
            extracted_pages.append(page_text)

    full_text = "\n\n".join(extracted_pages)
    logger.info("OCR complete", total_chars=len(full_text), total_pages=len(images))
    return full_text


def _clean_ocr_page(text: str) -> str:
    """
    Basic OCR post-processing cleanup.

    - Remove form feed characters
    - Collapse excessive blank lines
    - Fix common OCR character substitutions
    """
    text = text.replace("\x0c", "\n")  # form feed
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines)


def run_ocr_pipeline(
    pdf_path: Path | None = None,
    lang: str = "eng",
    dpi: int = 300,
) -> None:
    """
    Run the full OCR pipeline: scan PDF → extract text → save to data/02_extracted/.

    Args:
        pdf_path: Specific PDF to OCR. If None, scans RAW_DIR for scanned PDFs.
        lang: Tesseract language ('eng', 'khm', 'eng+khm').
        dpi: Image resolution for conversion.
    """
    if pdf_path:
        pdf_files = [pdf_path]
    else:
        # Process all PDFs in raw dir that are likely scanned (large files)
        all_pdfs = list(RAW_DIR.glob("*.pdf"))
        pdf_files = []
        for pdf in all_pdfs:
            # Skip already-extracted files
            stem = pdf.stem
            out_path = EXTRACTED_DIR / f"{stem}_ocr.txt"
            if out_path.exists():
                logger.info("Skipping already-OCR'd file", pdf=pdf.name)
                continue
            # Check if this PDF was flagged as scanned
            meta_path = EXTRACTED_DIR / f"{stem}_meta.json"
            if meta_path.exists():
                import json
                meta = json.loads(meta_path.read_text())
                if meta.get("is_scanned"):
                    pdf_files.append(pdf)
            else:
                # If no meta, try to detect scanned
                pdf_files.append(pdf)

    if not pdf_files:
        logger.info("No scanned PDFs to process.")
        return

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    for pdf in pdf_files:
        logger.info("Processing scanned PDF", pdf=pdf.name)
        try:
            text = ocr_pdf(pdf_path=pdf, lang=lang, dpi=dpi)

            out_path = EXTRACTED_DIR / f"{pdf.stem}_ocr.txt"
            out_path.write_text(text, encoding="utf-8")
            logger.info("Saved OCR output", path=str(out_path), chars=len(text))

        except Exception as exc:
            logger.error("OCR failed for PDF", pdf=pdf.name, error=str(exc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR pipeline for scanned Cambodian legal PDFs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a specific PDF to OCR. If omitted, processes all flagged scanned PDFs.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help="Tesseract language code: 'eng', 'khm', 'eng+khm'. Default: eng.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution in DPI. Higher = better accuracy but slower. Default: 300.",
    )
    args = parser.parse_args()
    run_ocr_pipeline(pdf_path=args.input, lang=args.lang, dpi=args.dpi)
