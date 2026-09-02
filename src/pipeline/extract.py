"""
Extraction pipeline script.

Extracts text from raw legal PDFs in data/01_raw/
and outputs cleaned, extracted text files in data/02_extracted/.
"""

import json
from pathlib import Path

from src.config.logging import get_logger, setup_logging
from src.infrastructure.extractors.pymupdf_extractor import PyMuPDFExtractor

setup_logging()
logger = get_logger(__name__)

# Document configuration for extraction
DOC_CONFIGS = [
    {
        "law_name": "Civil Code 2007",
        "law_key": "civil_code_2007_en",
        "pdf_path": "data/01_raw/en/civil_code_2007_en.pdf",
        "language": "en",
        # Book 1 begins on page 9; pages 1-8 are Table of Contents
        "start_page": 9,
        "end_page": None,
    },
    {
        "law_name": "Law on Commercial Arbitration 2006",
        "law_key": "commercial_arbitration_2006_en",
        "pdf_path": "data/01_raw/en/commercial_arbitration_2006_en.pdf",
        "language": "en",
        "start_page": 2,
        "end_page": None,
    },
    {
        "law_name": "Law on Commercial Enterprises 2005",
        "law_key": "commercial_enterprises_2005_en",
        "pdf_path": "data/01_raw/en/commercial_enterprises_2005_en.pdf",
        "language": "en",
        "start_page": 1,
        "end_page": None,
    },
]


def run_extraction() -> None:
    """Execute text extraction for all configured legal PDFs."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    extracted_dir = base_dir / "data" / "02_extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    extractor = PyMuPDFExtractor(header_margin_pt=45.0, footer_margin_pt=45.0)

    for config in DOC_CONFIGS:
        pdf_file = base_dir / config["pdf_path"]
        law_key = config["law_key"]
        law_name = config["law_name"]

        if not pdf_file.exists():
            logger.warning(f"PDF file not found: {pdf_file}. Skipping.")
            continue

        logger.info(f"Processing '{law_name}' ({pdf_file.name})...")

        # Check if scanned
        is_scanned = extractor.is_scanned(pdf_file)
        if is_scanned:
            logger.warning(f"'{pdf_file.name}' appears to be scanned image PDF. OCR may be needed.")

        # Extract text
        extracted_text = extractor.extract(
            pdf_path=pdf_file,
            start_page=config["start_page"],
            end_page=config["end_page"],
        )

        # Output text file
        output_txt = extracted_dir / f"{law_key}.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        # Output metadata
        meta = {
            "law_name": law_name,
            "law_key": law_key,
            "language": config["language"],
            "source_pdf": str(config["pdf_path"]),
            "start_page": config["start_page"],
            "end_page": config["end_page"],
            "character_count": len(extracted_text),
            "line_count": len(extracted_text.splitlines()),
        }
        output_meta = extracted_dir / f"{law_key}_meta.json"
        with open(output_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"Successfully extracted '{law_name}': {len(extracted_text):,} chars -> {output_txt.name}"
        )


if __name__ == "__main__":
    run_extraction()
