"""
Domain layer — core business logic and port interfaces.

This layer has ZERO external dependencies beyond pydantic.
All infrastructure concerns are abstracted behind ports.
"""

from src.domain.entities import (
    Citation,
    Language,
    LegalChunk,
    LegalMetadata,
    QAResponse,
    RetrievedDocument,
)

__all__ = [
    "Citation",
    "Language",
    "LegalChunk",
    "LegalMetadata",
    "QAResponse",
    "RetrievedDocument",
]
