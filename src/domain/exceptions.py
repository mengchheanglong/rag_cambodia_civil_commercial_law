"""
Domain-specific exceptions.

These exceptions represent business rule violations and
error conditions in the legal RAG pipeline.
"""


class DomainError(Exception):
    """Base exception for all domain errors."""


class DocumentNotFoundError(DomainError):
    """Raised when a requested legal document is not found."""

    def __init__(self, document_name: str) -> None:
        self.document_name = document_name
        super().__init__(f"Legal document not found: {document_name}")


class ExtractionError(DomainError):
    """Raised when PDF text extraction fails."""

    def __init__(self, pdf_path: str, reason: str) -> None:
        self.pdf_path = pdf_path
        self.reason = reason
        super().__init__(f"Failed to extract text from '{pdf_path}': {reason}")


class ChunkingError(DomainError):
    """Raised when legal text chunking fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Chunking failed: {reason}")


class EmbeddingError(DomainError):
    """Raised when embedding generation fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Embedding generation failed: {reason}")


class RetrievalError(DomainError):
    """Raised when retrieval operations fail."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Retrieval failed: {reason}")


class GenerationError(DomainError):
    """Raised when LLM answer generation fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Answer generation failed: {reason}")


class CitationVerificationError(DomainError):
    """Raised when a citation cannot be verified against retrieved context."""

    def __init__(self, article_number: int, law_name: str) -> None:
        self.article_number = article_number
        self.law_name = law_name
        super().__init__(
            f"Citation for Article {article_number} of '{law_name}' "
            "could not be verified against retrieved context."
        )
