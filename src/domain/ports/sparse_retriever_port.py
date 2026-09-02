"""Abstract interface for sparse (keyword) retrievers."""

from abc import ABC, abstractmethod

from src.domain.entities import LegalChunk, RetrievedDocument


class SparseRetrieverPort(ABC):
    """
    Port for BM25 / keyword-based sparse retrieval.

    Implementations handle tokenization and BM25 scoring for
    exact-match and keyword-based legal queries.
    """

    @abstractmethod
    def index(self, chunks: list[LegalChunk]) -> None:
        """
        Build or update the sparse index from legal chunks.

        Args:
            chunks: Legal text chunks to index.
        """
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 50) -> list[RetrievedDocument]:
        """
        Search for relevant chunks using BM25 keyword matching.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievedDocument with sparse_score populated.
        """
        ...
