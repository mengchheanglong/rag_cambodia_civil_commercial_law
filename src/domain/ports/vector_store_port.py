"""Abstract interface for vector stores."""

from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities import LegalChunk, RetrievedDocument


class VectorStorePort(ABC):
    """
    Port for storing and searching vector embeddings.

    Implementations wrap specific vector databases (pgvector, FAISS, Chroma, etc.).
    """

    @abstractmethod
    def store(
        self,
        chunks: list[LegalChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store chunks and their embeddings in the vector database.

        Args:
            chunks: Legal text chunks with metadata.
            embeddings: Corresponding embedding vectors.

        Raises:
            StorageError: If the database operation fails.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 50,
        filters: Optional[dict] = None,
    ) -> list[RetrievedDocument]:
        """
        Search for similar chunks by vector similarity.

        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters (e.g., {"law_name": "Civil Code 2007"}).

        Returns:
            List of RetrievedDocument with dense_score populated.
        """
        ...

    @abstractmethod
    def delete_by_law(self, law_name: str) -> int:
        """
        Delete all chunks belonging to a specific law.

        Args:
            law_name: Name of the law to delete.

        Returns:
            Number of chunks deleted.
        """
        ...
