"""Abstract interface for embedding models."""

from abc import ABC, abstractmethod


class EmbeddingPort(ABC):
    """
    Port for generating vector embeddings from text.

    Implementations wrap specific embedding APIs (OpenAI, HuggingFace, etc.).
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            EmbeddingError: If the embedding API call fails.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a single query string.

        Args:
            query: The search query to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...
