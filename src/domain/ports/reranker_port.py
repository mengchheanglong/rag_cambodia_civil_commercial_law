"""Abstract interface for cross-encoder rerankers."""

from abc import ABC, abstractmethod

from src.domain.entities import RetrievedDocument


class RerankerPort(ABC):
    """
    Port for reranking retrieved documents using a cross-encoder.

    Implementations score (query, passage) pairs jointly for
    more accurate relevance ranking than bi-encoder retrieval.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """
        Rerank candidate documents using a cross-encoder model.

        Args:
            query: The original search query.
            documents: Candidate documents from hybrid retrieval.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of RetrievedDocument with rerank_score populated.
        """
        ...
