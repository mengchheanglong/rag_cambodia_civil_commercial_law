"""
Cross-Encoder Reranker Adapter.

Implements RerankerPort using sentence-transformers CrossEncoder.
Provides accurate (query, passage) joint scoring with fallback.
"""

from typing import Optional

from src.config.logging import get_logger
from src.config.settings import Settings, get_settings
from src.domain.entities import RetrievedDocument
from src.domain.ports.reranker_port import RerankerPort

logger = get_logger(__name__)


class CrossEncoderReranker(RerankerPort):
    """
    Reranker using a Cross-Encoder model.

    Scores (query, passage) pairs together to refine top-K candidates
    from hybrid retrieval down to the final top-N context articles.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_name = model_name or self._settings.reranker_model
        self._model = None
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazy load the CrossEncoder model."""
        if self._initialized:
            return
        self._initialized = True
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder reranker model", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("CrossEncoder reranker loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load CrossEncoder model '{self._model_name}': {e}. Using score-based fallback."
            )
            self._model = None

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """
        Rerank a list of retrieved documents for a query.

        Args:
            query: The user search query.
            documents: Candidate documents from hybrid retrieval.
            top_k: Number of top results to return.

        Returns:
            Reranked list of RetrievedDocument with rerank_score populated.
        """
        if not documents:
            return []

        self._lazy_init()

        if self._model:
            try:
                # Prepare (query, passage) pairs
                pairs = [(query, doc.chunk.content_with_context) for doc in documents]
                scores = self._model.predict(pairs)

                # Attach rerank scores
                scored_docs: list[RetrievedDocument] = []
                for doc, score in zip(documents, scores):
                    scored_docs.append(
                        RetrievedDocument(
                            chunk=doc.chunk,
                            dense_score=doc.dense_score,
                            sparse_score=doc.sparse_score,
                            rrf_score=doc.rrf_score,
                            rerank_score=float(score),
                        )
                    )

                # Sort descending by rerank score
                scored_docs.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
                return scored_docs[:top_k]

            except Exception as e:
                logger.warning(f"Error during CrossEncoder prediction: {e}. Falling back to RRF/dense scores.")

        # Fallback ranking: sort by RRF score or dense score
        fallback_docs = list(documents)
        fallback_docs.sort(
            key=lambda d: d.rrf_score if d.rrf_score is not None else (d.dense_score or 0.0),
            reverse=True,
        )
        for doc in fallback_docs:
            if doc.rerank_score is None:
                doc.rerank_score = doc.rrf_score or doc.dense_score

        return fallback_docs[:top_k]
