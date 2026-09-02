"""Unit tests for Hybrid Retrieval and RRF Fusion."""

from unittest.mock import MagicMock

import pytest

from src.application.dtos import RetrievalRequest
from src.application.use_cases.hybrid_retrieve import HybridRetrieveUseCase
from src.domain.entities import Language, LegalChunk, LegalMetadata, RetrievedDocument


@pytest.fixture
def sample_docs() -> list[RetrievedDocument]:
    """Create sample retrieved documents with dummy chunks."""
    docs = []
    for i in range(1, 5):
        chunk = LegalChunk(
            chunk_id=f"doc_{i}",
            content=f"Article {i} content",
            content_with_context=f"[Law] Article {i}",
            metadata=LegalMetadata(
                law_name="Civil Code 2007",
                article_number=i,
                language=Language.ENGLISH,
            ),
        )
        docs.append(RetrievedDocument(chunk=chunk))
    return docs


def test_reciprocal_rank_fusion_scoring(sample_docs):
    """RRF should combine dense and sparse ranks correctly."""
    # Mock ports
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 10
    mock_vector_store = MagicMock()
    mock_sparse_retriever = MagicMock()
    mock_reranker = MagicMock()

    # Dense returns [doc_1, doc_2, doc_3]
    mock_vector_store.search.return_value = [sample_docs[0], sample_docs[1], sample_docs[2]]
    # Sparse returns [doc_2, doc_1, doc_4]
    mock_sparse_retriever.search.return_value = [sample_docs[1], sample_docs[0], sample_docs[3]]

    # Pass through reranker
    mock_reranker.rerank.side_effect = lambda query, documents, top_k: documents[:top_k]

    use_case = HybridRetrieveUseCase(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        sparse_retriever=mock_sparse_retriever,
        reranker=mock_reranker,
        rrf_k=60,
    )

    request = RetrievalRequest(query="test query", top_k=3)
    results = use_case.execute(request)

    assert len(results) == 3
    # doc_1 (dense rank 0, sparse rank 1) -> RRF = 1/61 + 1/62
    # doc_2 (dense rank 1, sparse rank 0) -> RRF = 1/62 + 1/61
    # Both doc_1 and doc_2 should be at the top
    top_chunk_ids = [r.chunk.chunk_id for r in results[:2]]
    assert "doc_1" in top_chunk_ids
    assert "doc_2" in top_chunk_ids
    assert results[0].rrf_score > 0
