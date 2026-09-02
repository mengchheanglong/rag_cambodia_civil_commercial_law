"""Unit tests for BM25 Sparse Retriever."""

import pytest

from src.domain.entities import Language, LegalChunk, LegalMetadata
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever, tokenize_legal_text


@pytest.fixture
def sample_legal_chunks() -> list[LegalChunk]:
    """Sample chunks for testing BM25 indexing and querying."""
    meta1 = LegalMetadata(
        law_name="Civil Code 2007",
        chapter="Chapter 1 General",
        article_number=5,
        article_title="The principle of good faith",
        language=Language.ENGLISH,
    )
    c1 = LegalChunk(
        chunk_id="chk_5",
        content="5. (The principle of good faith) Rights shall be exercised and duties performed in good faith.",
        content_with_context="[Civil Code 2007] → [Chapter 1 General]\n5. (The principle of good faith) Rights shall be exercised and duties performed in good faith.",
        metadata=meta1,
    )

    meta2 = LegalMetadata(
        law_name="Civil Code 2007",
        chapter="Chapter 2 Contracts",
        article_number=315,
        article_title="Formation of contract",
        language=Language.ENGLISH,
    )
    c2 = LegalChunk(
        chunk_id="chk_315",
        content="315. A contract is formed when an offer and an acceptance are in agreement between the parties.",
        content_with_context="[Civil Code 2007] → [Chapter 2 Contracts]\n315. A contract is formed when an offer and an acceptance are in agreement between the parties.",
        metadata=meta2,
    )

    meta3 = LegalMetadata(
        law_name="Law on Commercial Arbitration 2006",
        chapter="CHAPTER II",
        article_number=10,
        article_title="Definition and form of arbitration agreement",
        language=Language.ENGLISH,
    )
    c3 = LegalChunk(
        chunk_id="chk_arb_10",
        content="Article 10: An arbitration agreement is an agreement by the parties to submit to arbitration disputes.",
        content_with_context="[Law on Commercial Arbitration 2006] → [CHAPTER II]\nArticle 10: An arbitration agreement is an agreement by the parties to submit to arbitration disputes.",
        metadata=meta3,
    )

    return [c1, c2, c3]


def test_tokenize_legal_text():
    """Tokenize should extract alphanumeric tokens and lowercase."""
    tokens = tokenize_legal_text("Article 315: Good Faith (Civil Code)")
    assert "article" in tokens
    assert "315" in tokens
    assert "good" in tokens
    assert "faith" in tokens


def test_bm25_search(tmp_path, sample_legal_chunks):
    """BM25 search should rank exact match article highest."""
    retriever = BM25Retriever(index_path=tmp_path / "test_bm25.pkl")
    retriever.index(sample_legal_chunks)

    # Search for contract formation
    results = retriever.search("contract offer acceptance", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.metadata.article_number == 315

    # Search for arbitration
    arb_results = retriever.search("arbitration dispute agreement", top_k=1)
    assert len(arb_results) == 1
    assert arb_results[0].chunk.metadata.article_number == 10
    assert arb_results[0].sparse_score > 0
