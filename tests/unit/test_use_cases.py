"""Unit tests for AnswerLegalQAUseCase and citation verification."""

from unittest.mock import MagicMock

from src.application.dtos import LegalQARequest
from src.application.use_cases.answer_legal_qa import AnswerLegalQAUseCase
from src.domain.entities import Language, LegalChunk, LegalMetadata, RetrievedDocument


def test_answer_legal_qa_with_verified_citation():
    """QA use case should verify citations against retrieved articles."""
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    chunk315 = LegalChunk(
        chunk_id="chk_315",
        content="Article 315. A contract is formed when an offer and acceptance agree.",
        content_with_context="[Civil Code 2007] Article 315",
        metadata=LegalMetadata(
            law_name="Civil Code 2007",
            article_number=315,
            language=Language.ENGLISH,
        ),
    )
    mock_retriever.execute.return_value = [RetrievedDocument(chunk=chunk315, rrf_score=0.95)]

    mock_llm.generate.return_value = (
        "Under the Cambodian Civil Code, a contract is formed upon mutual agreement (Civil Code 2007, Article 315)."
    )

    qa_use_case = AnswerLegalQAUseCase(retriever=mock_retriever, llm=mock_llm)
    req = LegalQARequest(question="How is a contract formed in Cambodia?")
    response = qa_use_case.execute(req)

    assert "Article 315" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0]["article_number"] == 315
    assert response.citations[0]["is_verified"] is True
    assert response.citations[0]["law_name"] == "Civil Code 2007"


def test_answer_legal_qa_unverified_citation_flagged():
    """If LLM cites an article not in context, is_verified should be False."""
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    # Context only contains Article 10
    chunk10 = LegalChunk(
        chunk_id="chk_10",
        content="Article 10 content...",
        content_with_context="[Civil Code 2007] Article 10",
        metadata=LegalMetadata(
            law_name="Civil Code 2007",
            article_number=10,
            language=Language.ENGLISH,
        ),
    )
    mock_retriever.execute.return_value = [RetrievedDocument(chunk=chunk10)]

    # Model hallucinates Article 999
    mock_llm.generate.return_value = "According to Article 999, parties must act fairly."

    qa_use_case = AnswerLegalQAUseCase(retriever=mock_retriever, llm=mock_llm)
    req = LegalQARequest(question="What does Article 999 say?")
    response = qa_use_case.execute(req)

    assert len(response.citations) == 1
    assert response.citations[0]["article_number"] == 999
    assert response.citations[0]["is_verified"] is False
