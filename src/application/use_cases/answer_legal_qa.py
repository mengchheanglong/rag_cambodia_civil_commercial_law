"""
Use Case: Answer a legal question with cited articles.

Pipeline: Question → Hybrid Retrieve → Rerank → LLM Generate (DeepSeek / OpenAI) → Verify Citations
"""

from src.config.logging import get_logger
from src.application.dtos import LegalQARequest, LegalQAResponse, RetrievalRequest
from src.application.use_cases.hybrid_retrieve import HybridRetrieveUseCase
from src.domain.entities import Citation, RetrievedDocument
from src.domain.ports.llm_port import LLMPort

logger = get_logger(__name__)

# The system prompt enforcing strict citation behavior
LEGAL_SYSTEM_PROMPT = """You are a legal assistant specializing in Cambodian Civil and Commercial Law.

RULES:
1. Answer strictly based on the provided legal articles. Do NOT use outside knowledge.
2. For every legal rule or statement you mention, cite the exact Law name, Chapter, and Article number.
3. Use this citation format: (Law Name, Article XX)
4. If the provided context does not contain the answer, state:
   "The provided legal texts do not explicitly address this question."
5. Be precise and concise. Use legal terminology accurately.
6. When multiple articles are relevant, cite all of them.
7. Never fabricate or assume article numbers that are not in the context.
"""


class AnswerLegalQAUseCase:
    """
    Orchestrates the full legal Q&A pipeline.

    Retrieves relevant legal articles via hybrid search,
    generates a cited answer using DeepSeek / OpenAI, and verifies
    that all citations reference articles in the retrieved context.
    """

    def __init__(
        self,
        retriever: HybridRetrieveUseCase,
        llm: LLMPort,
    ) -> None:
        self._retriever = retriever
        self._llm = llm

    def execute(self, request: LegalQARequest) -> LegalQAResponse:
        """
        Answer a legal question with article-level citations.

        Args:
            request: The legal question and retrieval parameters.

        Returns:
            LegalQAResponse with answer, reasoning, citations, and source articles.
        """
        logger.info("Processing legal question", question=request.question, model=request.model)

        # Step 1: Retrieve relevant articles
        retrieval_request = RetrievalRequest(
            query=request.question,
            top_k=request.top_k,
            law_filter=request.law_filter,
        )
        retrieved_docs = self._retriever.execute(retrieval_request)
        logger.info("Retrieved source articles", count=len(retrieved_docs))

        # Step 2: Generate answer with citations
        generate_kwargs = {
            "query": request.question,
            "context_documents": retrieved_docs,
            "system_prompt": LEGAL_SYSTEM_PROMPT,
        }
        if request.model and hasattr(self._llm, "generate"):
            try:
                answer = self._llm.generate(
                    query=request.question,
                    context_documents=retrieved_docs,
                    system_prompt=LEGAL_SYSTEM_PROMPT,
                    model_override=request.model,
                )
            except TypeError:
                answer = self._llm.generate(**generate_kwargs)
        else:
            answer = self._llm.generate(**generate_kwargs)

        reasoning_content = None
        raw_reasoning = getattr(self._llm, "last_reasoning_content", None)
        if isinstance(raw_reasoning, str):
            reasoning_content = raw_reasoning

        model_used = None
        if request.model:
            model_used = request.model
        else:
            raw_model = getattr(self._llm, "_model", None)
            if isinstance(raw_model, str):
                model_used = raw_model

        logger.info("Answer generated")

        # Step 3: Verify citations against retrieved context
        citations = self._verify_citations(answer, retrieved_docs)

        # Step 4: Build response
        source_articles = [
            {
                "law_name": doc.chunk.metadata.law_name,
                "article_number": doc.chunk.metadata.article_number,
                "chapter": doc.chunk.metadata.chapter,
                "content_preview": doc.chunk.content[:200],
                "relevance_score": doc.rerank_score or doc.rrf_score,
            }
            for doc in retrieved_docs
        ]

        return LegalQAResponse(
            question=request.question,
            answer=answer,
            reasoning_content=reasoning_content,
            model_used=model_used,
            citations=[c.model_dump() for c in citations],
            source_articles=source_articles,
        )

    def _verify_citations(
        self,
        answer: str,
        retrieved_docs: list[RetrievedDocument],
    ) -> list[Citation]:
        """
        Extract and verify article citations from the generated answer.

        Checks that every cited article number exists in the retrieved context.

        Args:
            answer: The LLM-generated answer text.
            retrieved_docs: The articles used as context.

        Returns:
            List of Citation objects with verification status.
        """
        import re

        # Extract cited article numbers from the answer
        citation_pattern = r"Article\s+(\d+)"
        cited_numbers = set(re.findall(citation_pattern, answer))

        # Build a lookup of retrieved article numbers
        retrieved_articles = {
            doc.chunk.metadata.article_number: doc for doc in retrieved_docs
        }

        citations: list[Citation] = []
        for article_num_str in cited_numbers:
            article_num = int(article_num_str)
            is_verified = article_num in retrieved_articles

            excerpt = ""
            law_name = "Unknown"
            if is_verified:
                doc = retrieved_articles[article_num]
                excerpt = doc.chunk.content[:300]
                law_name = doc.chunk.metadata.law_name

            citations.append(
                Citation(
                    law_name=law_name,
                    article_number=article_num,
                    excerpt=excerpt,
                    is_verified=is_verified,
                )
            )

            if not is_verified:
                logger.warning(
                    "Unverified citation detected",
                    article_number=article_num,
                )

        return citations
