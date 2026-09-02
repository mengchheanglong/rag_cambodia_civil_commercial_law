"""Hybrid search and retrieval endpoint."""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.application.dtos import RetrievalRequest
from src.application.use_cases.hybrid_retrieve import HybridRetrieveUseCase
from src.interfaces.api.dependencies import get_hybrid_retriever

router = APIRouter(tags=["Retrieval"])


class SearchQuery(BaseModel):
    query: str = Field(
        ...,
        json_schema_extra={"example": "How is a contract formed under the Civil Code?"},
    )
    top_k: int = Field(default=5, ge=1, le=50)
    law_filter: Optional[str] = Field(
        default=None,
        json_schema_extra={"example": "Civil Code 2007"},
    )


class RetrievedChunkResponse(BaseModel):
    law_name: str
    article_number: int
    article_title: Optional[str] = None
    book: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    content: str
    relevance_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[RetrievedChunkResponse]


@router.post("/retrieve", response_model=SearchResponse)
async def retrieve_articles(
    payload: SearchQuery,
    retriever: HybridRetrieveUseCase = Depends(get_hybrid_retriever),
) -> SearchResponse:
    """
    Perform hybrid retrieval (dense semantic + BM25 sparse) + reranking.
    """
    req = RetrievalRequest(
        query=payload.query,
        top_k=payload.top_k,
        law_filter=payload.law_filter,
    )
    docs = retriever.execute(req)

    results = [
        RetrievedChunkResponse(
            law_name=doc.chunk.metadata.law_name,
            article_number=doc.chunk.metadata.article_number,
            article_title=doc.chunk.metadata.article_title,
            book=doc.chunk.metadata.book,
            chapter=doc.chunk.metadata.chapter,
            section=doc.chunk.metadata.section,
            content=doc.chunk.content,
            relevance_score=doc.rerank_score or doc.rrf_score or doc.sparse_score,
        )
        for doc in docs
    ]

    return SearchResponse(
        query=payload.query,
        total_results=len(results),
        results=results,
    )
