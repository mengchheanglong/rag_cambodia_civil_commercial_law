"""Legal Question & Answering endpoint with citation verification."""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.application.dtos import LegalQARequest, LegalQAResponse
from src.application.use_cases.answer_legal_qa import AnswerLegalQAUseCase
from src.interfaces.api.dependencies import get_qa_use_case

router = APIRouter(tags=["Legal Q&A"])


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        json_schema_extra={"example": "What are the essential elements required to form a valid contract in Cambodia?"},
    )
    top_k: int = Field(default=5, ge=1, le=20)
    law_filter: Optional[str] = Field(
        default=None,
        json_schema_extra={"example": "Civil Code 2007"},
    )


@router.post("/qa", response_model=LegalQAResponse)
async def answer_question(
    payload: QuestionRequest,
    qa_use_case: AnswerLegalQAUseCase = Depends(get_qa_use_case),
) -> LegalQAResponse:
    """
    Generate a grounded legal answer citing specific articles from Cambodian law.
    """
    req = LegalQARequest(
        question=payload.question,
        top_k=payload.top_k,
        law_filter=payload.law_filter,
    )
    return qa_use_case.execute(req)
