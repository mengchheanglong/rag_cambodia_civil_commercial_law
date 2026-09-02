"""Health and readiness endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "rag_cambodia_law"
    version: str = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse()
