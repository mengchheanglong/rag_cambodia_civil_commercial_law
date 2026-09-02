"""
OpenAI Embedding Adapter.

Implements EmbeddingPort using OpenAI's text-embedding-3-large model.
Includes local file caching, batching, and retry logic with tenacity.
"""

import json
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.logging import get_logger
from src.config.settings import Settings, get_settings
from src.domain.exceptions import EmbeddingError
from src.domain.ports.embedding_port import EmbeddingPort

logger = get_logger(__name__)


class OpenAIEmbedding(EmbeddingPort):
    """
    Adapter for generating embeddings via OpenAI's API.

    Features:
    - Batched requests (up to 100 texts per API call)
    - Local disk caching to avoid duplicate API fees
    - Automatic exponential backoff retry logic
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model = self._settings.openai_embedding_model
        self._dim = self._settings.openai_embedding_dimensions

        api_key = self._settings.openai_api_key
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY is not set. OpenAIEmbedding will fail if API is called without a key."
            )

        self._client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None

        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        self._cache_dir = cache_dir or (base_dir / "data" / "indices" / "embedding_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def dimensions(self) -> int:
        """Return embedding dimensionality (e.g. 3072 for text-embedding-3-large)."""
        return self._dim

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding vector for a single query string."""
        results = self.embed([query])
        return results[0]

    def embed(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """
        Generate embeddings for a list of texts in batches.

        Checks local disk cache first before making API calls.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Process a single batch of texts with retry."""
        if not self._client:
            raise EmbeddingError("OpenAI client is not initialized. Please configure OPENAI_API_KEY.")

        try:
            return self._call_openai_with_retry(batch)
        except Exception as e:
            logger.error("OpenAI embedding API call failed", error=str(e))
            raise EmbeddingError(f"OpenAI embedding error: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_openai_with_retry(self, batch: list[str]) -> list[list[float]]:
        """Call OpenAI embedding endpoint with tenacity retry."""
        response = self._client.embeddings.create(
            input=batch,
            model=self._model,
            dimensions=self._dim,
        )
        return [data.embedding for data in response.data]
