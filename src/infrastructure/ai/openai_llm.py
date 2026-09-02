"""
OpenAI LLM Generation Adapter.

Implements LLMPort using OpenAI Chat Completions (e.g. GPT-4o / GPT-4o-mini).
Enforces context grounding and structured article citation rules.
"""

from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.logging import get_logger
from src.config.settings import Settings, get_settings
from src.domain.entities import RetrievedDocument
from src.domain.exceptions import GenerationError
from src.domain.ports.llm_port import LLMPort
from src.infrastructure.ai.prompts import (
    LEGAL_QA_SYSTEM_PROMPT,
    LEGAL_QA_USER_TEMPLATE,
    format_context,
)

logger = get_logger(__name__)


class OpenAILLM(LLMPort):
    """
    Adapter for generating grounded legal answers via OpenAI's Chat API.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model = self._settings.openai_llm_model
        self._temperature = self._settings.openai_llm_temperature

        api_key = self._settings.openai_api_key
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY is not set. OpenAILLM will fail if generate() is called without a key."
            )

        self._client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None

    def generate(
        self,
        query: str,
        context_documents: list[RetrievedDocument],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a cited legal answer grounded strictly in the provided articles.

        Args:
            query: The user question.
            context_documents: Retrieved context articles.
            system_prompt: Optional system prompt override.
            temperature: Sampling temperature override.

        Returns:
            Generated answer text.
        """
        if not self._client:
            raise GenerationError(
                "OpenAI client is not initialized. Please set OPENAI_API_KEY in your .env file."
            )

        # Format context articles
        articles_data = [
            {
                "law_name": doc.chunk.metadata.law_name,
                "article_number": doc.chunk.metadata.article_number,
                "content": doc.chunk.content,
            }
            for doc in context_documents
        ]
        formatted_context = format_context(articles_data)

        user_content = LEGAL_QA_USER_TEMPLATE.format(
            question=query,
            context=formatted_context,
        )

        sys_prompt = system_prompt or LEGAL_QA_SYSTEM_PROMPT
        temp = temperature if temperature is not None else self._temperature

        try:
            return self._call_chat_completion(sys_prompt, user_content, temp)
        except Exception as e:
            logger.error("LLM generation failed", error=str(e))
            raise GenerationError(f"OpenAI generation error: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_chat_completion(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float,
    ) -> str:
        """Execute OpenAI chat completion with retry."""
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content.strip()
