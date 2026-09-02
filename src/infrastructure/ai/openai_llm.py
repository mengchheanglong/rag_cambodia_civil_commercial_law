"""
LLM Generation Adapter (DeepSeek & OpenAI compatible).

Implements LLMPort using OpenAI-compatible API standard:
- Supports DeepSeek Chat (Flash / V3 / V4 fast mode) via 'deepseek-chat'
- Supports DeepSeek Reasoner (Pro / R1 reasoning mode) via 'deepseek-reasoner'
- Supports OpenAI models ('gpt-4o', 'gpt-4o-mini')
- Enforces statutory context grounding and article citation rules.
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
    Adapter for generating grounded legal answers via DeepSeek / OpenAI API.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._settings = settings or get_settings()

        # Determine effective API key
        self._api_key = (
            api_key
            or self._settings.deepseek_api_key
            or self._settings.llm_api_key
            or self._settings.openai_api_key
        )

        # Determine effective Base URL
        if base_url:
            self._base_url = base_url
        elif self._settings.deepseek_api_key or self._settings.llm_provider == "deepseek":
            self._base_url = self._settings.deepseek_base_url or "https://api.deepseek.com"
        elif self._settings.llm_base_url:
            self._base_url = self._settings.llm_base_url
        else:
            self._base_url = None  # default OpenAI endpoint

        # Determine effective model
        self._model = (
            model
            or (self._settings.deepseek_model if self._settings.deepseek_api_key else None)
            or self._settings.llm_model
            or self._settings.openai_llm_model
        )
        self._temperature = self._settings.llm_temperature

        if not self._api_key:
            logger.warning(
                "No LLM API key configured (DEEPSEEK_API_KEY / OPENAI_API_KEY). "
                "OpenAILLM will fail if generate() is called without a key."
            )

        self._client: Optional[OpenAI] = (
            OpenAI(api_key=self._api_key, base_url=self._base_url)
            if self._api_key
            else None
        )
        self.last_reasoning_content: Optional[str] = None

    def generate(
        self,
        query: str,
        context_documents: list[RetrievedDocument],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Generate a cited legal answer grounded strictly in the provided articles.

        Args:
            query: The user question.
            context_documents: Retrieved context articles.
            system_prompt: Optional system prompt override.
            temperature: Sampling temperature override.
            model_override: Optional model override (e.g. 'deepseek-reasoner').

        Returns:
            Generated answer text.
        """
        if not self._client:
            raise GenerationError(
                "LLM client is not initialized. Please set DEEPSEEK_API_KEY or OPENAI_API_KEY in .env."
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
        model_name = model_override or self._model

        try:
            return self._call_chat_completion(sys_prompt, user_content, temp, model_name)
        except Exception as e:
            logger.error("LLM generation failed", error=str(e), model=model_name)
            raise GenerationError(f"LLM generation error ({model_name}): {e}")

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
        model_name: str,
    ) -> str:
        """Execute chat completion with retry (supports DeepSeek Chat & Reasoner)."""
        # DeepSeek Reasoner does not support temperature parameter in some versions
        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if "reasoner" not in model_name.lower():
            kwargs["temperature"] = temperature

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        # Capture reasoning content from DeepSeek R1 / Reasoner if present
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            self.last_reasoning_content = message.reasoning_content
        else:
            self.last_reasoning_content = None

        return message.content.strip() if message.content else ""
