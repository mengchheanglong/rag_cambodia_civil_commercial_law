"""Abstract interface for Large Language Models."""

from abc import ABC, abstractmethod

from src.domain.entities import RetrievedDocument


class LLMPort(ABC):
    """
    Port for LLM-based text generation.

    Implementations wrap specific LLM APIs (OpenAI, Anthropic, local models, etc.).
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        context_documents: list[RetrievedDocument],
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate an answer grounded in the provided legal context.

        Args:
            query: The user's legal question.
            context_documents: Retrieved articles to ground the answer.
            system_prompt: Optional system prompt override.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            Generated answer text with article citations.

        Raises:
            GenerationError: If the LLM API call fails.
        """
        ...
