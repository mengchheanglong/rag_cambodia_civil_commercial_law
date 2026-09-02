"""
Application settings loaded from environment variables.

Configured for DeepSeek Flash LLM and Cambodian Legal Retrieval.
Copy .env.example to .env and fill in your values.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the RAG Cambodia Law application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── DeepSeek LLM (Dedicated Provider) ────────────────────────────
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"  # DeepSeek Flash
    llm_temperature: float = 0.1

    # ── Embeddings (OpenAI / Multilingual) ────────────────────────────
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimensions: int = 3072

    # ── Database ────────────────────────────────────────────────────
    database_url: str = "postgresql+psycopg://user:password@localhost:5432/rag_cambodia_law"
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # ── Retrieval ───────────────────────────────────────────────────
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rerank_top_k: int = 5
    rrf_k: int = 60
    hybrid_weight_dense: float = 0.5
    hybrid_weight_sparse: float = 0.5

    # ── Reranker ────────────────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-large"
    cohere_api_key: str = ""

    # ── Application ─────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    log_format: str = "console"  # "console" or "json"


def get_settings() -> Settings:
    """Factory function to create a Settings instance."""
    return Settings()
