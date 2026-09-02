"""
Embedding & Indexing Pipeline script.

Loads structured article chunks from data/04_chunks/,
builds the BM25 sparse index, generates vector embeddings,
and indexes everything into the vector store.
"""

import json
from pathlib import Path

from src.config.logging import get_logger, setup_logging
from src.config.settings import get_settings
from src.domain.entities import LegalChunk
from src.infrastructure.ai.openai_embedding import OpenAIEmbedding
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever
from src.infrastructure.storage.pgvector_repository import PgVectorRepository

setup_logging()
logger = get_logger(__name__)


def run_embedding_and_indexing() -> None:
    """Load chunks and build both BM25 and Vector indices."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    chunks_dir = base_dir / "data" / "04_chunks"
    settings = get_settings()

    # 1. Load all chunks from data/04_chunks/
    chunk_files = list(chunks_dir.glob("*_chunks.json"))
    if not chunk_files:
        logger.error("No chunk files found in data/04_chunks/. Run 'python -m src.pipeline.chunk' first.")
        return

    all_chunks: list[LegalChunk] = []
    for chunk_file in chunk_files:
        logger.info(f"Loading chunks from {chunk_file.name}...")
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            chunks = [LegalChunk.model_validate(item) for item in data]
            all_chunks.extend(chunks)

    logger.info(f"Total legal article chunks loaded: {len(all_chunks):,}")

    # 2. Build and save BM25 Sparse Index
    logger.info("Building BM25 Sparse Index...")
    bm25 = BM25Retriever()
    bm25.index(all_chunks)
    logger.info("BM25 Sparse Index successfully built and saved to disk.")

    # 3. Dense Vector Embeddings
    vector_repo = PgVectorRepository()

    if not settings.openai_api_key or "your-api-key" in settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set in .env. BM25 sparse search is fully ready! "
            "To generate dense vector embeddings, add your OPENAI_API_KEY to .env and re-run."
        )
        return

    logger.info(
        f"Generating embeddings using OpenAI model '{settings.openai_embedding_model}' ({settings.openai_embedding_dimensions} dims)..."
    )
    embedder = OpenAIEmbedding(settings=settings)
    texts_to_embed = [chunk.content_with_context for chunk in all_chunks]

    # Embed in batches with progress logging
    batch_size = 100
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i : i + batch_size]
        logger.info(f"Embedding batch {i // batch_size + 1}/{(len(texts_to_embed) + batch_size - 1) // batch_size} ({len(batch)} chunks)...")
        batch_embs = embedder.embed(batch)
        all_embeddings.extend(batch_embs)

    logger.info(f"Generated {len(all_embeddings)} embeddings. Storing in vector repository...")
    vector_repo.store(all_chunks, all_embeddings)
    logger.info("Vector repository indexing complete.")


if __name__ == "__main__":
    run_embedding_and_indexing()
