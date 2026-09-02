"""
Use Case: Ingest a legal document into the RAG system.

Pipeline: PDF → Extract → Clean → Chunk → Embed → Store
"""

from pathlib import Path

from src.config.logging import get_logger
from src.application.dtos import IngestDocumentRequest, IngestDocumentResponse
from src.domain.ports.extractor_port import ExtractorPort
from src.domain.ports.chunker_port import ChunkerPort
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.ports.sparse_retriever_port import SparseRetrieverPort

logger = get_logger(__name__)


class IngestDocumentUseCase:
    """
    Orchestrates the full document ingestion pipeline.

    Extracts text from a legal PDF, chunks it by article hierarchy,
    generates embeddings, and stores everything in the vector database
    and sparse index.
    """

    def __init__(
        self,
        extractor: ExtractorPort,
        chunker: ChunkerPort,
        embedder: EmbeddingPort,
        vector_store: VectorStorePort,
        sparse_retriever: SparseRetrieverPort,
    ) -> None:
        self._extractor = extractor
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_retriever = sparse_retriever

    def execute(self, request: IngestDocumentRequest) -> IngestDocumentResponse:
        """
        Run the full ingestion pipeline for a single legal document.

        Args:
            request: Ingestion parameters (PDF path, law name, language, pages).

        Returns:
            IngestDocumentResponse with counts of created chunks and embeddings.
        """
        logger.info(
            "Starting document ingestion",
            law_name=request.law_name,
            pdf_path=request.pdf_path,
        )

        # Step 1: Extract text from PDF
        raw_text = self._extractor.extract(
            pdf_path=Path(request.pdf_path),
            start_page=request.start_page,
            end_page=request.end_page,
        )
        logger.info("Text extracted", length=len(raw_text))

        # Step 2: Chunk by legal article hierarchy
        chunks = self._chunker.chunk(
            text=raw_text,
            law_name=request.law_name,
            language=request.language,
        )
        logger.info("Chunks created", count=len(chunks))

        # Step 3: Generate embeddings
        texts_to_embed = [chunk.content_with_context for chunk in chunks]
        embeddings = self._embedder.embed(texts_to_embed)
        logger.info("Embeddings generated", count=len(embeddings))

        # Step 4: Store in vector database
        self._vector_store.store(chunks, embeddings)
        logger.info("Chunks stored in vector database")

        # Step 5: Update sparse index
        self._sparse_retriever.index(chunks)
        logger.info("Sparse index updated")

        return IngestDocumentResponse(
            law_name=request.law_name,
            chunks_created=len(chunks),
            chunks_embedded=len(embeddings),
        )
