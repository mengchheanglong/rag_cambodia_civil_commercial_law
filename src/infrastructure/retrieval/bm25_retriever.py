"""
BM25 Sparse Retriever Adapter.

Implements SparseRetrieverPort using rank-bm25 (BM25Okapi).
Supports indexing, querying, serialization to disk, and legal-specific tokenization.
"""

import pickle
from pathlib import Path
from typing import Optional

import regex as re
from rank_bm25 import BM25Okapi

from src.config.logging import get_logger
from src.domain.entities import LegalChunk, RetrievedDocument
from src.domain.ports.sparse_retriever_port import SparseRetrieverPort

logger = get_logger(__name__)


def tokenize_legal_text(text: str) -> list[str]:
    """
    Tokenize legal text for BM25 keyword matching.

    Preserves numbers (like article numbers "315") and legal terms.
    Normalizes case and removes punctuation.
    """
    cleaned = text.lower()
    # Extract alphanumeric tokens, keeping legal identifiers intact
    tokens = re.findall(r"\b[a-z0-9_\u1780-\u17ff]+\b", cleaned)
    return tokens


class BM25Retriever(SparseRetrieverPort):
    """
    Sparse keyword retriever implementing BM25Okapi algorithm.

    High recall for exact article citations (e.g. 'Article 315')
    and statutory terminology.
    """

    def __init__(self, index_path: Optional[Path] = None) -> None:
        self._chunks: list[LegalChunk] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None

        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        self._index_path = index_path or (base_dir / "data" / "indices" / "bm25_index.pkl")
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        if self._index_path.exists():
            self.load()

    def index(self, chunks: list[LegalChunk]) -> None:
        """
        Build or update the BM25 index from a list of LegalChunk objects.

        Args:
            chunks: Legal chunks to index.
        """
        if not chunks:
            logger.warning("Empty chunk list passed to BM25 index.")
            return

        self._chunks = list(chunks)
        self._corpus_tokens = [
            tokenize_legal_text(chunk.content_with_context) for chunk in self._chunks
        ]
        self._bm25 = BM25Okapi(self._corpus_tokens)

        logger.info(
            "BM25 index built",
            total_documents=len(self._chunks),
            vocab_size=len(self._bm25.idf) if hasattr(self._bm25, "idf") else None,
        )
        self.save()

    def search(self, query: str, top_k: int = 50) -> list[RetrievedDocument]:
        """
        Search the BM25 index for the most relevant chunks.

        Args:
            query: The search query string.
            top_k: Maximum number of candidate results.

        Returns:
            List of RetrievedDocument with sparse_score populated.
        """
        if not self._bm25 or not self._chunks:
            logger.warning("BM25 index is empty. Returning empty results.")
            return []

        query_tokens = tokenize_legal_text(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top_k indices sorted descending by score
        indexed_scores = [(idx, score) for idx, score in enumerate(scores) if score > 0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = indexed_scores[:top_k]

        results: list[RetrievedDocument] = []
        for idx, score in top_matches:
            results.append(
                RetrievedDocument(
                    chunk=self._chunks[idx],
                    sparse_score=float(score),
                )
            )

        return results

    def save(self) -> None:
        """Serialize BM25 index and chunks to disk."""
        try:
            with open(self._index_path, "wb") as f:
                pickle.dump(
                    {
                        "chunks": self._chunks,
                        "corpus_tokens": self._corpus_tokens,
                        "bm25": self._bm25,
                    },
                    f,
                )
            logger.info("Saved BM25 index to disk", path=str(self._index_path))
        except Exception as e:
            logger.error("Failed to save BM25 index", error=str(e))

    def load(self) -> bool:
        """Load BM25 index and chunks from disk if available."""
        try:
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)
                self._chunks = data.get("chunks", [])
                self._corpus_tokens = data.get("corpus_tokens", [])
                self._bm25 = data.get("bm25")
            logger.info(
                "Loaded BM25 index from disk",
                total_documents=len(self._chunks),
                path=str(self._index_path),
            )
            return True
        except Exception as e:
            logger.warning("Could not load BM25 index from disk", error=str(e))
            return False
