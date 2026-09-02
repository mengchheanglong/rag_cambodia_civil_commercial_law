"""Unit tests for the legal hierarchical chunker."""

from src.infrastructure.chunking.legal_hierarchical_chunker import LegalHierarchicalChunker


class TestLegalHierarchicalChunker:
    """Tests for Article-level chunking with metadata extraction."""

    def setup_method(self) -> None:
        self.chunker = LegalHierarchicalChunker()

    def test_chunks_english_articles(self, sample_article_text_en: str) -> None:
        """Should split text into individual Article chunks."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        assert len(chunks) == 4  # Articles 1, 2, 315, 316

    def test_article_numbers_extracted(self, sample_article_text_en: str) -> None:
        """Each chunk should have the correct article number."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        article_numbers = [c.metadata.article_number for c in chunks]
        assert article_numbers == [1, 2, 315, 316]

    def test_chapter_metadata_tracked(self, sample_article_text_en: str) -> None:
        """Articles should inherit their parent Chapter context."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        # Articles 1 & 2 are in Chapter 1
        assert "CHAPTER 1" in (chunks[0].metadata.chapter or "")
        # Articles 315 & 316 are in Chapter 2
        assert "CHAPTER 2" in (chunks[2].metadata.chapter or "")

    def test_book_metadata_tracked(self, sample_article_text_en: str) -> None:
        """All articles should have Book 4 as parent context."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        for chunk in chunks:
            assert "BOOK 4" in (chunk.metadata.book or "")

    def test_content_with_context_has_prefix(self, sample_article_text_en: str) -> None:
        """content_with_context should include hierarchical prefix."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        assert "[Civil Code 2007]" in chunks[0].content_with_context

    def test_chunk_ids_are_unique(self, sample_article_text_en: str) -> None:
        """Each chunk should have a unique chunk_id."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunks_khmer_articles(self, sample_article_text_kh: str) -> None:
        """Should split Khmer text into individual Article chunks."""
        chunks = self.chunker.chunk(
            text=sample_article_text_kh,
            law_name="ក្រមរដ្ឋប្បវេណី",
            language="kh",
        )
        assert len(chunks) == 2  # មាត្រា ១ and មាត្រា ២

    def test_empty_text_raises_error(self) -> None:
        """Should raise ChunkingError for empty input."""
        import pytest
        from src.domain.exceptions import ChunkingError

        with pytest.raises(ChunkingError):
            self.chunker.chunk(text="", law_name="Test", language="en")
