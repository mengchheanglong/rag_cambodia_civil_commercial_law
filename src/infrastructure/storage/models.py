"""
SQLAlchemy ORM models for PostgreSQL + pgvector storage.
"""

from datetime import date
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy ORM."""
    pass


class LegalChunkModel(Base):
    """
    SQLAlchemy model representing an embedded legal article chunk.

    Includes:
    - Primary key and text content
    - Full hierarchical statutory metadata
    - pgvector Vector embedding column (default 3072 dimensions)
    """

    __tablename__ = "legal_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    law_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    law_name_kh: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    book: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    chapter: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    article_number: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    article_title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    language: Mapped[str] = mapped_column(String(10), default="en", index=True)
    promulgation_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_with_context: Mapped[str] = mapped_column(Text, nullable=False)

    # 3072-dimensional vector for OpenAI text-embedding-3-large
    embedding: Mapped[list[float]] = mapped_column(Vector(3072), nullable=True)

    # HNSW Index for fast vector similarity search
    __table_args__ = (
        Index(
            "idx_legal_chunks_embedding_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("idx_legal_chunks_lookup", "law_name", "article_number"),
    )
