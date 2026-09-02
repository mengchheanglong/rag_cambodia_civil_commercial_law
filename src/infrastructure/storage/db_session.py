"""
Database engine and session management.

Provides SQLAlchemy session factory with connection pooling.
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.logging import get_logger
from src.config.settings import Settings, get_settings
from src.infrastructure.storage.models import Base

logger = get_logger(__name__)


def create_db_engine(settings: Settings | None = None):
    """Create SQLAlchemy engine from settings."""
    cfg = settings or get_settings()
    engine = create_engine(
        cfg.database_url,
        pool_size=cfg.database_pool_size,
        max_overflow=cfg.database_max_overflow,
        pool_pre_ping=True,
    )
    return engine


def init_db(engine) -> None:
    """Initialize database tables and pgvector extension."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL / initialize schema: {e}")


def get_session_factory(engine) -> sessionmaker[Session]:
    """Create a sessionmaker bound to the given engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
