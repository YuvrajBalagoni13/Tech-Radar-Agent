"""Database connection and session setup using asyncpg and pgvector."""

import logging
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from app.core.config import settings

logger = logging.getLogger("techradar.database")

# Base model for SQLAlchemy entities
Base = declarative_base()

# Async engine with connection pooling
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=(settings.LOG_LEVEL.upper() == "DEBUG"),
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for yielding an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create extensions and tables on startup."""
    logger.info("Initializing database extensions and schemas...")
    async with engine.begin() as conn:
        # Enable pgvector and uuid-ossp extensions
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        logger.info("PostgreSQL extensions 'vector' and 'uuid-ossp' verified.")

        # Create all tables defined in Base metadata
        await conn.run_sync(Base.metadata.create_all)
        logger.info("SQLAlchemy entity schemas successfully synchronized.")


async def close_db() -> None:
    """Close engine connection pool on shutdown."""
    logger.info("Disposing of database engine connection pool...")
    await engine.dispose()
    logger.info("Database connection pool closed.")
