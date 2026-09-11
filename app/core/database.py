"""
Database connection pool, engine lifecycle, and pgvector extension management.
Provides asynchronous SQLAlchemy sessions and LangGraph checkpoint persistence connections.
"""

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

# Declarative metadata base for all SQLAlchemy entities
Base = declarative_base()

# Asynchronous engine with enterprise-grade connection pooling
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=(settings.LOG_LEVEL.upper() == "DEBUG"),
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,
)

# Async session factory configured with expire_on_commit=False to avoid lazy-load race conditions
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency injection generator yielding an async SQLAlchemy session.
    Guarantees session cleanup and automatic rollback on unhandled exceptions.
    """
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
    """
    Bootstrap database schema and register pgvector extension.
    Executes during FastAPI application startup lifecycle.
    """
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
    """
    Gracefully dispose of database engine connection pools on shutdown.
    """
    logger.info("Disposing of database engine connection pool...")
    await engine.dispose()
    logger.info("Database connection pool closed.")
