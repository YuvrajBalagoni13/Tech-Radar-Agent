"""
SQLAlchemy ORM Entities for PostgreSQL with pgvector.
Implements UserProfile and ProcessedRelease entities with HNSW indexes for sub-10ms similarity search.
"""

import uuid
from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    DateTime,
    Index,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from pgvector.sqlalchemy import Vector
from app.core.database import Base


class UserProfile(Base):
    """
    User profile entity defining developer technical interests, notification endpoints,
    and the dense 1536-dimensional embedding vector representing their technical radar footprint.
    """

    __tablename__ = "user_profiles"

    user_id = Column(String(64), primary_key=True, index=True)
    discord_id = Column(String(64), unique=True, nullable=True, index=True)
    telegram_chat_id = Column(String(64), unique=True, nullable=True, index=True)
    tracked_domains = Column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    ignored_keywords = Column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    profile_summary = Column(Text, nullable=False, default="", server_default="")
    profile_embedding = Column(Vector(1536), nullable=True)
    similarity_threshold = Column(Float, nullable=False, default=0.82, server_default="0.82")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_user_profiles_profile_embedding_hnsw",
            profile_embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"profile_embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<UserProfile user_id={self.user_id} discord_id={self.discord_id} threshold={self.similarity_threshold}>"


class ProcessedRelease(Base):
    """
    Processed technical release item entity ensuring strict idempotency via SHA-256 content hashing.
    Indexed with HNSW vector search to allow semantic historical deduplication and query lookups.
    """

    __tablename__ = "processed_releases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    source_url = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    release_embedding = Column(Vector(1536), nullable=True)
    ingested_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index(
            "ix_processed_releases_release_embedding_hnsw",
            release_embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"release_embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<ProcessedRelease id={self.id} hash={self.content_hash[:8]} title={self.title[:30]}>"
