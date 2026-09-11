"""
Application configuration module powered by Pydantic v2 Settings.
Validates environment variables, connection strings, API tokens, and operational thresholds.
"""

from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration management with type validation and environment loading.
    Conforms to the 12-Factor App methodology for enterprise cloud deployments.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # Application Lifecycle & Networking
    ENVIRONMENT: str = Field(default="development", description="Environment stage: development, test, production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging verbosity level")
    APP_HOST: str = Field(default="0.0.0.0", description="FastAPI host interface binding")
    APP_PORT: int = Field(default=8000, description="FastAPI port binding")

    # PostgreSQL with pgvector Database Settings
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://techradar:techradar_secret@localhost:5432/techradar_db",
        description="Async SQLAlchemy database connection string",
    )
    SYNC_DATABASE_URL: Optional[str] = Field(
        default="postgresql://techradar:techradar_secret@localhost:5432/techradar_db",
        description="Synchronous database connection string for Alembic/LangGraph savers",
    )
    DB_POOL_SIZE: int = Field(default=10, description="SQLAlchemy connection pool maximum base connections")
    DB_MAX_OVERFLOW: int = Field(default=20, description="Max burst connections beyond pool_size")
    DB_POOL_TIMEOUT: int = Field(default=30, description="Timeout seconds waiting for a free connection")

    # Redis Cache & Message Broker
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL for caching and locks")

    # Groq LPU & LLM Providers
    GROQ_API_KEY: str = Field(default="", description="Groq API authentication token (gsk_...)")
    GROQ_API_BASE: str = Field(default="https://api.groq.com/openai/v1", description="Groq OpenAI-compatible API base URL")
    LLM_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Primary generative LLM hosted on Groq LPUs")
    EMBEDDING_DIMENSIONS: int = Field(default=1536, description="Vector space dimensionality for pgvector")

    # Algorithmic & Channel Routing Settings
    PRIMARY_NOTIFICATION_CHANNEL: str = Field(
        default="TELEGRAM",
        description="Primary channel for radar alerts: 'DISCORD', 'TELEGRAM', or 'BOTH'",
    )
    DEFAULT_SIMILARITY_THRESHOLD: float = Field(
        default=0.82,
        description="Cosine similarity threshold for relevance filtering (0.0 to 1.0)",
    )
    ESCALATION_TIMEOUT_HOURS: int = Field(
        default=6,
        description="Hours before unacknowledged Discord alerts escalate to Telegram",
    )
    INGESTION_INTERVAL_MINUTES: int = Field(
        default=60,
        description="Cron/interval schedule in minutes for polling technical feeds",
    )

    # Search & Technical Scrapers
    TAVILY_API_KEY: str = Field(default="", description="Tavily API key for domain-scoped technical search")

    # Discord Bot Integration (Supports direct messages / DMs and channels)
    DISCORD_BOT_TOKEN: str = Field(default="", description="Discord bot authentication token")
    DISCORD_APPLICATION_ID: str = Field(default="", description="Discord Application ID for slash commands")
    DISCORD_PUBLIC_KEY: str = Field(default="", description="Discord Interaction verification public key")
    DISCORD_USER_ID: str = Field(default="", description="Target Discord User Snowflake ID for Direct Message (DM) alerts")
    DISCORD_CHANNEL_ID: str = Field(default="", description="Optional fallback Discord channel ID for radar alerts")

    # Telegram Bot Integration (100% Free Forever)
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Telegram Bot authentication token from @BotFather")
    TELEGRAM_CHAT_ID: str = Field(default="", description="Target Telegram chat ID or channel username for alerts")

    # PDF Compilation & Storage
    PDF_OUTPUT_DIR: str = Field(
        default="./artifacts/pdfs",
        description="Directory path where compiled PDF tutorials are stored",
    )

    @field_validator("DEFAULT_SIMILARITY_THRESHOLD")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("DEFAULT_SIMILARITY_THRESHOLD must be strictly bounded between 0.0 and 1.0")
        return v


# Global singleton settings instance
settings = Settings()
