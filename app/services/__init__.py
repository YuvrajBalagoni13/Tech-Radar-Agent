"""
Services package: Ingestion pipelines, Multi-channel Notifiers, Search retrieval, and PDF Compiler.
"""

from app.services.ingestion import IngestionService
from app.services.notifier import (
    BaseNotifier,
    DiscordNotifier,
    TelegramNotifier,
    NotificationService,
)
from app.services.search import TechnicalSearchService
from app.services.pdf_generator import PDFGeneratorService
from app.services.embedding import (
    BaseEmbedder,
    DeterministicEmbedder,
    VoyageEmbedder,
    GeminiEmbedder,
    calculate_cosine_similarity,
    get_embedder,
    get_embedding,
    compare_embeddings,
    compare_embeddings_batch,
)

__all__ = [
    "IngestionService",
    "BaseNotifier",
    "DiscordNotifier",
    "TelegramNotifier",
    "NotificationService",
    "TechnicalSearchService",
    "PDFGeneratorService",
    "BaseEmbedder",
    "DeterministicEmbedder",
    "VoyageEmbedder",
    "GeminiEmbedder",
    "calculate_cosine_similarity",
    "get_embedder",
    "get_embedding",
    "compare_embeddings",
    "compare_embeddings_batch",
]
