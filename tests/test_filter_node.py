"""
Unit and Integration Tests for Three-Stage Domain Filtering Pipeline.
Verifies cryptographic deduplication, zero-cost negative keyword filtering,
and pgvector cosine similarity evaluation.
"""

import math
from unittest.mock import AsyncMock, patch
import pytest
from app.agent.nodes.domain_filter_node import (
    calculate_cosine_similarity,
    domain_filter_node,
    generate_embedding,
)
from app.core.config import settings
from app.agent.state import AgentState
from app.models.schemas import TechReleaseItem
from app.services.ingestion import IngestionService


def test_cosine_similarity_mathematics():
    """Verify standard cosine distance mathematical properties."""
    # Identical vectors => 1.0
    vec_a = [1.0, 2.0, 3.0]
    assert math.isclose(calculate_cosine_similarity(vec_a, vec_a), 1.0, rel_tol=1e-5)

    # Orthogonal vectors => 0.0
    vec_b = [1.0, 0.0, 0.0]
    vec_c = [0.0, 1.0, 0.0]
    assert math.isclose(calculate_cosine_similarity(vec_b, vec_c), 0.0, abs_tol=1e-5)

    # Empty or mismatched vectors => 0.0
    assert calculate_cosine_similarity([], []) == 0.0
    assert calculate_cosine_similarity([1.0], [1.0, 2.0]) == 0.0


@pytest.mark.asyncio
async def test_embedding_generation_normalization():
    """Ensure embedding vectors have 1536 dimensions and unit Euclidean norm."""
    vec = await generate_embedding("Distributed systems consensus protocols via Raft")
    assert len(vec) == settings.EMBEDDING_DIMENSIONS
    norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-4)


def test_sha256_idempotent_hash_calculation():
    """Verify SHA-256 deduplication hashing is deterministic and case-insensitive."""
    url1 = "https://arxiv.org/abs/2401.0001"
    title1 = "FlashAttention-3: Fast and Memory-Efficient Exact Attention"

    hash1 = IngestionService.calculate_content_hash(url1, title1)
    hash2 = IngestionService.calculate_content_hash(f" {url1} ", f"{title1.upper()} ")
    assert hash1 == hash2
    assert len(hash1) == 64


@pytest.mark.asyncio
async def test_filter_stage_2_negative_keywords():
    """Verify Stage 2 drops items containing ignored keywords with 0 LLM cost."""
    from unittest.mock import patch
    item = TechReleaseItem(
        title="Exclusive Crypto NFT Airdrop for Solana Traders",
        source_url="https://hype.crypto/airdrop",
        summary="Claim your free memecoin and NFT tokens instantly.",
        source_type="hackernews",
    )

    state: AgentState = {
        "release_item": item,
        "user_id": "test_user_stage2",
        "relevance_score": 0.0,
        "is_relevant": False,
        "matched_domains": [],
        "filter_reason": "",
        "notification_status": "PENDING",
        "user_prompt_override": None,
        "research_notes": [],
        "tutorial_markdown": None,
        "pdf_artifact_path": None,
        "error_logs": [],
    }

    with patch("app.services.dedup_store.DeduplicationStore.is_seen", return_value=False):
        result = await domain_filter_node(state)

    assert result["is_relevant"] is False
    assert result["relevance_score"] == 0.0
    assert "Stage 2: Matched negative keyword" in result["filter_reason"]


@pytest.mark.asyncio
async def test_filter_stage_3_relevant_item():
    """Verify Stage 3 passes high-affinity distributed systems and AI items."""
    from unittest.mock import patch
    item = TechReleaseItem(
        title="Scalable Vector Databases: HNSW Graph Partitioning under High Concurrency",
        source_url="https://arxiv.org/abs/2409.0002",
        summary="A deep dive into distributed systems, vector databases, and high-throughput HNSW index tuning.",
        source_type="arxiv",
    )

    state: AgentState = {
        "release_item": item,
        "user_id": "test_user_stage3",
        "relevance_score": 0.0,
        "is_relevant": False,
        "matched_domains": [],
        "filter_reason": "",
        "notification_status": "PENDING",
        "user_prompt_override": None,
        "research_notes": [],
        "tutorial_markdown": None,
        "pdf_artifact_path": None,
        "error_logs": [],
    }

    with patch("app.services.dedup_store.DeduplicationStore.is_seen", return_value=False):
        result = await domain_filter_node(state)

    # In synthetic test mode, vector similarity between shared domain tokens should be non-zero
    assert result["relevance_score"] > 0.0
    assert len(result["matched_domains"]) > 0


@pytest.mark.asyncio
async def test_filter_stage_3_incorporates_profile_bio():
    """Verify that a specialized profile bio increases cosine similarity for closely matched topics."""
    from app.agent.nodes.domain_filter_node import generate_embedding, calculate_cosine_similarity
    from app.models.entities import UserProfile
    from app.bot.discord_client import sync_user_embedding

    # Profile with general topics
    user_generic = UserProfile(
        user_id="user_gen",
        tracked_domains=["Machine Learning"],
        profile_summary="",
    )
    await sync_user_embedding(user_generic)

    # Profile with detailed bio matching the target release
    user_specialized = UserProfile(
        user_id="user_spec",
        tracked_domains=["Machine Learning"],
        profile_summary="Senior research engineer working on speculative decoding and FlashAttention kernel optimization for transformers.",
    )
    await sync_user_embedding(user_specialized)

    release_vec = await generate_embedding("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness and Kernel Optimization")
    sim_generic = calculate_cosine_similarity(release_vec, user_generic.profile_embedding)
    sim_specialized = calculate_cosine_similarity(release_vec, user_specialized.profile_embedding)

    # Specialized profile matching words in the release should produce higher cosine similarity
    assert sim_specialized > sim_generic


@pytest.mark.asyncio
async def test_ingestion_service_top_k_unseen_pagination():
    """Verify that IngestionService skips seen items and yields unseen items up to limit."""
    from unittest.mock import AsyncMock, patch
    from app.services.ingestion import IngestionService
    from app.services.dedup_store import DeduplicationStore

    service = IngestionService()

    url1 = "https://news.ycombinator.com/item?id=101"
    title1 = "Seen Article 1"
    h1 = service.calculate_content_hash(url1, title1)

    url2 = "https://news.ycombinator.com/item?id=102"
    title2 = "Seen Article 2"
    h2 = service.calculate_content_hash(url2, title2)

    with patch.object(DeduplicationStore, "is_seen", side_effect=lambda h: h in (h1, h2)):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            from unittest.mock import MagicMock
            def mock_responses(url, *args, **kwargs):
                resp = MagicMock()
                resp.status_code = 200
                resp.raise_for_status.return_value = None
                if "topstories.json" in str(url):
                    resp.json.return_value = [101, 102, 103, 104]
                elif "item/101.json" in str(url):
                    resp.json.return_value = {"type": "story", "title": title1, "url": url1, "time": 1000}
                elif "item/102.json" in str(url):
                    resp.json.return_value = {"type": "story", "title": title2, "url": url2, "time": 1000}
                elif "item/103.json" in str(url):
                    resp.json.return_value = {"type": "story", "title": "Fresh Unseen Article 3", "url": "https://hn.com/3", "time": 1000}
                elif "item/104.json" in str(url):
                    resp.json.return_value = {"type": "story", "title": "Fresh Unseen Article 4", "url": "https://hn.com/4", "time": 1000}
                return resp

            mock_get.side_effect = mock_responses

            items = await service.fetch_hackernews_top(limit=2, only_unseen=True)
            assert len(items) == 2
            assert items[0].title == "Fresh Unseen Article 3"
            assert items[1].title == "Fresh Unseen Article 4"


@pytest.mark.asyncio
async def test_adhoc_research_prevalidation_bypass():
    """Verify that an ad-hoc research item with is_prevalidated=True bypasses Stage 1-3 filters."""
    item = TechReleaseItem(
        title="What Matters in On-Policy Distillation?",
        summary="Empirical study of on-policy RL distillation dynamics.",
        source_url="https://arxiv.org/abs/2609.99999",
        source_type="arxiv",
    )

    state: AgentState = {
        "release_item": item,
        "user_id": "user_adhoc",
        "search_plan": {"arxiv_categories": ["cs.LG", "cs.AI"]},
        "relevance_score": 0.88,
        "is_relevant": True,
        "is_prevalidated": True,
        "matched_domains": ["cs.LG"],
        "filter_reason": "Ad-hoc research match",
        "notification_status": "ESCALATED_TELEGRAM",
        "user_prompt_override": "Ad-hoc research query: distillation",
        "research_notes": [],
        "tutorial_markdown": None,
        "pdf_artifact_path": None,
        "error_logs": [],
    }

    # Even if DB dedup or user profile domains don't match, prevalidated item must pass
    result = await domain_filter_node(state)
    assert result["is_relevant"] is True
    assert result["relevance_score"] == 0.88
    assert "cs.LG" in result["matched_domains"]


@pytest.mark.asyncio
async def test_profile_embedding_caching():
    """Verify that _USER_PROFILE_CACHE caches profile embeddings and reuses them."""
    from app.agent.nodes.domain_filter_node import _USER_PROFILE_CACHE
    from app.models.entities import UserProfile

    user = UserProfile(
        user_id="test_cached_user",
        tracked_domains=["Computer Vision"],
        profile_summary="Diffusion models",
    )

    item = TechReleaseItem(
        title="Diffusion Models for Video",
        summary="Novel video diffusion architecture.",
        source_url="https://arxiv.org/abs/2609.88888",
        source_type="arxiv",
    )

    state: AgentState = {
        "release_item": item,
        "user_id": "test_cached_user",
        "search_plan": None,
        "relevance_score": 0.0,
        "is_relevant": False,
        "matched_domains": [],
        "filter_reason": "",
        "notification_status": "PENDING",
        "user_prompt_override": None,
        "research_notes": [],
        "tutorial_markdown": None,
        "pdf_artifact_path": None,
        "error_logs": [],
    }

    with patch("app.services.dedup_store.DeduplicationStore.is_seen", return_value=False), \
         patch("app.bot.telegram_handler._in_memory_profiles", {"test_cached_user": user}), \
         patch("app.agent.nodes.domain_filter_node.generate_embedding", wraps=generate_embedding) as mock_embed:
        
        # Clear cache for this user
        _USER_PROFILE_CACHE.pop("test_cached_user", None)
        
        # First call: populates cache
        res1 = await domain_filter_node(state)
        assert "test_cached_user" in _USER_PROFILE_CACHE
        calls_first = mock_embed.call_count

        # Second call: should reuse cached profile embedding
        res2 = await domain_filter_node(state)
        # Release item embedding is computed, but profile embedding should not be recomputed
        assert mock_embed.call_count == calls_first + 1


