"""Unit tests for embedding services, Voyage AI integration, and comparative evaluations."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.embedding import (
    DeterministicEmbedder,
    VoyageEmbedder,
    GeminiEmbedder,
    calculate_cosine_similarity,
    get_embedder,
    get_embedding,
    compare_embeddings,
)


@pytest.mark.asyncio
async def test_deterministic_embedder_dimensions_and_norm():
    """Verify deterministic embedder produces 1536-dim unit-normalized vector."""
    embedder = DeterministicEmbedder(dimensions=1536)
    vec = await embedder.embed("Distributed Consensus in Byzantine Environments")

    assert len(vec) == 1536
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_deterministic_similarity_behavior():
    """Verify semantic similarity for similar vs unrelated strings."""
    embedder = DeterministicEmbedder(dimensions=1536)

    v1 = await embedder.embed("Raft consensus and Paxos distributed algorithms")
    v2 = await embedder.embed("Paxos distributed algorithms and Raft state machine")
    v3 = await embedder.embed("Baking chocolate cookies with butter and sugar")

    sim_related = calculate_cosine_similarity(v1, v2)
    sim_unrelated = calculate_cosine_similarity(v1, v3)

    assert sim_related > sim_unrelated
    assert sim_related > 0.5


from app.core.config import settings


@pytest.mark.asyncio
async def test_voyage_embedder_missing_api_key():
    """Verify VoyageEmbedder raises ValueError when API key is missing, or falls back if allowed."""
    embedder_strict = VoyageEmbedder(api_key="", allow_fallback=False)
    with pytest.raises(ValueError, match="VOYAGE_API_KEY is not configured"):
        await embedder_strict.embed("Test paper abstract for zero-knowledge proofs")

    embedder_fallback = VoyageEmbedder(api_key="", allow_fallback=True)
    vec = await embedder_fallback.embed("Test paper abstract for zero-knowledge proofs")
    assert len(vec) == settings.EMBEDDING_DIMENSIONS
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_voyage_embedder_mock_api_success():
    """Verify VoyageEmbedder correctly calls Voyage AI API and handles 1024 to 1536 dimension alignment."""
    # Simulate a 1024-dim embedding from Voyage AI
    mock_1024_vec = [0.1] * 1024
    norm = sum(x * x for x in mock_1024_vec) ** 0.5
    mock_1024_vec = [x / norm for x in mock_1024_vec]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"embedding": mock_1024_vec, "index": 0}],
        "model": "voyage-4-lite",
        "usage": {"total_tokens": 12},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        embedder = VoyageEmbedder(api_key="pa-test-key", model="voyage-4-lite", target_dimensions=1536)
        vec = await embedder.embed("Recursive SNARK architecture for rollups", input_type="document")

        assert len(vec) == 1536
        # Cosine similarity between two such padded vectors remains identical
        assert mock_post.called
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["model"] == "voyage-4-lite"
        assert payload["input_type"] == "document"


@pytest.mark.asyncio
async def test_get_embedder_factory_selection():
    """Verify provider switching mechanism."""
    det = get_embedder("deterministic")
    assert isinstance(det, DeterministicEmbedder)

    voyage = get_embedder("voyage")
    assert isinstance(voyage, VoyageEmbedder)


@pytest.mark.asyncio
async def test_compare_embeddings_structure():
    """Verify compare_embeddings returns side-by-side metrics."""
    text_a = "Zero knowledge succinct non-interactive arguments of knowledge"
    text_b = "Recursive zk-SNARKs and cryptographic polynomial commitments"

    mock_vec = [0.1] * settings.EMBEDDING_DIMENSIONS
    norm = sum(x * x for x in mock_vec) ** 0.5
    mock_vec = [x / norm for x in mock_vec]

    with patch.object(VoyageEmbedder, "embed_batch", new_callable=AsyncMock) as mock_voyage:
        mock_voyage.return_value = [mock_vec, mock_vec]
        result = await compare_embeddings(text_a, text_b)

    assert "deterministic" in result
    assert "voyage" in result
    assert "similarity" in result["deterministic"]
    assert "similarity" in result["voyage"]
    assert result["deterministic"]["dimensions"] == settings.EMBEDDING_DIMENSIONS
    assert result["voyage"]["dimensions"] == settings.EMBEDDING_DIMENSIONS
    assert result["voyage"]["model"] == "voyage-4-lite"


@pytest.mark.asyncio
async def test_gemini_embedder_missing_api_key():
    """Verify GeminiEmbedder raises ValueError when API key is missing, or falls back if allowed."""
    embedder_strict = GeminiEmbedder(api_key="", allow_fallback=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        await embedder_strict.embed("Test paper abstract for zero-knowledge proofs")

    embedder_fallback = GeminiEmbedder(api_key="", allow_fallback=True)
    vec = await embedder_fallback.embed("Test paper abstract for zero-knowledge proofs")
    assert len(vec) == settings.EMBEDDING_DIMENSIONS
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_gemini_embedder_mock_api_success():
    """Verify GeminiEmbedder correctly formats batchEmbedContents and aligns 768 to target dimensions."""
    mock_768_vec = [0.05] * 768
    norm = sum(x * x for x in mock_768_vec) ** 0.5
    mock_768_vec = [x / norm for x in mock_768_vec]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embeddings": [{"values": mock_768_vec}],
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        embedder = GeminiEmbedder(
            api_key="AIzaSyTestKey",
            model="text-embedding-004",
            target_dimensions=1536,
        )
        vec = await embedder.embed("Self-supervised representation learning for robotics", input_type="query")

        assert len(vec) == 1536
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-4

        assert mock_post.called
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "requests" in payload
        assert len(payload["requests"]) == 1
        req = payload["requests"][0]
        # Verified auto-aliasing maps legacy text-embedding-004 to active gemini-embedding-001
        assert req["model"] == "models/gemini-embedding-001"
        assert req["taskType"] == "RETRIEVAL_QUERY"
        assert req["content"]["parts"][0]["text"] == "Self-supervised representation learning for robotics"


@pytest.mark.asyncio
async def test_get_embedder_gemini_factory():
    """Verify factory returns GeminiEmbedder for 'gemini' and 'google'."""
    gem = get_embedder("gemini")
    assert isinstance(gem, GeminiEmbedder)

    goog = get_embedder("google")
    assert isinstance(goog, GeminiEmbedder)

